"""
Sweep orchestrator: runs all 18 (combination, architecture) pairs across the 4 GPUs, one
pair per GPU at a time, launching the next queued pair on a GPU as soon as it frees up.
Each pair is its own uniquely-named, detached systemd .service (CUDA_VISIBLE_DEVICES set per
slot, absolute uv path -- a bare --unit= service does not inherit the interactive shell's
PATH), and this orchestrator treats `systemctl --user is-active` as the only trustworthy
completion signal, never the launching call's own exit status -- the same lesson learned the
hard way during an earlier timing checkpoint (a systemd scope can keep running after the
harness reports the launching command "killed").

Memory safety, two layers. The real root cause of the first two OOM rounds was NOT inter-pair
concurrency -- it was `run_sweep_pair.py` itself: every Optuna trial's call to build_loaders()
created brand-new DataLoaders with num_workers=4/persistent_workers=True, and the PREVIOUS
trial's worker processes were never torn down first. `ps aux` during the second crash showed a
SINGLE (combo, arch) pair running nine separate ~56GB-RSS processes (one main + two full sets
of four workers), each a near-complete duplicate of the parent rather than sharing memory.
Fixed at the source: num_workers=0 in run_sweep_pair.py. A solo post-fix measurement confirmed
the real single-pair footprint: one process, ~58GB RSS, no duplication.
  1. Any F-DATA pair, any architecture, is capped at MAX_FDATA concurrent (3 x ~58GB ~= 174GB,
     leaving real headroom on this machine's 251GB) -- not a heavy/light-by-architecture
     distinction (that distinction undercounted FNN's real cost before the num_workers fix,
     and is unnecessary now that a single pair's footprint is known directly rather than
     estimated). PM100 pairs are tiny regardless of architecture and aren't capped by this rule.
  2. A real `free -m` check before every launch regardless of the cap above -- the actual
     safety net, since a model's precise footprint during the transient sequence-construction
     phase (before training settles into steady state) isn't the same as its steady-state RSS.
The queue scan skips past pairs that would violate either check to find a runnable one
instead of leaving a GPU idle while a launchable pair waits later in the queue.

Retry-after-failure fix: a prior version's retry path called systemd-run with the SAME unit
name as a pair whose previous attempt had exited into "failed" state -- systemd refuses to
start a transient unit under a name that still exists in any state (active, failed, or
otherwise) until it's cleared, so the retry launch itself failed, and the pair was silently
dropped (replaced by the next queued pair, never actually relaunched). Fixed: `systemctl
--user reset-failed <unit>` before every retry attempt, and a requeue (not a drop) if the
retry launch itself still fails.

Run via: systemd-run --user --unit=notebook06-orchestrator -- bash -c '... uv run python -u
scripts/notebook06_deep_learning/orchestrate_sweep.py > /tmp/notebook06_orchestrator.log 2>&1'
(the orchestrator itself should also be detached -- it will run for many hours)
"""
import os
import pathlib
import re
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
# An absolute path is required (not a bare "uv"): a systemd --user transient unit does not
# inherit the interactive shell's PATH, so the launched service can't resolve a bare command
# name. Resolved from this process's own PATH at startup (where it's launched, that PATH does
# include uv) rather than hardcoded to one user's home directory; UV_BIN overrides it directly.
UV = os.environ.get("UV_BIN") or shutil.which("uv")
if not UV:
    raise RuntimeError("Could not find the 'uv' executable on PATH -- set UV_BIN to its "
                        "absolute path.")
N_GPUS = 4
MAX_FDATA = 3
POLL_SECONDS = 60
MAX_RETRIES = 3
MIN_AVAILABLE_MB = 40_000

QUEUE = [
    ("fdata_memory", "LSTM"), ("fdata_duration", "LSTM"), ("fdata_power", "LSTM"),
    ("fdata_memory", "TCN"), ("fdata_duration", "TCN"), ("fdata_power", "TCN"),
    ("fdata_memory", "FNN"), ("fdata_duration", "FNN"), ("fdata_power", "FNN"),
    ("pm100_power", "LSTM"), ("pm100_duration", "LSTM"), ("pm100_memory_c4", "LSTM"),
    ("pm100_power", "TCN"), ("pm100_duration", "TCN"), ("pm100_memory_c4", "TCN"),
    ("pm100_power", "FNN"), ("pm100_duration", "FNN"), ("pm100_memory_c4", "FNN"),
]

RESULTS_DIR = ROOT / "results"


def is_fdata(combo, arch):
    return combo.startswith("fdata")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [orchestrator] {msg}", flush=True)


def unit_name(combo, arch):
    return f"notebook06-sweep-{combo}-{arch.lower()}"


def is_active(unit):
    r = subprocess.run(["systemctl", "--user", "is-active", f"{unit}.service"],
                        capture_output=True, text=True)
    return r.stdout.strip() == "active"


def reset_failed(unit):
    subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.service"],
                    capture_output=True, text=True)


def available_mb():
    r = subprocess.run(["free", "-m"], capture_output=True, text=True)
    line = r.stdout.splitlines()[1].split()  # Mem: total used free shared buff/cache available
    return int(line[6])


def result_exists(combo, arch):
    return (RESULTS_DIR / f"notebook06_sweep_result_{combo}_{arch}.csv").exists()


def discover_running_pairs():
    """Resume support: finds any notebook06-sweep-* units already ACTIVE (e.g. this
    orchestrator restarting after the previous instance was stopped or crashed, while the
    actual training jobs it launched -- detached systemd units, independent of the launching
    process -- kept running). Parses (combo, arch, gpu) back out of each unit's own
    ExecStart command line rather than assuming anything about how this orchestrator run got
    started."""
    r = subprocess.run(["systemctl", "--user", "list-units", "notebook06-sweep-*",
                         "--all", "--plain", "--no-legend"], capture_output=True, text=True)
    running = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[3] != "running":
            continue
        unit = parts[0]
        show = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ExecStart",
                                "--value"], capture_output=True, text=True).stdout
        m = re.search(r"CUDA_VISIBLE_DEVICES=(\d+).*run_sweep_pair\.py (\S+) (\S+)", show)
        if m:
            gpu, combo, arch = int(m.group(1)), m.group(2), m.group(3)
            running[gpu] = (combo, arch)
            log(f"discovered already-running pair from a prior orchestrator instance: "
                f"{combo}/{arch} on gpu{gpu} (unit {unit})")
    return running


def launch(combo, arch, gpu):
    unit = unit_name(combo, arch)
    log_path = f"/tmp/notebook06_sweep_{combo}_{arch}.log"
    cmd = (f"cd {ROOT} && CUDA_VISIBLE_DEVICES={gpu} {UV} run python -u "
           f"scripts/notebook06_deep_learning/run_sweep_pair.py {combo} {arch} "
           f"> {log_path} 2>&1")
    r = subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--",
                         "bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAILED to launch {combo}/{arch} on gpu{gpu}: {r.stderr.strip()}")
        return False
    log(f"launched {combo}/{arch} on gpu{gpu} ({'fdata' if is_fdata(combo, arch) else 'pm100'}, "
        f"unit={unit}, log={log_path})")
    return True


def pop_next_runnable(queue, fdata_count):
    """Scans the queue for the first item launchable under the heavy-count cap, removing and
    returning it without disturbing the order of the rest. None if nothing qualifies."""
    for i, (combo, arch) in enumerate(queue):
        if not is_fdata(combo, arch) or fdata_count < MAX_FDATA:
            return queue.pop(i)
    return None


def try_fill_slot(queue, fdata_count, gpu):
    """Attempts to launch the next runnable queued pair onto `gpu`. Checks real available
    memory FIRST (the actual safety net), then the heavy-count cap. Returns the (possibly
    unchanged) fdata_count and whether anything was launched. Never loses a popped item: puts
    it back at the front of the queue if memory is short or the launch call itself fails."""
    avail = available_mb()
    if avail < MIN_AVAILABLE_MB:
        log(f"gpu{gpu} slot free but only {avail}MB available (< {MIN_AVAILABLE_MB}MB "
            f"threshold) -- leaving idle this round")
        return fdata_count, False

    item = pop_next_runnable(queue, fdata_count)
    if item is None:
        return fdata_count, False
    combo, arch = item
    if launch(combo, arch, gpu):
        if is_fdata(combo, arch):
            fdata_count += 1
        return fdata_count, (combo, arch)
    queue.insert(0, item)
    return fdata_count, False


queue = list(QUEUE)
slots = {g: None for g in range(N_GPUS)}
retry_counts = {}
done = []
failed = []
combo_progress = {}
fdata_count = 0

log(f"starting sweep: {len(queue)} pairs queued, {N_GPUS} GPUs, max {MAX_FDATA} concurrent "
    f"F-DATA jobs (any architecture), min {MIN_AVAILABLE_MB}MB available required per launch")

# Resume support: drop anything already done from the queue, and anything already running
# (from a prior orchestrator instance) gets its slot restored rather than relaunched.
already_running = discover_running_pairs()
for gpu, (combo, arch) in already_running.items():
    slots[gpu] = (combo, arch)
    if is_fdata(combo, arch):
        fdata_count += 1
    if (combo, arch) in queue:
        queue.remove((combo, arch))

queue = [(c, a) for c, a in queue if not result_exists(c, a)]
for c, a in list(QUEUE):
    if result_exists(c, a):
        done.append((c, a))
        combo_progress.setdefault(c, set()).add(a)
log(f"after resume check: {len(done)} already done, {len(already_running)} already running, "
    f"{len(queue)} still queued")

for gpu in range(N_GPUS):
    if slots[gpu] is not None:
        continue  # occupied by an already-running pair discovered above
    fdata_count, launched = try_fill_slot(queue, fdata_count, gpu)
    if launched:
        slots[gpu] = launched

while any(slots.values()) or queue:
    time.sleep(POLL_SECONDS)
    for gpu in range(N_GPUS):
        if slots[gpu] is None:
            continue
        combo, arch = slots[gpu]
        unit = unit_name(combo, arch)
        if is_active(unit):
            continue

        if is_fdata(combo, arch):
            fdata_count -= 1

        if result_exists(combo, arch):
            log(f"{combo}/{arch} on gpu{gpu} finished successfully (unit {unit} inactive, "
                f"results CSV present)")
            done.append((combo, arch))
            combo_progress.setdefault(combo, set()).add(arch)
            if combo_progress[combo] == {"FNN", "LSTM", "TCN"}:
                log(f"*** COMBO FINISHED: {combo} (all 3 architectures done) ***")
            slots[gpu] = None
        else:
            n_retries = retry_counts.get((combo, arch), 0)
            if n_retries < MAX_RETRIES:
                retry_counts[(combo, arch)] = n_retries + 1
                log(f"WARNING: {combo}/{arch} on gpu{gpu} exited without producing results "
                    f"(unit {unit} inactive, no results CSV) -- retry {n_retries + 1}/"
                    f"{MAX_RETRIES}, relaunching (Optuna's persistent storage resumes from "
                    f"whatever trials already completed)")
                reset_failed(unit)
                avail = available_mb()
                if avail < MIN_AVAILABLE_MB:
                    log(f"  only {avail}MB available -- deferring retry, requeuing instead "
                        f"of relaunching immediately")
                    queue.insert(0, (combo, arch))
                elif launch(combo, arch, gpu):
                    if is_fdata(combo, arch):
                        fdata_count += 1
                    continue  # slot stays assigned to the same pair, skip the refill below
                else:
                    log(f"ERROR: retry launch itself failed for {combo}/{arch} -- requeuing "
                        f"for a later attempt instead of dropping it")
                    queue.insert(0, (combo, arch))
                slots[gpu] = None
            else:
                log(f"ERROR: {combo}/{arch} failed {MAX_RETRIES} times, giving up on this pair "
                    f"-- moving on, needs manual investigation")
                failed.append((combo, arch))
                slots[gpu] = None

        if slots[gpu] is None:
            fdata_count, launched = try_fill_slot(queue, fdata_count, gpu)
            if launched:
                slots[gpu] = launched

log(f"ALL PAIRS PROCESSED. Done: {len(done)}/{len(QUEUE)}. Failed: {failed}")
