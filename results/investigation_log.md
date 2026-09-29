# Investigation log: diagnostics with no persisted numeric artifact

This logs real investigations that shaped decisions in `EXPERIMENT_TRACKER.md` but that
`results_ledger.csv` cannot carry a row for, because no script and/or no saved result exists
to source a precise number from. Excluding them from the ledger (rather than guessing a
number from memory or from rounded tracker prose) is the same discipline the ledger itself
uses elsewhere -- this file is where that discipline's casualties get written down instead of
silently dropped.

## 1. F-DATA power OOM crash: what actually happened, and what early stopping actually bought

**The crash itself** (2026-09-23 13:19:04, verified directly from `dmesg` in the transcript,
not inferred): the kernel OOM-killer terminated PID 2482138 (`session-1051.scope`, the
`papermill_05c_run2.log` process -- notebook 05c's own chained-kernel Optuna run for RandomForest/
XGBoost/LightGBM in one long-lived process). Confirmed genuine and system-wide, not a cgroup
limit: `oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,task=python3,pid=2482138`,
`Killed process 2482138 (python3) ... anon-rss:84636492kB`. The notebook's own `DeadKernelError:
Kernel died` traceback and file mtime (13:19:00) agree.

**What was also running at that exact moment, verified from a `ps` snapshot taken later the
same day (15:23:36 MDT) and cross-checked against each process's own log timestamps:** a
separate, deliberately-launched `lgbm_isolation_test.log` process (PID 3183574, a full
"LightGBM-only Optuna search, fresh process, no RF/XGBoost run before it") had been running
since 07:57:30 that morning -- already ~5.4 hours in by 13:19:04, confirmed still alive at
15:23:36 (elapsed 27,656s ≈ 7.7h). A third process, `lgbm_perfit_diagnostic.log` (PID 3311360),
had started around 13:03 and was still in its own data-loading phase (its first completed
measurement isn't logged until 13:25:02) at the moment of the 13:19:04 kill. So the crash
landed while **three** memory-heavy Python processes were concurrently active on this
machine -- the original chained run (killed), a multi-hour isolation-test search, and a
just-started diagnostic mid-load -- on a machine that was also running Zoom at the time
(visible in the same `dmesg`/process listing), not a dedicated headless compute box.

**The per-fit diagnostic (PID 3311360, later reported here as 1799.5s for the "mid" point)
was not itself crashed or killed by the OOM.** It survived the 13:19:04 event and continued
running. Once its concurrent memory footprint (together with the isolation test, ~82-84GB RSS
each, ~167GB combined against only ~68GB available and swap at 95%) was recognized as another
imminent OOM risk later that afternoon, **the diagnostic was deliberately, cleanly stopped**
(`kill 3311360 3311363`, confirmed via the transcript's own tool-use record) to protect the
still-running isolation test, which was judged the more important process to preserve. This
was a controlled shutdown, not a second crash.

**The 99.8%-reduction headline this project reported earlier overstated early stopping's
real effect, and the idle-machine re-measurement below shows why -- corrected twice now.**
The original 1799.5s "mid" figure was measured while the isolation test's competing Optuna
search was concurrently running (confirmed above) -- genuine multi-process contention, not a
property of the fit itself. A first idle-machine re-measurement used a zero-filled placeholder
for the 80 embedding-PCA columns instead of real data, which was methodologically wrong, not
just a disclosed simplification: a constant column gives LightGBM's histogram builder a single
degenerate bin, trivially cheap to split on, so that run's 2.7s/16.6s figures understated the
real cost and have been discarded. Re-measured properly using the REAL saved tuning matrices
(`fdata_power_prepared.joblib`'s `X_tune_train`/`X_tune_val`, the genuine 800k/200k-row, 93-column
matrices with the actual fitted embedding PCA, not a reconstruction) on a confirmed-idle machine
(`free -h`/`ps aux` checked immediately before launch: 245GB available, 0B swap, no other
lgbm/xgb/papermill process running; see
`scripts/fdata_power_calibrated_baseline/idle_machine_earlystop_baseline.py`, saved to
`data/interim/fdata_power_idle_machine_earlystop_baseline.joblib`):

| Point | No early stopping (idle, real data) | Early stopped (2026-09-26, session-reported) | Original contended measurement |
|---|---|---|---|
| MID (`n_est=200`, `num_leaves=100`) | **6.27s** (0.0314s/round), all 200 rounds built, peak RSS 11.86 GB | 4.1s, halted at round 89/200 | 1799.5s (contended) |
| EXPENSIVE (`n_est=500`, `num_leaves=255`) | **16.42s** (0.0328s/round), all 500 rounds built, peak RSS 11.90 GB | 4.0s, halted at round 45/500 | not separately measured |

On an idle machine, with real data, building the full 200 rounds takes **6.27 seconds** --
nowhere close to the 1799.5s originally reported, but this time genuinely *slower* than the
4.1s early-stopped run at that same point: early stopping now shows a real, if modest, benefit
at BOTH points (mid: 6.27s → 4.1s, ~35% saving; expensive: 16.42s → 4.0s, ~76% saving), not
just at the expensive one as the zero-filled version had wrongly suggested. The 99.8%
reduction figure is still overwhelmingly a contention artifact of the moment it was measured
in (6.27s and 1799.5s are not the same order of magnitude by a factor of ~290), not a real
property of "wasted rounds" at this scale -- that part of the correction holds. What's
different this time: early stopping's *real* benefit is a few seconds per fit, genuinely
present, just far smaller than the original 1795.4s figure claimed.

**What this investigation has NOT established, stated plainly rather than implied away**: these
are per-fit measurements at two specific hyperparameter points. They do not add up to, and were
never used to reconstruct, the original crashed run's full LightGBM tuning *stage* duration
(on the order of a day, within the ~30-hour total crash) -- that would require summing real
per-trial costs across however many Optuna trials LightGBM actually ran before the OOM, and
that trial-by-trial record was never logged in the original attempt. **The cause of the
original multi-hour LightGBM tuning stage's full duration was not isolated by this
investigation.** What was isolated, precisely: two individual fits are fast (single-digit to
tens of seconds) even without early stopping, once contention is removed -- which rules out
"an individual fit is intrinsically slow" as the stage-level explanation, but does not supply
a positive one in its place.

**Conclusion, corrected**: the crash's real driver was genuine, concurrent multi-process
memory pressure on a shared desktop machine (three heavy Python processes plus Zoom, at the
moment of a global OOM) -- not primarily "wasted LightGBM boosting rounds" as a CPU-time
problem at the level of an individual fit. Adding early stopping was reasonable, defensible
engineering hygiene with a real, modest, now-correctly-measured benefit at both points tested,
and it caps worst-case memory from runaway tree construction on some future higher-complexity
trial -- but it was not the singular, fully-explanatory fix the original "root-caused, not
thread contention" framing claimed, and the full multi-hour stage-level mechanism remains
open. The process-architecture fix (one fresh `systemd-run` process per model, instead of one
long-lived chained kernel accumulating multiple models' state) is the change most directly
responsible for this target's F-DATA power tuning completing cleanly afterward, independent of
whichever mechanism actually drove the original stage's full duration.

## 2. F-DATA power, XGBoost `tree_method="hist"` catastrophic-underfit bug: root-cause isolation

**No scripts exist for these checks at all** -- they were run ad hoc (inline, one-off) during
the investigation, not saved as files in the session scratch directory the way every other
diagnostic in this project was. The numbers below are **session-reported** (same transcript and
same caveat as section 1 above -- read from the live session record, not from any file this repo
tracks, and not independently recomputable without refitting). Four checks, in order, all
against the same baseline: XGBoost tuned at `best_params`, `tree_method="hist"`, full 16.3M-row
train split, raw-space R² ≈ 0.31 (reproduced twice: 0.3094 on the original run and again,
unchanged, after the float64 recast below):

1. **Hyperparameter-independence check**: a different, untuned XGBoost config
   (`n_estimators=300, max_depth=8, learning_rate=0.05`) also scored **R²≈0.34 -- assistant
   narration, not a located raw tool output.** Unlike the other three checks below (each
   traced to a literal `print()` line in a tool_result), this figure comes from the
   assistant's own prose summary of the result in the transcript ("lands around R²≈0.34");
   the underlying raw command output that would have printed the exact value was not found
   on a direct search of the transcript. Treat this one number as a paraphrase, not a
   verbatim-sourced figure -- the qualitative conclusion (a completely different, untuned
   config also failed under `hist`, ruling out "bad Optuna params") is still well-supported,
   just not to the same numeric precision as the other three checks.
2. **Float32/float64 precision check**: recasting the feature matrix to float64,
   R²=0.3094 exactly, unchanged from the baseline -- ruling out a precision/rounding
   explanation (`hist` re-casts to float32 internally regardless of input dtype, so this was
   never going to matter).
3. **Embedding-columns check**: XGBoost fit on the 13 non-embedding columns only,
   R²=0.2168 (elapsed 236.1s) -- comparably catastrophic to the full-feature baseline's ≈0.31,
   if anything slightly *worse*, not better -- ruling out the 80 embedding-PCA columns as the
   cause, since removing them didn't fix (or even improve) the score. (A stray log file,
   `xgb_noembed_test.log`, survived in the scratch directory from this check, but with no
   corresponding script and no joblib, its contents were not independently re-verifiable from
   a repo artifact either way -- the number above comes from the transcript directly, not that
   log file.)
4. **`tree_method` sweep**: `approx` R²=0.3129 (elapsed 373.4s) -- same failure mode as `hist`;
   `exact` R²=0.9888 (elapsed 3502.6s) -- recovered full accuracy, becoming the adopted
   final-fit configuration (see `results_ledger.csv`, `fdata/power` XGBoost, `tree_method=exact`).

All four are genuine, real checks -- not fabricated -- but none of them produced a persisted,
precise numeric artifact in this repo (only the transcript). `fdata_power_xgb_result.joblib`'s
own `note` field independently states the same finding in rounded prose ("R2 ~0.31 vs 0.99"),
which is what the ledger build script cites and excludes on its own terms (see there for why
that specific rounded figure, distinct from these session-reported ones, doesn't clear the bar
either).

## 3. PM100 duration: embedding-style field check

Part of the bounded investigation pass before closing PM100 duration's R²=0.6994 as final
(alongside the scripted, joblib-backed 200-trial extended search, which IS in the ledger as
`ext200_trials`/`investigated_rejected`). This second check -- confirming PM100 has no
embedding-style field comparable to F-DATA's SBert-embedding column -- is a structural/schema
fact (yes/no), not a metric value, so it was never going to be a ledger row regardless of
whether a script exists for it. No script was saved; it was a direct column-listing check
against PM100's schema, run inline.

## 4. F-DATA power's calibrated analytical baseline: recovery

This one **is** in `results_ledger.csv` now -- logged here only for how it got there.
`src/roofline.py` has implemented `calibrate_fdata_power_model`/`predict_fdata_power` (F-DATA's
analog to PM100's calibrated power model) since early in this project, and `notebooks/05c`'s own
markdown states the model was found feasible. But no cell in `05c` ever called it or reported
its metrics -- checked directly (no `calibrate_fdata_power_model` reference anywhere in the
notebook's committed cells). The result had, in fact, been computed once, during the original
investigation: a real raw tool output survives in this project's own Claude Code session
transcript (coeffs `p_idle=107.98663207789151, gamma=-112.62417703748197`, test
`R2=0.9775433474047817, MAPE=514.4758112932082`), and was even briefly visible in an
intermediate notebook draft's comparison table -- but that draft was never the version saved,
and the joblib was never written.

`scripts/fdata_power_calibrated_baseline/fit_and_evaluate.py` closes the gap: it reproduces the
exact `05c` pipeline (cells 6-16: load → filter → sanitize → `handle_msza_sentinel` → 70/30
chronological split), calibrates on the training-period-only split, evaluates on test, and
saves to `data/interim/fdata_power_calibrated_baseline.joblib`. Run 2026-09-28 -- reproduced the
transcript's figures with **zero deviation** on all four values (R², MAPE, `p_idle`, `gamma`),
confirming the original computation was both real and correct; OLS calibration is fully
deterministic, so exact reproduction is the expected (and achieved) result, not a coincidence.
Train/test row counts also matched `fdata_power_prepared.joblib` exactly (16,315,866 /
6,992,514). This joblib is now a genuine, recomputation-verifiable source -- see
`results_ledger.csv`, `fdata/power`, model `CalibratedFDataPowerModel`.
