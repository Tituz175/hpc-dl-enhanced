"""
Sequence construction for notebook 06's LSTM/TCN input, plus the leakage test that verifies
it. Used by notebook 06's sequence-construction investigation and by the LSTM/TCN training
code that followed it.

Decided from that investigation (see notebook 06's sequence-construction section):
  - sequence length = 10 (SEQ_LEN below)
  - policy = zero-pad + boolean mask for every job, never exclude a user or a job -- this
    keeps LSTM/TCN's row population identical to FNN/classical ML's, so no row-population
    caveat is needed when comparing across architectures for any of the six combinations.

A job's sequence input is its user's SEQ_LEN most recent jobs strictly before it (by submit
time), oldest first, left-padded with zero rows where fewer than SEQ_LEN exist. The mask is
True for a real (non-padding) position, False for padding.
"""
import numpy as np
import pandas as pd

SEQ_LEN = 10


def build_sequence_index(user_ids, times, tie_policy="exclude"):
    """Returns, for every row i in the input (in its ORIGINAL row order), the array of row
    indices (into the same input) making up that row's sequence (oldest first, -1 for
    padding), and the corresponding boolean mask (True = real, False = padding).

    user_ids, times: array-like, same length, any row order (not required to be pre-sorted).

    tie_policy:
      - "exclude" (default, unchanged from every prior release of this function): a same-user
        candidate exactly tied with the target job's own submit time is excluded from that one
        slot, which then becomes padding -- no further lookback is attempted for that slot. This
        is what the no-leakage check and the padding-reality numbers in
        `06a_sequence_investigation.ipynb` describe.
      - "skip_back": for each job, look back PAST any same-user jobs that tie its own submit
        time, taking the SEQ_LEN most recent same-user jobs with a STRICTLY earlier submit time,
        regardless of how many tied rows sit in between. Two jobs that tie each other (but not
        the target) are both still eligible history for a later target. This exists to measure
        whether the "exclude" policy's padding loss (see 06a) is itself what limits LSTM/TCN, as
        a PM100-only variant -- see the module docstring's decision section, which still
        describes "exclude" as the adopted default.

    Returns: seq_idx (n_rows, SEQ_LEN) int array, mask (n_rows, SEQ_LEN) bool array.
    """
    assert tie_policy in ("exclude", "skip_back"), f"unknown tie_policy: {tie_policy!r}"
    n = len(user_ids)
    df = pd.DataFrame({"user_id": np.asarray(user_ids), "time": pd.to_datetime(times),
                        "orig_row": np.arange(n)})
    df = df.sort_values(["user_id", "time"], kind="mergesort").reset_index(drop=True)

    seq_idx = np.full((n, SEQ_LEN), -1, dtype=np.int64)
    mask = np.zeros((n, SEQ_LEN), dtype=bool)

    orig_row = df["orig_row"].to_numpy()
    user_arr = df["user_id"].to_numpy()
    times_sorted = df["time"].to_numpy()
    # position within each user's own chronologically-sorted job list (0-indexed), vectorized
    pos_in_user = df.groupby("user_id", sort=False).cumcount().to_numpy()

    if tie_policy == "exclude":
        # Count back from the target's own sorted position -- unchanged from every prior
        # release of this function.
        ref_pos = np.arange(n)
        pos_in_user_at_ref = pos_in_user
        check_time = True
    else:
        # Count back from the START of the target's own same-(user, time) block instead of
        # from the target itself. Since df is sorted ascending by (user, time), every row
        # strictly before that block start is automatically strictly earlier than the target
        # (sharing the target's own time is exactly what put them in the same block), and every
        # row in or after it is not -- so no per-k time comparison is needed, only the block
        # boundary. This is the one piece of new logic "skip_back" adds; everything after it is
        # the same k-step-back loop "exclude" already uses, just counting from a different
        # reference position.
        same_as_prev = np.zeros(n, dtype=bool)
        same_as_prev[1:] = (user_arr[1:] == user_arr[:-1]) & (times_sorted[1:] == times_sorted[:-1])
        block_start_raw = np.where(same_as_prev, np.nan, np.arange(n))
        ref_pos = pd.Series(block_start_raw).ffill().to_numpy().astype(np.int64)
        pos_in_user_at_ref = pos_in_user[ref_pos]
        check_time = False

    for k in range(1, SEQ_LEN + 1):
        # the k-th most recent prior job, in this user's sorted order, is k positions back
        # from `ref_pos` -- valid only where pos_in_user_at_ref >= k (that many same-user rows
        # exist strictly before the reference position). Under "exclude", ref_pos is the
        # target's own position, so an extra STRICT time check is needed: some same-user jobs
        # share an identical submit timestamp (e.g. batch submissions), so a same-user row k
        # positions back in a stable ascending sort can have a time EQUAL to the target's
        # (never later, since the sort is ascending) -- that tie must be excluded, not treated
        # as "strictly prior", or the leakage check below correctly fails. Under "skip_back",
        # ref_pos is already the start of the target's own tied block, so this is guaranteed by
        # construction and no per-k check is needed.
        src_sorted_pos = ref_pos - k
        valid = pos_in_user_at_ref >= k
        if check_time:
            valid &= np.where(pos_in_user_at_ref >= k, times_sorted[np.clip(src_sorted_pos, 0, None)],
                               times_sorted[0]) < times_sorted
        dest_col = SEQ_LEN - k  # most recent prior job goes in the last real slot before target
        rows_valid = np.where(valid)[0]
        seq_idx[orig_row[rows_valid], dest_col] = orig_row[src_sorted_pos[rows_valid]]
        mask[orig_row[rows_valid], dest_col] = True

    return seq_idx, mask


def verify_no_leakage(user_ids, times, seq_idx, mask, sample_size=None, seed=0):
    """Explicit, real test (not an assumption): for every checked row, every REAL (mask=True)
    sequence position must (a) belong to the same user as the target row, (b) have a strictly
    earlier time than the target row, and (c) not be the target row itself. Raises AssertionError
    with a concrete counterexample on the first violation found; returns the number of rows and
    (row, position) pairs checked on success."""
    user_ids = np.asarray(user_ids)
    times = pd.to_datetime(pd.Series(times)).to_numpy()
    n = len(user_ids)

    rows = np.arange(n)
    if sample_size is not None and sample_size < n:
        rows = np.random.default_rng(seed).choice(n, size=sample_size, replace=False)

    n_checked_positions = 0
    for i in rows:
        for k in range(SEQ_LEN):
            if not mask[i, k]:
                continue
            j = seq_idx[i, k]
            assert j != i, f"row {i}: sequence position {k} points to itself"
            assert user_ids[j] == user_ids[i], (
                f"row {i}: sequence position {k} (row {j}) belongs to user {user_ids[j]}, "
                f"target row's user is {user_ids[i]}")
            assert times[j] < times[i], (
                f"row {i} (time {times[i]}): sequence position {k} (row {j}) has time "
                f"{times[j]}, which is not strictly before the target")
            n_checked_positions += 1
    return len(rows), n_checked_positions
