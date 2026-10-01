"""
Core machinery for notebook 08's user-clustered paired bootstrap of the R2 difference
between two fitted models on the same test set.

Distinct from notebook 05d's multi-seed variance check: 05d asks whether refitting with a
different training seed changes the outcome. This asks whether the observed test-set
performance gap between two SPECIFIC fitted models is distinguishable from what finite-
test-set sampling alone could produce. The two answer different questions and are reported
side by side, not combined into one number.

Method: resample USERS with replacement (not rows), preserving within-user row correlation.
Per-user sufficient statistics (row count, sum of y, sum of y^2, sum of squared residuals for
each of the two models) are precomputed once; each bootstrap iteration reconstructs R2_A,
R2_B, and their difference algebraically from these sums, without touching individual rows.
y is centered (subtract a fixed constant, the uncentered global mean) before accumulating
sums, purely for floating-point numerical stability at F-DATA's scale (Sum(y^2) and
(Sum(y))^2/n can both be huge and nearly equal, losing precision in the subtraction that
computes SS_tot) -- centering does not change R2, since R2 is invariant to a constant shift
in y.
"""
import numpy as np


def compute_per_user_stats(user_ids, y_true, pred_a, pred_b):
    """y_true, pred_a, pred_b must all be in the SAME space (all-raw or all-log) and the
    same row order as user_ids. Returns a dict of per-user arrays (indexed 0..n_users-1) and
    the array of unique user ids in that same order, plus the centering constant used."""
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    center = float(np.mean(y_true))
    y_c = y_true - center

    users, inverse = np.unique(np.asarray(user_ids), return_inverse=True)
    n_users = len(users)

    n_u = np.bincount(inverse, minlength=n_users).astype(float)
    sy_u = np.bincount(inverse, weights=y_c, minlength=n_users)
    sy2_u = np.bincount(inverse, weights=y_c ** 2, minlength=n_users)
    ssa_u = np.bincount(inverse, weights=(y_c - (pred_a - center)) ** 2, minlength=n_users)
    ssb_u = np.bincount(inverse, weights=(y_c - (pred_b - center)) ** 2, minlength=n_users)

    return dict(users=users, n_u=n_u, sy_u=sy_u, sy2_u=sy2_u, ssa_u=ssa_u, ssb_u=ssb_u,
                center=center, n_users=n_users, n_rows=len(y_true))


def r2_from_sufficient_stats(n_total, sy_total, sy2_total, ss_res_total):
    """SS_tot via the computational formula (sum of squares minus the mean-correction term),
    valid regardless of centering since R2 is shift-invariant."""
    ss_tot = sy2_total - (sy_total ** 2) / n_total
    return 1.0 - ss_res_total / ss_tot


def run_cluster_bootstrap(stats, n_iterations=2000, seed=0):
    """Vectorized user-cluster bootstrap. Returns an array of n_iterations values, each
    (R2_A_raw, R2_B_raw) reconstructed from a resample of users drawn with replacement."""
    rng = np.random.default_rng(seed)
    n_users = stats["n_users"]
    idx = rng.integers(0, n_users, size=(n_iterations, n_users))

    r2_a = np.empty(n_iterations)
    r2_b = np.empty(n_iterations)
    for it in range(n_iterations):
        counts = np.bincount(idx[it], minlength=n_users).astype(float)
        n_total = counts @ stats["n_u"]
        sy_total = counts @ stats["sy_u"]
        sy2_total = counts @ stats["sy2_u"]
        ssa_total = counts @ stats["ssa_u"]
        ssb_total = counts @ stats["ssb_u"]
        r2_a[it] = r2_from_sufficient_stats(n_total, sy_total, sy2_total, ssa_total)
        r2_b[it] = r2_from_sufficient_stats(n_total, sy_total, sy2_total, ssb_total)
    return r2_a, r2_b


def run_rowlevel_bootstrap(user_ids, y_true, pred_a, pred_b, n_iterations=2000, seed=0):
    """Brute-force reference implementation: for each iteration, draw users with replacement,
    gather every ROW belonging to each drawn user (with its multiplicity), and compute R2
    directly via sklearn-equivalent arithmetic on the actual resampled rows. Used only as a
    cross-check that the sufficient-statistics reconstruction above is correct -- not run at
    full scale, since it is exactly what the sufficient-statistics approach exists to avoid."""
    from sklearn.metrics import r2_score

    user_ids = np.asarray(user_ids)
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    users = np.unique(user_ids)
    n_users = len(users)
    rows_by_user = {u: np.where(user_ids == u)[0] for u in users}

    rng = np.random.default_rng(seed)
    r2_a = np.empty(n_iterations)
    r2_b = np.empty(n_iterations)
    for it in range(n_iterations):
        drawn = rng.choice(users, size=n_users, replace=True)
        row_idx = np.concatenate([rows_by_user[u] for u in drawn])
        r2_a[it] = r2_score(y_true[row_idx], pred_a[row_idx])
        r2_b[it] = r2_score(y_true[row_idx], pred_b[row_idx])
    return r2_a, r2_b


def top5_user_share(stats):
    """Share of total rows and total sum-of-squared-y held by the 5 heaviest users, by row
    count -- a supposedly user-level bootstrap result is suspect if a handful of users
    dominate it."""
    order = np.argsort(-stats["n_u"])[:5]
    row_share = stats["n_u"][order].sum() / stats["n_rows"]
    # sum-of-squares share uses sy2_u (centered) since that's what drives SS_res/SS_tot
    ss_share = stats["sy2_u"][order].sum() / stats["sy2_u"].sum()
    return float(row_share), float(ss_share)
