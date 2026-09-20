"""Shared paired-holdout test harness (cluster bootstrap + the target default).

Extracted 2026-09-20 (PROGRESS.md S25) after `target="direct"` turned out to
give a WRONG answer for a feature (METAR weather, S24): rejected under direct
(+4.86s, P(worse)=0.917), adopted under mixed (-1.21s, P(worse)=0.013) --
LIRF alone swung 33s between the two. `direct` is the fast day-to-day target;
`mixed` (flip for LIRF, direct everywhere else) is what actually ships (S11).
Every feature test from here on defaults to DEFAULT_TARGET so this can't
recur by a copy-pasted `target="direct"` going unnoticed.

Note what this does NOT change: LightGBM pools all 10 airports into one
model, so re-parameterising LIRF's reconstruction changes shared splits and
the early-stopping curve -- the other 9 airports are not independently
verified by a mixed rerun, they're unchanged by construction (see S25 #1:
the 9 non-LIRF per-airport deltas were byte-identical between a direct-only
and a mixed run of the same weather treatment). What a mixed rerun actually
tests is LIRF specifically, and the shared early-stopping curve that comes
from fitting both regressors together.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TARGET = "mixed"


def cluster_bootstrap(
    se_base_sum: np.ndarray, se_treat_sum: np.ndarray, n_by_cluster: np.ndarray,
    n_resamples: int = 3000, seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired cluster bootstrap on the RMSE delta itself (resampled per draw,
    not a linear MSE approximation). Returns (point, ci_lo, ci_hi, p_worse).
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(se_base_sum)
    n_tot = n_by_cluster.sum()
    point = (se_treat_sum.sum() / n_tot) ** 0.5 - (se_base_sum.sum() / n_tot) ** 0.5
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n_clusters, n_clusters)
        n_b = n_by_cluster[idx].sum()
        rmse_b_base = (se_base_sum[idx].sum() / n_b) ** 0.5
        rmse_b_treat = (se_treat_sum[idx].sum() / n_b) ** 0.5
        boots[b] = rmse_b_treat - rmse_b_base
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_worse = float((boots > 0).mean())
    return point, lo, hi, p_worse
