"""
Ranking-oriented benchmark metrics: precision@10% (point estimate) and the underlying
precision@k primitive.

For a prioritisation task the decision is "rank the designs, take the top fraction", so
the benchmark is described by rank-based, threshold-free quantities: PR-AUC (overall,
primary) and precision@10% (of the top 10% of the ranked list, how many are binders).
The top-fraction cut is max(1, int(N*pct)) (see top_k_at_percent), matching the RF
notebook. precision_at_k is the underlying primitive used by precision_at_percent.

Both are reported as POINT ESTIMATES. The benchmark is small and dominated by deep
mutational scans (many non-independent variants of one parent), so an i.i.d. row bootstrap
would understate the variance and give falsely narrow CIs. A lineage-grouped (cluster)
bootstrap -- resampling parents, not rows -- is the correct uncertainty estimate and is
left as future work (see the "To Test" note in metrics.py).

All functions take a DIRECTION-ALIGNED score (higher = better binder) and the binary
label; missing scores are dropped.
"""

from __future__ import annotations

import numpy as np


def _clean(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    m = ~np.isnan(s) & ~np.isnan(y)
    return y[m].astype(int), s[m]


def precision_at_k(y_true, scores, k: int) -> tuple[float, int, int]:
    """Precision among the top-k by score. Returns (precision, n_binders_in_topk, k_used).

    Ties are broken deterministically (stable sort), so the value is reproducible.
    k_used = min(k, n available); precision is n_binders_in_topk / k_used."""
    y, s = _clean(y_true, scores)
    if s.size == 0:
        return (np.nan, 0, 0)
    k_used = int(min(k, s.size))
    top = np.argsort(-s, kind="stable")[:k_used]     # highest scores first, stable ties
    n_pos = int(y[top].sum())
    return (n_pos / k_used, n_pos, k_used)


def top_k_at_percent(n: int, pct: float) -> int:
    """Number of items in the top `pct` fraction of `n` (>=1). Matches the notebook's
    precision@10% definition: max(1, int(n * pct))."""
    return max(1, int(n * pct))


def precision_at_percent(y_true, scores, pct: float) -> tuple[float, int, int]:
    """precision@`pct` = precision among the top pct-fraction of the ranked list.
    Returns (precision, n_binders_in_top, k_used)."""
    y, s = _clean(y_true, scores)
    if s.size == 0:
        return (np.nan, 0, 0)
    return precision_at_k(y, s, top_k_at_percent(s.size, pct))
