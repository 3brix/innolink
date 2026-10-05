"""precision@pct (point estimate) and the precision@k primitive it is built on.

Every function takes a direction-aligned score (higher = better) and a binary label; missing
scores are dropped. The top-fraction cut is max(1, int(N*pct)).
"""

from __future__ import annotations

import numpy as np


def _clean(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    m = ~np.isnan(s) & ~np.isnan(y)
    return y[m].astype(int), s[m]


def precision_at_k(y_true, scores, k: int) -> tuple[float, int, int]:
    """Precision among the top k by score -> (precision, n_binders_in_top, k_used)."""
    y, s = _clean(y_true, scores)
    if s.size == 0:
        return (np.nan, 0, 0)
    k_used = int(min(k, s.size))
    top = np.argsort(-s, kind="stable")[:k_used]     # highest scores first, stable ties
    n_pos = int(y[top].sum())
    return (n_pos / k_used, n_pos, k_used)


def top_k_at_percent(n: int, pct: float) -> int:
    """Items in the top `pct` fraction of `n`: max(1, int(n * pct))."""
    return max(1, int(n * pct))


def precision_at_percent(y_true, scores, pct: float) -> tuple[float, int, int]:
    """Precision among the top `pct` fraction -> (precision, n_binders_in_top, k_used)."""
    y, s = _clean(y_true, scores)
    if s.size == 0:
        return (np.nan, 0, 0)
    return precision_at_k(y, s, top_k_at_percent(s.size, pct))