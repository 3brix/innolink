"""
Effect sizes for binder vs non-binder separation, per metric.
All scores are direction-aligned via the YAML first, so every effect size points the same way.
Cliff's delta (non-parametric; metrics like docking scores / energies are rarely normal) and Mann-Whitney AUROC + p-value. --> primary
Cohen's d is provided as a secondary check only (normality caveat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from preprocessing.align import get_direction


def _aligned_pos_neg(df, col, directions):
    """Return (positives, negatives) of the direction-aligned score, or None."""
    clean = df[[col, "binder"]].dropna()
    if clean["binder"].nunique() < 2:
        return None
    score = clean[col].values * get_direction(col, directions)
    y = clean["binder"].astype(int).values
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return pos, neg


def compute_auroc_pvalue(df: pd.DataFrame, metrics: list[str], directions: dict[str, int]) -> pd.DataFrame:
    """Mann-Whitney U -> AUROC and (one-sided, pos > neg) p-value per metric."""
    rows = []
    for col in metrics:
        pn = _aligned_pos_neg(df, col, directions)
        if pn is None:
            continue
        pos, neg = pn
        u, p = mannwhitneyu(pos, neg, alternative="greater")
        rows.append({"metric": col, "auroc": round(u / (len(pos) * len(neg)), 4), "p_value": round(float(p), 6)})
    out = pd.DataFrame(rows)
    return out.sort_values("auroc", ascending=False) if "auroc" in out.columns else out


def cliffs_delta(df: pd.DataFrame, metrics: list[str], directions: dict[str, int]) -> pd.DataFrame:
    """
    Cliff's delta per metric: delta = 2 * AUROC - 1, in [-1, 1]. 
    """
    rows = []
    for col in metrics:
        pn = _aligned_pos_neg(df, col, directions)
        if pn is None:
            continue
        pos, neg = pn
        u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
        auroc = u / (len(pos) * len(neg))
        rows.append({"metric": col, "cliffs_delta": round(2 * auroc - 1, 4)})
    out = pd.DataFrame(rows)
    return out.sort_values("cliffs_delta", key=abs, ascending=False) if "cliffs_delta" in out.columns else out


def cohens_d(df: pd.DataFrame, metrics: list[str], directions: dict[str, int]) -> pd.DataFrame:
    """
    Cohen's d per metric (pooled-SD standardized mean difference).
    """
    rows = []
    for col in metrics:
        pn = _aligned_pos_neg(df, col, directions)
        if pn is None:
            continue
        pos, neg = pn
        if len(pos) < 2 or len(neg) < 2:
            continue
        n1, n2 = len(pos), len(neg)
        pooled = np.sqrt(((n1 - 1) * pos.std(ddof=1) ** 2 + (n2 - 1) * neg.std(ddof=1) ** 2) / (n1 + n2 - 2))
        d = (pos.mean() - neg.mean()) / pooled if pooled > 0 else np.nan
        rows.append({"metric": col, "cohens_d": round(float(d), 4)})
    out = pd.DataFrame(rows)
    return out.sort_values("cohens_d", key=abs, ascending=False) if "cohens_d" in out.columns else out