"""Combine several metrics into one score per sample.

Names match analysis.composite.cv, so a column in composite_scores.csv and a row in
composite_cv_eval.csv naming the same rule ARE the same rule:
  consensus -- equal-weight mean of the z-scores
  weighted  -- mean weighted by per-source mean ap_norm
  product   -- geometric mean of Phi(z); high only if high on all (the primary rule)
"""


from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from config.paths import METRIC_YAML
from analysis.evaluation.metrics import load_metric_families, top_metrics_distinct_family
from analysis.evaluation.redundancy import spearman_correlation

def select_decorrelated_metrics(rankings, df, top_n=20, method="family", corr_threshold=0.9, metric_yaml=METRIC_YAML):
    """Complementary subset of the top-N by PR-AUC.

    method="family": best metric per family. method="correlation": greedy best-first, keeping a
    metric only if its Spearman with every kept metric is below `corr_threshold`.
    """
    ranked = rankings.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False)
    candidates = ranked["metric"].head(top_n).tolist()

    if method == "family":
        top = ranked.head(top_n)
        families = load_metric_families(metric_yaml)
        return top_metrics_distinct_family(top, families, top_n=len(top))["metric"].tolist()

    # NaN = too few overlapping samples; treat as 'not known to be redundant'
    corr = spearman_correlation(df, candidates).abs().fillna(0.0)
    keep = []
    for m in candidates:
        if all(corr.loc[m, k] < corr_threshold for k in keep):
            keep.append(m)
    return keep


def _weights(rankings, cols, column="ap_norm_ds_mean"):
    """Weights from the per-source mean ap_norm, not pooled PR-AUC (Appendix_composite_selection
    A.2: pooled PR-AUC also responds to between-source score offsets). Falls back to pr_auc for a
    single-source set; below no-skill on average -> weight 0."""
    src = column if column in rankings.columns else "pr_auc"
    w = rankings.set_index("metric")[src].reindex(cols).fillna(0.0).to_numpy(float)
    return np.clip(w, 0.0, None)


def mean_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Equal-weight mean of the columns; NaNs skipped per row."""
    return df[cols].mean(axis=1)


def weighted_mean_score(df: pd.DataFrame, cols: list[str], weights) -> pd.Series:
    """Weighted mean of the columns; NaNs skipped and weights renormalised."""
    vals = df[cols].to_numpy(float)
    w = np.asarray(weights, float)
    if w.sum() == 0:
        return df[cols].mean(axis=1)
    mask = ~np.isnan(vals)
    wsum = (mask * w).sum(axis=1)
    num = np.nansum(np.where(mask, vals, 0.0) * w, axis=1)
    out = np.divide(num, wsum, out=np.full(len(df), np.nan), where=wsum > 0)
    return pd.Series(out, index=df.index)


def product_score(df: pd.DataFrame, cols: list[str], reference: pd.DataFrame | None = None,
                  eps: float = 1e-9) -> pd.Series:
    """Geometric mean of Phi(z): a soft-AND "high on ALL metrics" ranking score.

    A ranking score, NOT a calibrated probability -- it ignores inter-metric correlation.

    'reference' are the rows mu/sd are fitted on. None fits on `df` itself, which makes each
    score depend on which rows are in the table; pass the labelled subset instead (see CLAUDE.md
    on the scaler default).
    """
    X = df[cols].to_numpy(float)
    R = X if reference is None else reference[cols].to_numpy(float)
    mu = np.nanmean(R, axis=0)
    sd = np.nanstd(R, axis=0)
    sd = np.where(sd > 0, sd, 1.0)          # zero-variance column -> leave centred (z = 0)
    z = (X - mu) / sd
    p = norm.cdf(z)
    logp = np.log(np.clip(p, eps, 1.0))     # NaN stays NaN and is ignored below
    with np.errstate(invalid="ignore"):
        gm = np.exp(np.nanmean(logp, axis=1))
    return pd.Series(gm, index=df.index)


def build_composites(df, cols, rankings=None, methods=("consensus", "weighted", "product"),
                     reference=None):
    """Add composite score columns to a copy of df, named as cv.evaluate_composites reports them."""
    out = df.copy()
    if "consensus" in methods:
        out["composite_consensus"] = mean_score(df, cols)
    if "weighted" in methods and rankings is not None:
        out["composite_weighted"] = weighted_mean_score(df, cols, _weights(rankings, cols))
    if "product" in methods:
        out["composite_product"] = product_score(df, cols, reference=reference)
    return out