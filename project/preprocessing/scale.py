"""
Two methods:

  standard : (x - mean)   / std   (sklearn StandardScaler; unit variance)
  robust   : (x - median) / IQR   (sklearn RobustScaler; outlier / skew resistant)

Both are affine per column -> monotonic, so per-metric rankings are unchanged;
scaling only changes cross-metric commensurability. NaNs are ignored on fit and
preserved on transform; a zero-variance / zero-IQR column maps to all-zeros.
Meta columns pass through untouched.

Scaling is dataset-dependent, so for a pooled set fit on the pool. 
It composes with align/normalize by scaling the corresponding variant table; 
align and any of these scalers commute up to sign, so `*_scaled_aligned` is the same whichever order it is built.
"""


from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from preprocessing.metric_meta import get_metric_columns

SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def scale_metrics(df: pd.DataFrame, metrics: list[str], method: str = "standard") -> pd.DataFrame:
    """Standardize the listed metric columns by `method` ('standard' | 'robust')."""
    if method not in SCALERS:
        raise ValueError(f"unknown scaler method {method!r}; use one of {list(SCALERS)}")
    out = df.copy()
    out[metrics] = SCALERS[method]().fit_transform(out[metrics].to_numpy(dtype=float))
    return out


def scale_dataframe(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """Standardize every numeric metric column of `df`. Entry point for run_scale."""
    return scale_metrics(df, get_metric_columns(df), method)
