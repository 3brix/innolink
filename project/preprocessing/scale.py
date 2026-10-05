"""Per-column scaling: 'standard' (mean/std) or 'robust' (median/IQR).

Both are affine per column, so per-metric rankings are unchanged -- only cross-metric
commensurability. NaNs are ignored on fit and preserved on transform; a zero-variance column
maps to all-zeros. Meta columns pass through. Fit on the pool for a pooled set.
"""


from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from preprocessing.metric_meta import get_metric_columns

# in final pipeline StandardScaler is used (RobustScaler retained as alternative)

SCALERS = {"standard": StandardScaler, "robust": RobustScaler}  
def fit_scaler(reference: pd.DataFrame, metrics: list[str], method: str = "standard"):
    """Fit one scaler on the reference table (normally merged_aligned.csv).

    One shared fit is what makes merged / eval / design comparable; see CLAUDE.md on the
    scaler default."""
    if method not in SCALERS:
        raise ValueError(f"unknown scaler method {method!r}; use one of {list(SCALERS)}")
    scaler = SCALERS[method]()
    scaler.fit(reference[metrics].to_numpy(dtype=float))
    return scaler


def apply_scaler(df: pd.DataFrame, metrics: list[str], scaler) -> pd.DataFrame:
    """Apply a fitted scaler. Missing columns are added as NaN to keep the reference's columns."""
    out = df.copy()
    values = out.reindex(columns=metrics).to_numpy(dtype=float)
    out[metrics] = scaler.transform(values)
    return out


def scale_metrics(df: pd.DataFrame, metrics: list[str], method: str = "standard") -> pd.DataFrame:
    """Fit and apply on one table. Only correct when `df` is the only table being scaled."""
    return apply_scaler(df, metrics, fit_scaler(df, metrics, method))


def scale_dataframe(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """Standardize every numeric metric column of `df`, fit on `df` itself."""
    return scale_metrics(df, get_metric_columns(df), method)
