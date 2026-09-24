"""
Everything here operates on the wide, one-row-per-sample merged table: numeric-metric selection, binder labeling, and the wide -> long reshape for the separability / agreement plots.
"""

from __future__ import annotations

import pandas as pd
from preprocessing.metric_meta import get_metric_columns


def get_numeric_metrics(df: pd.DataFrame) -> list[str]:
    """Numeric prediction-metric columns (excludes meta/label/experimental columns).

    Alias for the analysis layer -- the single implementation lives in preprocessing.metric_meta.get_metric_columns,
    so preprocessingand analysis can't drift apart.
    """
    return get_metric_columns(df)


def make_long_scores(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Convert wide score dataframe into long format for plotting.
    Required columns: sample, source, type, binder, binder_type
    Output: sample | source | type | binder | binder_type | metric | value
    """
    return df.melt(
        id_vars=["sample", "source", "type", "binder", "binder_type"],
        value_vars=metrics, var_name="metric", value_name="value",
    )