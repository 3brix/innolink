"""Wide (one row per sample) table helpers: metric selection and the wide -> long reshape."""

from __future__ import annotations

import pandas as pd
from preprocessing.metric_meta import get_metric_columns, get_all_metric_columns
from preprocessing.metadata import add_binder_class  # noqa: F401  -- re-exported


def get_numeric_metrics(df: pd.DataFrame) -> list[str]:
    """The predictor set: numeric metrics minus the filter-only categories.

    Alias for metric_meta.get_metric_columns. Descriptive analyses want get_all_numeric_metrics.
    """
    return get_metric_columns(df)


def get_all_numeric_metrics(df: pd.DataFrame) -> list[str]:
    """Every numeric metric, filter-only categories included (descriptive analyses)."""
    return get_all_metric_columns(df)


def make_long_scores(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Melt to sample | source | type | binder | binder_class | metric | value."""
    return df.melt(
        id_vars=["sample", "source", "type", "binder", "binder_class"],
        value_vars=metrics, var_name="metric", value_name="value",
    )