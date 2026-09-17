from __future__ import annotations

import pandas as pd

from config.analysis import EXCLUDE_COLUMNS


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric metric columns (excludes the meta columns)."""
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if c not in EXCLUDE_COLUMNS]