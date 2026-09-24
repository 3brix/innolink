
from __future__ import annotations

import pandas as pd

# re-export canonical metric metadata (function in preprocessing.metric_meta)
from preprocessing.metric_meta import (
    get_metric_columns,
    load_directions as load_metric_directions,
    get_direction,
)


def align_metrics(df: pd.DataFrame, metrics: list[str], directions: dict[str, int]) -> pd.DataFrame:
    """Flip each listed metric by its direction so higher == better."""
    out = df.copy()
    for col in metrics:
        out[col] = out[col] * get_direction(col, directions)
    return out


def align_dataframe(df: pd.DataFrame, metric_yaml) -> pd.DataFrame:
    """Align every numeric metric column by its YAML direction."""
    metrics = get_metric_columns(df)
    directions = load_metric_directions(metric_yaml)
    return align_metrics(df, metrics, directions)