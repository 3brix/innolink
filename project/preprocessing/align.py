
from __future__ import annotations

import yaml
import pandas as pd

from preprocessing.metric_meta import get_metric_columns


def load_metric_directions(yaml_path) -> dict[str, int]:
    """Load metric directionality from the YAML, sorted longest-key-first (to prevent duplicates for example in cases like esmfold / esmfold2)"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    raw = {name: info["direction"] for name, info in data["metrics"].items()}
    return dict(sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True))


def get_direction(col: str, directions: dict[str, int]) -> int:
    """Return +1 / -1 for a column; default +1."""
    for base, direction in directions.items():
        if col == base or col.endswith(f"_{base}"):
            return direction
    return 1


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