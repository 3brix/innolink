"""
Canonical metric metadata: the single place that reads config/metric_data.yaml and config/thresholds.yaml 
A metric column name is matched to its metadata by exact name or longest suffix.

"""

from __future__ import annotations

import yaml
import pandas as pd

from config.analysis import EXCLUDE_COLUMNS, FILTER_ONLY_CATEGORIES
from config.paths import METRIC_YAML, THRESHOLDS_YAML

# Feature-column selection
def get_metric_columns(df: pd.DataFrame, exclude_categories=FILTER_ONLY_CATEGORIES) -> list[str]:
    """Returns the numeric columns. Excludes meta/label/experimental columns and the filtering-only categories """
    numeric = df.select_dtypes(include="number").columns
    cols = [c for c in numeric if c not in EXCLUDE_COLUMNS]
    if exclude_categories:
        cats = load_categories()
        cols = [c for c in cols if get_category(c, cats) not in exclude_categories]
    return cols

# Metadata loaders (one YAML read each, longest-key-first)
def _load_metric_field(yaml_path, field: str) -> dict:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    raw = {name: info.get(field) for name, info in data["metrics"].items()}
    return dict(sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True))


def load_directions(yaml_path=METRIC_YAML) -> dict[str, int]:
    """{metric: +1/-1} directionality."""
    return _load_metric_field(yaml_path, "direction")


def load_families(yaml_path=METRIC_YAML) -> dict[str, str]:
    """{metric: family}."""
    return _load_metric_field(yaml_path, "family")


def load_categories(yaml_path=METRIC_YAML) -> dict[str, str]:
    """{metric: category} (confidence / interface / sequence / developability / energy / ...)."""
    return _load_metric_field(yaml_path, "category")


def load_scales(yaml_path=METRIC_YAML) -> dict[str, str]:
    """{metric: scale} (confidence / error / energy / developability / ...)."""
    return _load_metric_field(yaml_path, "scale")


def load_thresholds(path=THRESHOLDS_YAML) -> dict:
    """{family: threshold} from thresholds.yaml; null entries dropped."""
    data = yaml.safe_load(open(path)) or {}
    thr = data.get("thresholds", data)
    return {k: v for k, v in thr.items() if v is not None}


# Column -> metadata (exact / longest-suffix match)
def _match_suffix(col: str, mapping: dict, default=None):
    for base, val in mapping.items():
        if col == base or col.endswith(f"_{base}"):
            return val
    return default


def get_direction(col: str, directions: dict[str, int]) -> int:
    """Return +1 / -1 for a column; default +1."""
    return _match_suffix(col, directions, default=1)


def get_family(col: str, families: dict[str, str]) -> str | None:
    """Return a column's family via exact / longest-suffix match; None if unknown."""
    return _match_suffix(col, families, default=None)


def get_category(col: str, categories: dict[str, str]) -> str | None:
    """Return a column's category via exact / longest-suffix match; None if unknown."""
    return _match_suffix(col, categories, default=None)
