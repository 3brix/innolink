"""
Long-table helpers for the separability / agreement plots.

Metric -> model grouping and the YAML-metadata enrichment that organizes the long score table. Plots in plots.py.
"""

from __future__ import annotations

import yaml
import pandas as pd
from config.analysis import MODELS


def assign_model(metric: str, models: list[str] = MODELS) -> str | None:
    """
    Map a metric column to its model via the longest matching prefix.
    """
    matches = [m for m in models if metric.startswith(m)]
    return max(matches, key=len) if matches else None


def get_model_groups(metrics: list[str]) -> dict[str, list[str]]:
    """
    Group metrics by model, assigning each metric to exactly one model.
    """
    groups: dict[str, list[str]] = {model: [] for model in MODELS}
    for metric in metrics:
        model = assign_model(metric)
        if model is not None:
            groups[model].append(metric)
    return groups


def load_metric_metadata(yaml_path) -> dict[str, dict]:
    """
    Load per-metric metadata (family / category / scale) from the metric YAML.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return {
        name: {"family": info.get("family"), "category": info.get("category"), "scale": info.get("scale")}
        for name, info in data["metrics"].items()
    }


def enrich_long_scores(long_df, metadata, models=MODELS) -> pd.DataFrame:
    """
    Attach "metric_family" and "model" columns to a long dataframe.
    """
    long_df = long_df.copy()
    long_df["metric_family"] = long_df["metric"].map(lambda m: metadata.get(m, {}).get("family"))
    long_df["category"] = long_df["metric"].map(lambda m: metadata.get(m, {}).get("category"))
    long_df["scale"] = long_df["metric"].map(lambda m: metadata.get(m, {}).get("scale"))
    long_df["model"] = long_df["metric"].map(lambda m: assign_model(m, models))
    return long_df