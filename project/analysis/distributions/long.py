"""Long-table helpers for the separability / agreement plots: model grouping + YAML metadata."""

from __future__ import annotations

import pandas as pd
from config.analysis import MODELS
from config.paths import METRIC_YAML
from preprocessing.metric_meta import load_families, load_categories, load_scales


def assign_model(metric: str, models: list[str] = MODELS) -> str | None:
    """Map a metric column to its model via the longest matching prefix."""
    matches = [m for m in models if metric.startswith(m)]
    return max(matches, key=len) if matches else None


def get_model_groups(metrics: list[str]) -> dict[str, list[str]]:
    """Group metrics by model, one model per metric."""
    groups: dict[str, list[str]] = {model: [] for model in MODELS}
    for metric in metrics:
        model = assign_model(metric)
        if model is not None:
            groups[model].append(metric)
    return groups


def load_metric_metadata(yaml_path=METRIC_YAML) -> dict[str, dict]:
    """Per-metric family / category / scale, via the metric_meta loaders."""
    fam, cat, scl = load_families(yaml_path), load_categories(yaml_path), load_scales(yaml_path)
    return {name: {"family": fam.get(name), "category": cat.get(name), "scale": scl.get(name)}
            for name in fam}


def enrich_long_scores(long_df, metadata, models=MODELS) -> pd.DataFrame:
    """Attach metric_family, category, scale and model columns to a long frame."""
    long_df = long_df.copy()
    long_df["metric_family"] = long_df["metric"].map(lambda m: metadata.get(m, {}).get("family"))
    long_df["category"] = long_df["metric"].map(lambda m: metadata.get(m, {}).get("category"))
    long_df["scale"] = long_df["metric"].map(lambda m: metadata.get(m, {}).get("scale"))
    long_df["model"] = long_df["metric"].map(lambda m: assign_model(m, models))
    return long_df