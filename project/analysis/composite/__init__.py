"""Composite scoring: combine several single metrics into one score per sample."""

from .combine import (
    select_decorrelated_metrics,
    mean_score,
    weighted_mean_score,
    product_score,
    build_composites,
)
from .filtering import (
    quality_filter,
)
from .cv import (
    per_source_metrics,
    dataset_aware_select,
    stability_select,
    cv_composite_scores,
    evaluate_composites,
    selection_votes,
)

__all__ = [
    "select_decorrelated_metrics",
    "mean_score",
    "weighted_mean_score",
    "product_score",
    "build_composites",
    "quality_filter",
    "per_source_metrics",
    "dataset_aware_select",
    "stability_select",
    "selection_votes",
    "cv_composite_scores",
    "evaluate_composites",
]

