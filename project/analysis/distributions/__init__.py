"""Metric distribution and separability analysis (raw data only)."""
 
from .wide import (
    get_numeric_metrics,
    add_binder_type,
    make_long_scores,
)
from .long import (
    assign_model,
    get_model_groups,
    load_metric_metadata,
    enrich_long_scores,
)
from .plots import (
    plot_metric_distribution,
    plot_metric_separability,
    plot_model_agreement,
)
 
__all__ = [
    "get_numeric_metrics",
    "add_binder_type",
    "make_long_scores",
    "assign_model",
    "get_model_groups",
    "load_metric_metadata",
    "enrich_long_scores",
    "plot_metric_distribution",
    "plot_metric_separability",
    "plot_model_agreement",
]
 