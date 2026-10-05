"""Metric evaluation: rankings, effect sizes, and redundancy structure."""

from .metrics import calculate_all_metrics, top_metrics_distinct_family, load_metric_families, get_family, pass_mask, precision_recall_at, per_fold_metrics, fold_summary
from .effect_sizes import compute_auroc_pvalue, cliffs_delta, cohens_d
from .redundancy import (
    prepare_metric_matrix,
    metric_matrix,
    spearman_correlation,
    label_metrics,
    pca_embedding,
    umap_embedding,
)

__all__ = [
    "calculate_all_metrics",
    "pass_mask",
    "precision_recall_at",
    "per_fold_metrics",
    "fold_summary",
    "top_metrics_distinct_family",
    "load_metric_families",
    "get_family",
    "compute_auroc_pvalue",
    "cliffs_delta",
    "cohens_d",
    "prepare_metric_matrix",
    "metric_matrix",
    "spearman_correlation",
    "label_metrics",
    "pca_embedding",
    "umap_embedding",
    "plot_top_metrics_bar",
    "plot_top_metrics_bar_by_dataset",
    "add_ap_norm",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_precision_ranking",
    "plot_metric_comparison_grid",
    "plot_quadrant",
    "plot_embedding",
    "plot_spearman_heatmap",
    "plot_spearman_heatmap_index",
    "plot_class_distributions",
]


# ---------------------------------------------------------------------------
# Plotting is imported LAZILY. Pipeline stages import this package for its analysis functions and
# must not pay for matplotlib/seaborn: a plain `from .plots import ...` here made every stage
# depend on them, which is how run_composite once died with "No module named seaborn" in an
# environment that had scikit-learn but no plotting stack.
#
# `from analysis.evaluation import plot_x` still works -- it just triggers the
# import at that moment (PEP 562).
# ---------------------------------------------------------------------------
_PLOT_NAMES = ['plot_top_metrics_bar', 'plot_top_metrics_bar_by_dataset', 'add_ap_norm', 'plot_metric_scores', 'plot_roc_curves', 'plot_pr_curves', 'plot_precision_ranking', 'plot_metric_comparison_grid', 'plot_quadrant', 'plot_embedding', 'plot_spearman_heatmap', 'plot_spearman_heatmap_index', 'plot_class_distributions']


def __getattr__(name):
    if name in _PLOT_NAMES:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + _PLOT_NAMES)
