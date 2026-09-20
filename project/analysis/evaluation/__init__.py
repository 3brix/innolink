"""Metric evaluation: rankings, effect sizes, and redundancy structure."""

from .metrics import calculate_all_metrics, top_metrics_distinct_family, load_metric_families, get_family, pass_mask, precision_recall_at, select_threshold
from .effect_sizes import compute_auroc_pvalue, cliffs_delta, cohens_d
from .redundancy import (
    prepare_metric_matrix,
    spearman_correlation,
    label_metrics,
    pca_embedding,
    umap_embedding,
)
from .plots import (
    plot_top_metrics_bar,
    plot_roc_curves,
    plot_pr_curves,
    plot_precision_ranking,
    plot_metric_comparison_grid,
    plot_quadrant,
    plot_embedding,
    plot_spearman_heatmap,
    plot_class_distributions,
)

__all__ = [
    "calculate_all_metrics",
    "pass_mask",
    "precision_recall_at",
    "select_threshold",
    "top_metrics_distinct_family",
    "load_metric_families",
    "get_family",
    "compute_auroc_pvalue",
    "cliffs_delta",
    "cohens_d",
    "prepare_metric_matrix",
    "spearman_correlation",
    "label_metrics",
    "pca_embedding",
    "umap_embedding",
    "plot_top_metrics_bar",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_precision_ranking",
    "plot_metric_comparison_grid",
    "plot_quadrant",
    "plot_embedding",
    "plot_spearman_heatmap",
    "plot_class_distributions",
]