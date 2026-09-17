"""
Sample-level & outlier views over the scored metrics (pure viz; driven from
profiles_plots.ipynb). Cross-metric figures expect the STANDARDIZED, aligned
table; per_dataset / outliers read RAW values. See plots.py.
"""

from analysis.profiles.plots import (
    usable_metrics,
    class_colors,
    plot_sample_clustermap,
    plot_filter_heatmap,
    load_thresholds,
    plot_sample_profiles,
    plot_per_dataset,
    plot_outliers,
    outlier_table,
    cluster_association,
)

__all__ = [
    "usable_metrics",
    "class_colors",
    "plot_sample_clustermap",
    "plot_filter_heatmap",
    "load_thresholds",
    "plot_sample_profiles",
    "plot_per_dataset",
    "plot_outliers",
    "outlier_table",
    "cluster_association",
]