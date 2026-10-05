"""Design ranking: RF (primary) + product composite -> shortlist and disagreement report."""

from .consensus import (
    add_rank,
    build_consensus,
    shortlist,
    disagreements,
    rank_agreement,
)

__all__ = [
    "add_rank",
    "build_consensus",
    "shortlist",
    "disagreements",
    "rank_agreement",
] + ['plot_rf_importances', 'plot_rf_importance_comparison', 'plot_rf_performance', 'plot_design_scores', 'plot_ranking_curves', 'plot_pooled_vs_per_source', 'plot_per_source_comparison', 'plot_standardisation_sensitivity', 'plot_fold_pr_auc', 'plot_pooled_vs_fold_mean', 'method_key', 'method_display', 'method_colour']


# Plotting is imported LAZILY, as in analysis.evaluation: run_consensus.py imports this package
# for build_consensus / shortlist and must not pull in matplotlib.
_PLOT_NAMES = ['plot_rf_importances', 'plot_rf_importance_comparison', 'plot_rf_performance', 'plot_design_scores', 'plot_ranking_curves', 'plot_pooled_vs_per_source', 'plot_per_source_comparison', 'plot_standardisation_sensitivity', 'plot_fold_pr_auc', 'plot_pooled_vs_fold_mean', 'method_key', 'method_display', 'method_colour']


def __getattr__(name):
    if name in _PLOT_NAMES:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + _PLOT_NAMES)
