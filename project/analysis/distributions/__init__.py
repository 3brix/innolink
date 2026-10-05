"""Metric distribution and separability analysis (raw data only)."""
 
from .wide import (
    get_numeric_metrics,
    get_all_numeric_metrics,
    add_binder_class,
    make_long_scores,
)
from .summaries import (
    benchmark_structure,
)
from .long import (
    assign_model,
    get_model_groups,
    load_metric_metadata,
    enrich_long_scores,
)
 
__all__ = [
    "get_numeric_metrics",
    "get_all_numeric_metrics",
    "add_binder_class",
    "make_long_scores",
    "assign_model",
    "get_model_groups",
    "load_metric_metadata",
    "enrich_long_scores",
    "benchmark_structure",
    "plot_metric_distribution",
    "plot_metric_separability",
    "plot_metric_separability_single",
    "plot_model_agreement",
    "plot_benchmark_structure",
]
 
 


# ---------------------------------------------------------------------------
# Plotting is imported LAZILY. Pipeline stages import this package for its analysis functions and
# must not pay for matplotlib/seaborn: a plain 'from .plots import ...' here made every stage
# depend on them, which is how run_composite once died with "No module named seaborn" in an
# environment that had scikit-learn but no plotting stack.
#
# 'from analysis.distributions import plot_x' still works -- it just triggers the
# import at that moment (PEP 562).
# ---------------------------------------------------------------------------
_PLOT_NAMES = ['plot_metric_distribution', 'plot_metric_separability', 'plot_metric_separability_single', 'plot_model_agreement',
               'plot_benchmark_structure']


def __getattr__(name):
    if name in _PLOT_NAMES:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + _PLOT_NAMES)
