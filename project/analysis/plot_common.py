"""Figure helpers shared by the evaluation and ranking plot modules.

Saving (`_finish`) and the metric bar-chart machinery live here so neither module has to import
the other's internals: `plot_top_metrics_bar` (evaluation) and `plot_rf_importances` (ranking)
draw the same kind of bar and must stay one visual language.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.legend as mlegend

from config import palette as P
from analysis import figures


def _slug(label):
    """'peptide  (n=60, prevalence=0.62)' -> '_peptide'; '' for no label. Keeps the per-dataset
    figures from overwriting each other (and the pooled one) under auto-saving."""
    return f"_{str(label).split()[0].strip().lower()}" if label else ""


def _finish(save_path, show, name=None):
    """Save and close. `save_path` wins; otherwise `name` is auto-saved when a notebook has called
    analysis.figures.set_figure_dir(). Legends anchored outside the axes are handed to savefig
    explicitly -- bbox_inches="tight" crops to the AXES and would truncate the longest label."""
    path = save_path or figures.figure_path(name)
    if path:
        fig = plt.gcf()
        extra = [a for ax in fig.axes for a in ax.get_children() if isinstance(a, mlegend.Legend)]
        figures.save_figure(fig, path, extra)
    plt.show() if show else plt.close()


CATEGORY_HATCH = {"global": "", "interface": "//", "geometry": "", "sequence": "", "developability": "", "energy": ""}


# Filter-only metrics are GREYED, not faded: full opacity, no model colour, since a metric outside
# the feature set is not a ranking signal to attribute to a model.
# The legend lives in its own invisible axes, LEGEND_W inches wide: an axes is always inside the
# figure's tight bbox, so the longest label cannot be cropped the way an anchored legend was.
LEGEND_W = 2.4


def _bar_axes(n_bars, height=4.8):
    """(fig, plot axes, legend axes) for a metric bar chart: bars left, legend strip right."""
    main_w = 0.45 * n_bars + 2.5
    fig, (ax, legend_ax) = plt.subplots(1, 2, figsize=(main_w + LEGEND_W, height),
                                        gridspec_kw={"width_ratios": [main_w, LEGEND_W]})
    legend_ax.axis("off")
    return fig, ax, legend_ax


def _metric_bars(ax, metrics, values, width=0.7, filter_only=None):
    """One bar per metric, coloured by model family and hatched by category -> (families, categories).

    `filter_only` greys a bar instead of colouring it, for metrics outside the feature set; their
    family comes back as None so the legend lists no colour that no bar carries."""
    from analysis.distributions.long import assign_model
    from preprocessing.metric_meta import load_categories, get_category
    metrics = list(metrics)
    families = [P.model_family(assign_model(m) or "") for m in metrics]
    categories_map = load_categories()
    categories = [get_category(m, categories_map) for m in metrics]
    if filter_only is None:
        filter_only = np.zeros(len(metrics), dtype=bool)
    with plt.rc_context({"hatch.linewidth": 1.2}):
        for xi, value, family, category, grey in zip(np.arange(len(metrics)), values, families, categories, filter_only):
            colour = P.NEUTRAL if grey else P.MODEL_FAMILY.get(family, P.NEUTRAL)
            # no hatch on a filter-only bar: grey already says "not a ranking feature", and its
            # category only restates that --
            # one fact, and barely legible on NEUTRAL anyway.
            hatch = "" if grey else CATEGORY_HATCH.get(category, "")
            # two passes: the hatch takes the edge colour, so a frame would blacken it
            # too. Fill + light hatch first, then an unfilled bar for the outline.
            ax.bar(xi, value, width=width, color=colour, hatch=hatch,
                   edgecolor=P.SURFACE, linewidth=2, zorder=2)
            ax.bar(xi, value, width=width, fill=False, edgecolor=P.INK, linewidth=0.8, zorder=4)
    # None for greyed bars, so no legend key is listed that no bar carries
    families = [None if grey else f for f, grey in zip(families, filter_only)]
    categories = [None if grey else c for c, grey in zip(categories, filter_only)]
    return families, categories


def _model_category_legends(ax, families, categories, any_filter_only=False):
    """One legend with section headers (model / category / feature set), drawn on _bar_axes'
    legend axes. One legend, not three: hand-positioned blocks collide when the font changes.
    """
    def header(text):
        """A section title inside the legend: an invisible swatch carrying the heading."""
        return mpatches.Patch(alpha=0, linewidth=0), text

    rows = [header("Model")]
    rows += [(mpatches.Patch(facecolor=P.MODEL_FAMILY[f], edgecolor=P.INK, linewidth=0.8), P.model_label(f))
             for f in P.MODEL_FAMILY if f in families]

    with plt.rc_context({"hatch.linewidth": 1.2}):
        category_rows = [(mpatches.Patch(facecolor=P.NEUTRAL, edgecolor=P.INK, linewidth=0.8,
                                         hatch=CATEGORY_HATCH[c]), P.capitalize_first(c))
                         for c in CATEGORY_HATCH if c in categories]
        if category_rows:
            rows += [header(""), header("Category")] + category_rows
        if any_filter_only:   # grey is a THIRD encoding here, so it needs saying
            rows += [header(""), header("Feature set"),
                     (mpatches.Patch(facecolor=P.NEUTRAL, edgecolor=P.INK, linewidth=0.8), "Filter-only")]

        handles, labels = zip(*rows)
        ax.legend(handles=list(handles), labels=list(labels), frameon=False,
                  loc="upper left", bbox_to_anchor=(0, 1), handlelength=1.6, labelspacing=0.55)
