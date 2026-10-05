"""Distribution and separability plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

from config import palette as P
from analysis import figures


def _figure_target(output_dir, name):
    """Destination for the multi-figure plots: an explicit output_dir wins, otherwise the directory
    analysis.figures.set_figure_dir() configured. Falls back to the CWD only if neither is set."""
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.png"
    return figures.figure_path(name) or Path(f"{name}.png")
import matplotlib.pyplot as plt
from config.analysis import MODELS


def plot_metric_distribution(df, metric, group="binder", bins=30, save_path=None):
    """Plot distribution of a metric by group (wide table). Auto-saved when a notebook has called
    analysis.figures.set_figure_dir(); `save_path` still wins."""
    plt.figure(figsize=(7, 5))
    for label, group_df in df.groupby(group):
        plt.hist(group_df[metric].dropna(), bins=bins, alpha=0.5, label=str(label))
    plt.xlabel(P.capitalize_first(metric))
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    path = save_path or figures.figure_path(f"metric_distribution_{metric}_by_{group}")
    if path:
        figures.save_figure(plt.gcf(), path)
    plt.show()


SEPARABILITY_CLASS_ORDER = ["Binder", "Non-Binder", "Mutant+", "Mutant-", "Design", "Target Shuffle"]
# the binary label: 'binder' is 0/1 for the labelled benchmark and '?' for the unlabelled designs
BINDER_LABELS = {"0": "Non-binder", "1": "Binder", "?": "Design"}
BINDER_ORDER = ["Binder", "Non-binder", "Design"]


def plot_metric_separability_single(long_df, metric, x="binder", hue="source", colors_map=None,
                                    class_order=None, hue_order=None, include_designs=False,
                                    figsize=(7.5, 5.0), save_path=None, show=False):
    """One metric, full size: strip over box, split by class. (The grid version is for scanning.)

    `x` defaults to 'binder', the BINARY label the benchmark is scored on; pass "binder_class" for
    the five-way breakdown. Unlabelled designs are dropped unless 'include_designs'. 'hue' splits
    each class and follows the PALETTE's order, so the legend does not reshuffle when the data
    does; 'hue_order' overrides.
    """
    colors_map = colors_map or P.COLORS_MAP
    d = long_df[long_df["metric"] == metric].dropna(subset=["value"]).copy()
    if d.empty:
        raise ValueError(f"no observed values for metric {metric!r}")
    if x == "binder":
        d[x] = d[x].astype(str).map(BINDER_LABELS).fillna(d[x].astype(str))
        default_order = BINDER_ORDER
    else:
        default_order = class_order or SEPARABILITY_CLASS_ORDER
    if not include_designs:
        d = d[d[x] != "Design"]
    order = [c for c in (class_order or default_order) if c in set(d[x])]
    if not order:
        raise ValueError(f"no classes left for {metric!r} on x={x!r}")

    P.apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    shared = dict(data=d, x=x, y="value", order=order, ax=ax, legend=False)
    if hue:
        # seaborn raises on a palette missing a hue level, so cover whatever is present:
        # an unmapped source falls back to NEUTRAL instead of
        # killing the figure. P.colors_for applies the same rule everywhere else.
        present = set(d[hue].dropna())
        canonical = {"source": P.SOURCE, "dataset": P.DATASET}.get(hue, colors_map)
        hue_levels = hue_order or ([v for v in canonical if v in present]
                                   + [v for v in d[hue].dropna().unique() if v not in canonical])
        palette = {v: colors_map.get(v, P.NEUTRAL) for v in hue_levels}
        missing = [v for v in hue_levels if v not in colors_map]
        if missing:
            import warnings
            warnings.warn(f"plot_metric_separability_single: no palette entry for {sorted(missing)}"
                          f" -- drawn in NEUTRAL grey; add them to config/palette.py to distinguish them")
        shared |= dict(hue=hue, hue_order=hue_levels, palette=palette, dodge=True)
    else:
        shared |= dict(color=P.THEME[0])
    sns.boxplot(width=0.8, showfliers=False, boxprops=dict(alpha=0.45), **shared)
    sns.stripplot(jitter=0.14, size=4.5, edgecolor="white", linewidth=0.4, **shared)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.grid(axis="x", visible=False)
    # counts per class: a box over 11 points must not read like one over 700
    counts = d[x].value_counts()
    ax.set_xticks(range(len(order)), [f"{c}\n(n={int(counts[c])})" for c in order], fontsize=12)
    if hue:
        present = [v for v in palette if v in set(d[hue])]
        handles = [mpatches.Patch(color=palette[v], label=P.dataset_label(v)) for v in present]
        ax.legend(handles=handles, title=P.capitalize_first(hue), frameon=False,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    path = save_path or figures.figure_path(f"separability_{metric}_by_{x}")
    if path:
        figures.save_figure(fig, path, [a for a in ax.get_children() if hasattr(a, "get_texts")])
    plt.show() if show else plt.close()


def plot_metric_separability(long_df, model_groups, output_dir=None, colors_map=None, show=False):
    """Metric separability across binder classes: per model group, one subplot per metric,
    strip over box, coloured by source.
    """
    colors_map = colors_map or P.COLORS_MAP
    x_order = ["Binder", "Non-Binder", "Mutant+", "Mutant-", "Design"]  # 'Target Shuffle', 

    for model, metrics in model_groups.items():
        model_df = long_df[long_df["metric"].isin(metrics)].copy()
        if model_df.empty:
            continue

        valid_metrics = [
            m for m in metrics
            if not model_df[model_df["metric"] == m]["value"].isna().all()
        ]
        if not valid_metrics:
            continue

        ncols = 4
        nrows = (len(valid_metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        last_used = -1
        for i, metric in enumerate(valid_metrics):
            ax = axes[i]
            plot_df = model_df[model_df["metric"] == metric].dropna(subset=["value"])
            if plot_df.empty:
                continue
            last_used = i

            sns.stripplot(
                data=plot_df, x="binder_class", y="value", order=x_order, hue="source",
                dodge=True, jitter=0.1, palette=colors_map, size=3, ax=ax, legend=False,
            )
            sns.boxplot(
                data=plot_df, x="binder_class", y="value", order=x_order, hue="source",
                dodge=True, palette=colors_map, width=0.7, showfliers=False, ax=ax, legend=False,
                boxprops=dict(alpha=0.5),
            )
            ax.set_xlabel(metric)   # the metric names the panel, not a title
            ax.set_ylabel("Score")

        for j in range(last_used + 1, len(axes)):  # remove unused axes
            fig.delaxes(axes[j])

        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in colors_map.items()
        ]

        fig.legend(
            handles=handles,
            title="Source",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
        )


        plt.tight_layout(rect=[0, 0, 1, 0.98])
        figures.save_figure(fig, _figure_target(output_dir, f"{model}_separability"))
        plt.show() if show else plt.close()


def plot_model_agreement(long_df, output_dir=None, model_order=None, hue="binder_class", palette=None, ylabel="Score", show=False):
    """One violin figure per metric family: models side by side, split by binder class.

    'ylabel' is one label for all families, or a {family: label} mapping when they carry different
    units (REU vs A^2 vs REU/A^2).
    """
    # the project palette, not seaborn's Set2, so binder classes keep the
    # same colours here as in every other figure
    palette = palette or (P.BINDER_CLASS if hue == "binder_class" else P.COLORS_MAP)
    sns.set_theme(style="white")

    # Hardcode your fixed order here
    fixed_hue_order = ["Binder", "Mutant+",  "Mutant-", "Non-Binder", "Design", ] #"Target Shuffle"

    for family, family_df in long_df.groupby("metric_family"):
        plot_df = family_df.dropna(subset=["value", "model"])
        if plot_df.empty:
            continue

        split = plot_df[hue].nunique() == 2  
        present = list(plot_df["model"].unique())
        order = [m for m in (model_order or MODELS) if m in present]
        if not order:
            continue

        plt.figure(figsize=(12, 5))
        sns.violinplot(
            data=plot_df, x="model", y="value", hue=hue, order=order,
            hue_order=fixed_hue_order,  # <--- Applied internally
            split=split, inner="quartile", cut=0, density_norm="width", palette=palette,
        )
        # no title: the family is in the file name, as everywhere else in the project
        plt.xlabel("Model")
        plt.ylabel(ylabel.get(family, "Score") if isinstance(ylabel, dict) else ylabel)
        plt.xticks(range(len(order)), [P.model_name(m) for m in order], rotation=30, ha="right")
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        figures.save_figure(plt.gcf(), _figure_target(output_dir, f"{family}_model_agreement"))
        plt.show() if show else plt.close()

# two panels in one figure, so the axis labels run 2pt under the
# project default (palette.apply_plot_style's axes.labelsize) to leave the panels room.
AXIS_LABEL_SIZE = 13


def plot_benchmark_structure(groups, folds, save_path=None, show=False):
    """Two panels on the benchmark's grouping structure.

    Left  -- lineage-group sizes, sorted, coloured by dataset: the effective sample size is the
             number of GROUPS, and they are skewed.
    Right -- CV fold composition stacked by dataset, prevalence annotated above each bar.
             folds=None draws the left panel alone.
    """
    P.apply_plot_style()
    if folds is None or not len(folds):
        fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    # --- left: group sizes
    colors = [P.DATASET.get(d, P.NEUTRAL) for d in groups["dataset"]]
    ax1.bar(range(len(groups)), groups["n"], color=colors, width=0.8)
    med = float(groups["n"].median())
    ax1.axhline(med, color=P.INK_SOFT, lw=1, ls="--", zorder=0)
    ax1.annotate(f"median {med:.0f}", (len(groups) - 0.5, med), xytext=(0, 4),
                 textcoords="offset points", ha="right", va="bottom", fontsize=11, color=P.INK_SOFT)
    ax1.set_xlabel(f"Lineage group (n = {len(groups)}, largest first)", fontsize=AXIS_LABEL_SIZE)
    ax1.set_ylabel("Samples", fontsize=AXIS_LABEL_SIZE)

    seen = list(dict.fromkeys(groups["dataset"]))          # palette order = largest group first
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=P.DATASET.get(d, P.NEUTRAL),
                                      label=P.dataset_label(d)) for d in seen],
               frameon=False, ncol=2, loc="upper right", handlelength=1.2, columnspacing=1.0)

    # horizontal grid only (a vertical line between categorical bars reads as a divider);
    # set before the folds=None early return
    def _grid(ax):
        ax.grid(axis="y", color=P.GRID, lw=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

    # --- right: fold composition
    if ax2 is None:
        _grid(ax1)
        fig.tight_layout()
        path = save_path or figures.figure_path("benchmark_structure_lineage_groups")
        if path:
            figures.save_figure(fig, path)
        if show:
            plt.show()
        return fig
    ds_cols = [c for c in folds.columns if c not in ("fold", "n", "prevalence", "n_groups")]
    ds_cols = sorted(ds_cols, key=lambda d: -folds[d].sum())
    bottom = np.zeros(len(folds))
    for d in ds_cols:
        ax2.bar(folds["fold"], folds[d], bottom=bottom, label=P.dataset_label(d), width=0.68,
                color=P.DATASET.get(d, P.NEUTRAL), edgecolor=P.SURFACE, linewidth=2)
        bottom += folds[d].to_numpy()
    for x, tot, prev in zip(folds["fold"], bottom, folds["prevalence"]):
        ax2.annotate(f"prev {prev:.2f}", (x, tot), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=11, color=P.INK_SOFT)
    ax2.set_xlabel("Cross-validation fold", fontsize=AXIS_LABEL_SIZE)
    ax2.set_ylabel("Samples", fontsize=AXIS_LABEL_SIZE)
    ax2.set_xticks(folds["fold"])
    ax2.set_ylim(0, bottom.max() * 1.12)

    for ax in (ax1, ax2):
        _grid(ax)
    fig.tight_layout()
    path = save_path or figures.figure_path("benchmark_structure")
    if path:
        figures.save_figure(fig, path)
    if show:
        plt.show()
    return fig
