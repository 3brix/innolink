"""
Evaluation figures.

Each function takes prepared inputs (the rankings table, the raw dataframe + directions for the curves, or precomputed embeddings) 
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from preprocessing.align import get_direction




# to do: migrate to config
DEFAULT_MARKER_MAP = {0: "X", "0": "X", 1: "o", "1": "o", "?": "D"}
DEFAULT_ALPHA_MAP = {"target_shuffle": 0.4, "original": 1.0, "alanine_scan": 1.0}


def _finish(save_path, show):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_top_metrics_bar(rankings, top_n=15, save_path=None, show=False):
    """Grouped bar of ROC-AUC / PR-AUC / precision@10% / precision for the top-N metrics by PR-AUC."""
    cols = [("aligned_roc", "ROC-AUC", "#e74c3c"), ("pr_auc", "PR-AUC", "#2ecc71"),
            ("precision_at_pct", "Precision@10%", "#3498db"), ("precision", "Precision", "#f39c12")]
    top = rankings.head(top_n)
    x = np.arange(len(top))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (col, label, color) in enumerate(cols):
        ax.bar(x + (i - 1.5) * width, top[col], width, label=label, color=color, alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_title(f"Top {top_n} Metrics by PR-AUC", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top["metric"], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    _finish(save_path, show)


def plot_roc_curves(df, rankings, directions, top_n=15, save_path=None, show=False):
    """ROC curves for the top-N metrics (scores direction-aligned)."""
    top = rankings.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in top.iterrows():
        metric = row["metric"]
        clean = df[[metric, "binder"]].dropna()
        if clean["binder"].nunique() < 2:
            continue
        score = clean[metric].values * get_direction(metric, directions)
        fpr, tpr, _ = roc_curve(clean["binder"].astype(int), score)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{metric} (AUC={row['aligned_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Top {top_n} ROC Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    _finish(save_path, show)


def plot_pr_curves(df, rankings, directions, top_n=15, save_path=None, show=False):
    """Precision-Recall curves for the top-N metrics (scores direction-aligned --> not necessary)."""
    top = rankings.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in top.iterrows():
        metric = row["metric"]
        clean = df[[metric, "binder"]].dropna()
        if clean["binder"].nunique() < 2:
            continue
        y = clean["binder"].astype(int)
        score = clean[metric].values * get_direction(metric, directions)
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=f"{metric} (PR-AUC={row['pr_auc']:.3f})")
    ax.hlines(df["binder"].mean(), 0, 1, colors="k", linestyles="--", alpha=0.3)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Top {top_n} Precision-Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    _finish(save_path, show)


def plot_precision_ranking(rankings, top_n=10, label_map=None, save_path=None, show=False):
    """Horizontal bar of the top-N metrics by precision ."""
    label_map = label_map or {}   # should create at some point. config?
    top = rankings.sort_values("precision", ascending=False).head(top_n)
    y = np.arange(len(top))
    labels = [label_map.get(m, m) for m in top["metric"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y, top["precision"], color="#f39c12", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Precision", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    _finish(save_path, show)


def plot_metric_comparison_grid(rankings, save_path=None, show=False):
    """2x3 scatter grid comparing ROC-AUC / PR-AUC / precision@10% / precision pairwise."""
    pairs = [
        ("pr_auc", "aligned_roc", "PR-AUC", "ROC-AUC", "steelblue"),
        ("aligned_roc", "precision_at_pct", "ROC-AUC", "Precision@10%", "seagreen"),
        ("pr_auc", "precision_at_pct", "PR-AUC", "Precision@10%", "coral"),
        ("precision", "pr_auc", "Precision", "PR-AUC", "steelblue"),
        ("precision", "aligned_roc", "Precision", "ROC-AUC", "seagreen"),
        ("precision", "precision_at_pct", "Precision", "Precision@10%", "coral"),
    ]
    d = rankings.fillna(0)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (xk, yk, xl, yl, color) in zip(axes.flatten(), pairs):
        ax.scatter(d[xk], d[yk], alpha=0.6, s=50, color=color)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(f"{xl} vs {yl}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.suptitle("Metric Comparison: ROC-AUC vs PR-AUC vs F1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(save_path, show)


def plot_quadrant(df, rankings, colors_map, marker_map=None, alpha_map=None, save_path=None, show=False):
    """
    Scatter the top-2 PR-AUC metrics against each other, one point per sample.
    Colored by source, styled by binder; per-sample means are plotted with the per-metric optimal thresholds(raw) as reference lines.
    """
    marker_map = marker_map or DEFAULT_MARKER_MAP
    alpha_map = alpha_map or DEFAULT_ALPHA_MAP

    top = rankings.sort_values("pr_auc", ascending=False).head(20)
    if len(top) < 2:
        return
    m1i, m2i = top.iloc[4], top.iloc[2]
    m1, t1 = m1i["metric"], m1i["opt_threshold_raw"]
    m2, t2 = m2i["metric"], m2i["opt_threshold_raw"]

    plot_df = df.groupby(["source", "sample", "type", "binder"])[[m1, m2]].agg(["mean", "std"]).reset_index()
    plot_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in plot_df.columns]
    plot_df["binder"] = plot_df["binder"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 8))
    for t, subdf in plot_df.groupby("type"):
        sns.scatterplot(
            data=subdf, x=f"{m1}_mean", y=f"{m2}_mean", hue="source", palette=colors_map,
            style="binder", markers=marker_map, s=80, alpha=alpha_map.get(t, 1.0),
            edgecolor="w", linewidth=0.6, legend=False, ax=ax,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.axvline(t1, color="black", linestyle="--", alpha=0.3)
    ax.axhline(t2, color="black", linestyle="--", alpha=0.3)

    max_val = max(plot_df[f"{m1}_mean"].max(), plot_df[f"{m2}_mean"].max()) * 1.05
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel(f"{m1}\n(PR-AUC: {m1i['pr_auc']:.3f})", fontsize=11, labelpad=10)
    ax.set_ylabel(f"{m2}\n(PR-AUC: {m2i['pr_auc']:.3f})", fontsize=11, labelpad=10)

    active_sources = plot_df["source"].dropna().unique()

    handles = [mpatches.Patch(color="none", label="Source")]
    for source in active_sources:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=colors_map[source],
                marker="s",
                linestyle="None",
                markersize=8,
                label=source,
            )
        )

    ax.legend(
        handles=handles,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.05, 0.7),
        frameon=False,
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, right=0.78)
    _finish(save_path, show)


def plot_embedding(emb, metric_names, labels, title, xlabel="Dim 1", ylabel="Dim 2",
                   annotate=False, color_dict=None, save_path=None, show=False):
    """
    Scatter a 2D metric embedding (PCA or UMAP), colored by "labels" (per-metric series / list (e.g. family or model). 
    Colors are auto-assigned from a colormap unless "color_dict" is provided.
    """
    labels = list(labels)
    cats = sorted(set(labels))
    if color_dict is None:
        cmap = plt.get_cmap("tab20")
        color_dict = {c: cmap(i % 20) for i, c in enumerate(cats)}
    point_colors = [color_dict[l] for l in labels]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(emb[:, 0], emb[:, 1], c=point_colors, s=60, zorder=2)
    if annotate:
        for i, name in enumerate(metric_names):
            ax.annotate(name, (emb[i, 0], emb[i, 1]), fontsize=6, alpha=0.75)
    handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=8, color=color_dict[c], label=c) for c in cats]
    ax.legend(handles=handles, title="Group", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    _finish(save_path, show)


def plot_spearman_heatmap(corr, save_path=None, show=False):
    """Heatmap of a Spearman correlation matrix among metrics."""
    size = max(8, 0.5 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.6}, xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title("Spearman Correlation (metrics)")
    plt.tight_layout()
    _finish(save_path, show)


# Class-distribution panels ---> to do: config
DISTRIBUTION_CLASS_ORDER = ["Binder", "Non-Binder", "Design", "Target Shuffle", "Mutant+", "Mutant-" "Unknown"]
DISTRIBUTION_PALETTE = {
    "Binder": "#2ca02c",
    "Non-Binder": "#d62728",
    "Design": "#f70eff",
    "Target Shuffle": "#ff7f0e",
    "Mutant+": "#67b3bd",
    "Mutant-": "#9467bd",
    "Unknown": "#9425bd",
}


def plot_class_distributions(
    df,
    rankings=None,
    metrics=None,
    top_n=5,
    class_order=None,
    palette=None,
    save_path=None,
    show=False,
):
    """
    Per-metric class distributions of the RAW score, one metric per row: left = violin + strip (individual points), right = shared-bin histogram, both colored by "binder_type".

    "metrics" selects which columns to plot, in order. If None, the top "top_n" rows of "rankings" are used. 
    with "metrics.top_metrics_distinct_family" to plot the strongest separators across distinct families. 
    Raw scores, so read each panel in its own direction (metrics, like pae are lower = better). "pr_auc" is shown in each violin title.
    """
    class_order = class_order or DISTRIBUTION_CLASS_ORDER
    palette = palette or DISTRIBUTION_PALETTE
    if metrics is None:
        if rankings is None:
            raise ValueError("pass either `metrics` or `rankings`")
        metrics = rankings.head(top_n)["metric"].tolist()

    pr = None
    if rankings is not None and "pr_auc" in rankings.columns:
        pr = rankings.set_index("metric")["pr_auc"]

    n = len(metrics)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    axes = np.atleast_2d(axes)

    for i, metric in enumerate(metrics):
        ax_v, ax_h = axes[i, 0], axes[i, 1]
        d = df[[metric, "binder_type"]].dropna()

        # Left: violin + stripplot (individual points)
        sns.violinplot(
            data=d, x="binder_type", y=metric, order=class_order,
            hue="binder_type", palette=palette, legend=False,
            inner=None, cut=0, density_norm="width", ax=ax_v,
        )
        for art in ax_v.collections:
            art.set_alpha(0.30)
        sns.stripplot(
            data=d, x="binder_type", y=metric, order=class_order,
            hue="binder_type", palette=palette, legend=False,
            size=3, jitter=0.15, edgecolor="white", linewidth=0.3, ax=ax_v,
        )
        title = metric
        if pr is not None and metric in pr.index:
            title = f"{metric}  (PR-AUC = {float(pr.loc[metric]):.2f})"
        ax_v.set_title(title, fontsize=10, fontweight="bold")
        ax_v.set_xlabel("")
        ax_v.set_ylabel("raw score")

        # Right: histogram colored by binder_type, on ONE shared bin grid so the
        # bars line up across classes (per-class bins would misalign).
        vmin, vmax = d[metric].min(), d[metric].max()
        edges = np.linspace(vmin, vmax, 26) if vmax > vmin else 25
        for cls in class_order:
            vals = d.loc[d["binder_type"] == cls, metric]
            if len(vals):
                ax_h.hist(vals, bins=edges, alpha=0.5, color=palette.get(cls), label=cls)
        ax_h.set_title(f"{metric} \u2014 distribution", fontsize=10)
        ax_h.set_xlabel("raw score")
        ax_h.set_ylabel("count")
        ax_h.legend(fontsize=8)

    plt.suptitle(
        f"Top {n} separating scores across distinct families \u2014 class distributions (raw)",
        fontsize=16, y=1.001,
    )
    plt.tight_layout()
    _finish(save_path, show)