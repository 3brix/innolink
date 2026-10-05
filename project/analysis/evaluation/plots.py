"""Evaluation figures. Each function takes prepared inputs -- the rankings table, the raw
frame plus directions for the curves, or a precomputed embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from config import palette as P
from analysis.plot_common import _slug, _finish, _bar_axes, _metric_bars, _model_category_legends
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from preprocessing.align import get_direction




# to do: migrate to config
DEFAULT_MARKER_MAP = {0: "X", "0": "X", 1: "o", "1": "o", "?": "D"}
DEFAULT_ALPHA_MAP = {"target_shuffle": 0.4, "original": 1.0, "alanine_scan": 1.0}


def _model_set(rankings, model_set_only):
    """Every rankings-based plot takes 'model_set_only': True keeps only metrics in the model/benchmark
    feature set, i.e. drops the filter-only categories not used in the rf and composite."""
    return rankings[rankings["in_model_set"]] if model_set_only and "in_model_set" in rankings.columns else rankings



# baselines
MEASURE_SPECS = [
    ("pr_auc",           "PR-AUC",        "prevalence"),
    ("aligned_roc",      "ROC-AUC",       0.5),
    ("precision_at_pct", "Precision@10%", "prevalence"),
]

# drop?
def plot_metric_scores(rankings, top_n=15, measures=None, sort_by="pr_auc", model_set_only=False,
                       save_path=None, show=False):
    """Sorted horizontal dot plot per measure, each with its own baseline, the quantity does not start at zero, and a bar implies a zero origin."""
    import numpy as np
    measures = measures or MEASURE_SPECS
    measures = [m for m in measures if m[0] in rankings.columns]
    top = _model_set(rankings, model_set_only).sort_values(sort_by, ascending=False).head(top_n).iloc[::-1]   # best at the TOP
    labels = top["metric"].tolist()
    y = np.arange(len(top))

    fig, axes = plt.subplots(1, len(measures), figsize=(4.4 * len(measures), 0.36 * len(top) + 2.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, title, base) in zip(axes, measures):
        baseline = float(top["prevalence"].median()) if base == "prevalence" else float(base)
        vals = top[col].astype(float)
        # stem from the baseline, so length encodes the margin over chance
        ax.hlines(y, baseline, vals, color=P.GRID, linewidth=2.0, zorder=1)
        ax.scatter(vals, y, s=46, color=P.THEME[0], zorder=3)
        # the dashed line is the chance reference; the x label already names the measure
        ax.axvline(baseline, color=P.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    nice = {m[0]: m[1] for m in MEASURE_SPECS}.get(sort_by, sort_by)
    plt.tight_layout()
    _finish(save_path, show, name=f"metric_scores_by_{sort_by}")


# Metric category -> hatch, the second encoding next to model colour. Fixed per category.
# Family -> (colour, marker, filled) for the embeddings. A global/interface pair shares a hue;
# circle = global, triangle = interface. 6 hues is the palette ceiling, so lis shares ipsae's hue
# and is told apart by a hollow marker.
FAMILY_STYLE = {
    "plddt": (P.THEME[0], "o", True), "iplddt": (P.THEME[0], "^", True),
    "pae":   (P.THEME[1], "o", True), "ipae":   (P.THEME[1], "^", True),
    "pde":   (P.THEME[2], "o", True), "ipde":   (P.THEME[2], "^", True),
    "ptm":   (P.THEME[3], "o", True), "iptm":   (P.THEME[3], "^", True),
    "ipsae": (P.THEME[4], "^", True), "lis":    (P.THEME[4], "^", False),
    "pdockq2": (P.THEME[5], "^", True),
}


# Per-score y-label and the no-skill value. 'pr_auc' is sklearn's average_precision_score --
# average precision, not a trapezoidal area; ap_norm is prevalence-corrected, so no-skill is 0.
SCORE_SPECS = {"precision_at_pct": ("Precision@10%", "prevalence", (0, 1)),
               "pr_auc":           ("PR-AUC", "prevalence", (0, 1)),
               "ap_norm":          ("Prevalence-normalized AP", 0.0, (0, 1)),
               "ap_norm_ds_mean":  ("Prevalence-normalized AP", 0.0, (0, 1)),
               "ap_norm_ds_min":   ("Prevalence-normalized AP (dataset min)", 0.0, (0, 1))}


def add_ap_norm(rankings) -> pd.DataFrame:
    """ap_norm = (PR-AUC - prevalence) / (1 - prevalence): 0 = no-skill, 1 = perfect.

    Makes per-dataset values comparable; raw PR-AUC is not (prevalence runs 0.09-0.74 here).
    """
    if "ap_norm" in rankings.columns or not {"pr_auc", "prevalence"} <= set(rankings.columns):
        return rankings
    out = rankings.copy()
    out["ap_norm"] = (out["pr_auc"] - out["prevalence"]) / (1 - out["prevalence"])
    return out


def plot_top_metrics_bar(rankings, score="precision_at_pct", top_n=10, model_set_only=False, label=None,
                         metric_labels=False, xlabel="Metrics", save_path=None, show=False):
    """Top-N metrics by `score` ("precision_at_pct", "pr_auc" or "ap_norm"), high to low.

    `model_set_only` keeps the feature set; `label` is a corner annotation for when several of
    these figures are read together.
    """
    P.apply_plot_style()
    r = _model_set(add_ap_norm(rankings), model_set_only)
    tie_break = "precision_at_pct" if score != "precision_at_pct" else "pr_auc"
    top = r.dropna(subset=[score]).sort_values([score, tie_break], ascending=False).head(top_n)
    x = np.arange(len(top))

    # model_set_only=False keeps filter-only metrics in the figure, greyed:
    # not silently ranked alongside the model features.
    filter_only = (~top["in_model_set"].to_numpy(bool)) if "in_model_set" in top.columns else None
    fig, ax, legend_ax = _bar_axes(len(top))
    families, categories = _metric_bars(ax, top["metric"], top[score].astype(float), filter_only=filter_only)
    ylabel, chance, ylim = SCORE_SPECS.get(score, (score, "prevalence", (0, 1)))
    baseline = float(top["prevalence"].median()) if chance == "prevalence" else float(chance)
    if baseline > ylim[0]:   # ap_norm's no-skill point IS the axis floor -> the line would just trace it
        # the dashed line is the chance reference; unlabelled, which frees the
        # right-hand strip for the legend block (an outside label collided with it)
        ax.axhline(baseline, color=P.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    # metric_labels=False drops the column names, for small multiples where rotated
    # names crowds the axis and the model / category composition is the point.
    ax.set_xticks(x, top["metric"] if metric_labels else [""] * len(top),
                  rotation=60, ha="right", fontsize=11)
    # the x-label slot carries the dataset statement when there is one, otherwise names the axis
    if label or xlabel:
        ax.set_xlabel(label or xlabel, fontweight="bold" if label else "normal", labelpad=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    _model_category_legends(legend_ax, families, categories,
                            any_filter_only=filter_only is not None and bool(filter_only.any()))
    plt.tight_layout()
    _finish(save_path, show, name=f"top_metrics_bar_{score}{_slug(label)}")


def plot_top_metrics_bar_by_dataset(by_dataset, score="pr_auc", model_set=None, datasets=None,
                                    save_path=None, show=False, **kwargs):
    """One plot_top_metrics_bar figure per dataset, from rankings_by_dataset.csv.

    Each panel is baselined on its own rows, so raw pr_auc / precision_at_pct heights are NOT
    comparable between datasets; score="ap_norm" is. `model_set` is the metric list to keep (that
    file has no in_model_set column). `save_path` becomes <stem>_<dataset><suffix>.
    """
    from pathlib import Path
    r = by_dataset.copy()
    if model_set is not None:
        r["in_model_set"] = r["metric"].isin(set(model_set))
    for ds in (datasets if datasets is not None else sorted(r["dataset"].unique())):
        g = r[r["dataset"] == ds]
        if g.empty:
            continue
        n, prev = int(g["n_eval"].iloc[0]), float(g["prevalence"].iloc[0])
        path = None
        if save_path:
            p = Path(save_path)
            path = p.with_name(f"{p.stem}_{ds}{p.suffix}")
        plot_top_metrics_bar(g, score=score,
                             label=f"{P.dataset_label(ds)}  (n={n}, prevalence={prev:.2f})",
                             save_path=path, show=show, **kwargs)


def plot_roc_curves(df, rankings, directions, top_n=10, model_set_only=False, save_path=None, show=False):
    """ROC curves for the top-N metrics (scores direction-aligned)."""
    top = _model_set(rankings, model_set_only).head(top_n)
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    _finish(save_path, show, name="roc_curves_top_metrics")


def plot_pr_curves(df, rankings, directions, top_n=10, model_set_only=False, save_path=None, show=False):
    """Precision-Recall curves for the top-N metrics (scores direction-aligned --> not necessary)."""
    top = _model_set(rankings, model_set_only).head(top_n)
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    _finish(save_path, show, name="pr_curves_top_metrics")


def plot_precision_ranking(rankings, top_n=10, label_map=None, score="precision_at_pct", model_set_only=False,
                           save_path=None, show=False):
    """Horizontal bar of the top-N metrics.Sorted by 'score, default precision@10%."""
    label_map = label_map or {}
    top = _model_set(rankings, model_set_only).dropna(subset=[score]).sort_values(score, ascending=False).head(top_n)
    y = np.arange(len(top))
    labels = [label_map.get(m, m) for m in top["metric"]]
    baseline = float(top["prevalence"].median()) if "prevalence" in top.columns else None

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(top) + 1.8))
    ax.barh(y, top[score], color=P.THEME[0], height=0.72)   # single series -> slot 1, no legend
    if baseline is not None:
        ax.axvline(baseline, color=P.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
        ax.annotate(f"chance {baseline:.2f}", xy=(baseline, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -10), textcoords="offset points", fontsize=8, color=P.INK_SOFT,
                    va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Precision@10%")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    plt.tight_layout()
    _finish(save_path, show, name=f"precision_ranking_{score}")


# the in-sample F1-optimal 'precision' column is excluded (Methods 2.3.2); it collapses onto the prevalence
COMPARISON_PAIRS = [
    ("pr_auc", "aligned_roc", "PR-AUC", "ROC-AUC"),
    ("aligned_roc", "precision_at_pct", "ROC-AUC", "Precision@10%"),
    ("pr_auc", "precision_at_pct", "PR-AUC", "Precision@10%"),
]


def plot_metric_comparison_grid(rankings, pairs=None, model_set_only=False, save_path=None, show=False):
    """One square scatter per measure pair, one point per metric, dashed y = x.

    `pairs` defaults to COMPARISION_PAIRS; `save_path` becomes <stem>_<x>_vs_<y><suffix>.
    """
    from pathlib import Path
    P.apply_plot_style()
    r = _model_set(rankings, model_set_only)
    for xk, yk, xl, yl in pairs or COMPARISON_PAIRS:
        d = r[[xk, yk]].dropna()
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.scatter(d[xk], d[yk], alpha=0.7, s=40, color=P.THEME[0], edgecolor=P.SURFACE, linewidth=0.8, zorder=3)
        ax.plot([0, 1], [0, 1], color=P.NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.tight_layout()
        path = None
        if save_path:
            p = Path(save_path)
            path = p.with_name(f"{p.stem}_{xk}_vs_{yk}{p.suffix}")
        _finish(path, show, name=f"metric_comparison_{xk}_vs_{yk}")


def plot_quadrant(df, rankings, colors_map, marker_map=None, alpha_map=None, metrics=None,
                  equal_scale=None, model_set_only=False, save_path=None, show=False):
    """Two metrics scattered against each other, one point per sample, coloured by source and
    styled by binder, with each metric's F1-optimal cutoff as the quadrant lines.

    metrics      (x, y); default the top two predictors by PR-AUC.
    equal_scale  force a square axis; default auto, only when both already share a scale.
    """
    marker_map = marker_map or DEFAULT_MARKER_MAP
    alpha_map = alpha_map or DEFAULT_ALPHA_MAP

    if metrics is not None:
        pick = rankings[rankings["metric"].isin(metrics)].set_index("metric").loc[list(metrics)].reset_index()
        if len(pick) < 2:
            return
        m1i, m2i = pick.iloc[0], pick.iloc[1]
    else:
        # predictors only:
        # an energy/developability column is filter-only and not a ranking metric
        pool = _model_set(rankings, model_set_only)
        pool = pool.sort_values("pr_auc", ascending=False)
        if len(pool) < 2:
            return
        m1i, m2i = pool.iloc[0], pool.iloc[1]
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
    ax.axvline(t1, color=P.INK_SOFT, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.axhline(t2, color=P.INK_SOFT, linestyle=(0, (4, 3)), linewidth=1.0)

    xs, ys = plot_df[f"{m1}_mean"], plot_df[f"{m2}_mean"]
    if equal_scale is None:
        # same scale only if the two ranges are comparable (within 2x) and similarly located
        xr, yr = xs.max() - xs.min(), ys.max() - ys.min()
        span = max(xr, yr) or 1.0
        equal_scale = (min(xr, yr) / span > 0.5) and abs(xs.median() - ys.median()) < span
    if equal_scale:
        lo = min(xs.min(), ys.min(), t1, t2)
        hi = max(xs.max(), ys.max(), t1, t2)
        pad = 0.05 * (hi - lo or 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        for axis, v, t in ((ax.set_xlim, xs, t1), (ax.set_ylim, ys, t2)):
            lo, hi = min(v.min(), t), max(v.max(), t)
            pad = 0.05 * (hi - lo or 1.0)
            axis(lo - pad, hi + pad)
    ax.set_xlabel(f"{m1}\n(PR-AUC: {m1i['pr_auc']:.3f})", fontsize=11, labelpad=10)
    ax.set_ylabel(f"{m2}\n(PR-AUC: {m2i['pr_auc']:.3f})", fontsize=11, labelpad=10)

    active_sources = plot_df["source"].dropna().unique()

    handles = [mpatches.Patch(color="none", label="Source")]
    for source in active_sources:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=colors_map.get(source, P.NEUTRAL),
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
    _finish(save_path, show, name=f"quadrant_{m1}_vs_{m2}")


def plot_embedding(emb, metric_names, labels, title=None, xlabel="Dim 1", ylabel="Dim 2",
                   annotate=False, color_dict=None, save_path=None, show=False):
    """Scatter a 2D metric embedding (PCA or UMAP), coloured by `labels` (family, model, ...).

    Categories and model families use the palette's own mapping; `color_dict` overrides. Any other
    grouping gets hue x marker shape, because 6 hues is the palette's ceiling and a colormap must
    never cycle. Which metrics are embedded is decided upstream.
    """
    labels_in, labels = labels, list(labels)   # keep the Series: its name ("family"/"model") titles the legend
    cats = sorted(set(labels))
    markers, filled = {c: "o" for c in cats}, {c: True for c in cats}
    if color_dict is None:
        known = {**P.CATEGORY, **P.MODEL_FAMILY}
        if all(c in FAMILY_STYLE for c in cats):
            cats = [c for c in FAMILY_STYLE if c in cats]   # legend in pair order, not alphabetical
            color_dict = {c: FAMILY_STYLE[c][0] for c in cats}
            markers = {c: FAMILY_STYLE[c][1] for c in cats}
            filled = {c: FAMILY_STYLE[c][2] for c in cats}
        elif all(c in known for c in cats):
            color_dict = {c: known[c] for c in cats}
        else:
            shapes = ["o", "s", "^", "D"]
            n = len(P.THEME)
            color_dict = {c: P.THEME[i % n] if i < n * len(shapes) else P.NEUTRAL for i, c in enumerate(cats)}
            markers = {c: shapes[min(i // n, len(shapes) - 1)] for i, c in enumerate(cats)}
            if len(cats) > n * len(shapes):
                import warnings
                warnings.warn(f"plot_embedding: {len(cats)} groups exceed {n * len(shapes)} hue x shape "
                              f"combinations; the rest share the neutral colour -- facet instead")

    fig, ax = plt.subplots(figsize=(11, 9))
    for c in cats:
        idx = [i for i, l in enumerate(labels) if l == c]
        col = color_dict.get(c, P.NEUTRAL)
        ax.scatter(emb[idx, 0], emb[idx, 1], marker=markers[c], s=60, zorder=2,
                   **(dict(color=col, edgecolor=P.SURFACE, linewidth=0.8) if filled[c] else dict(facecolor="none", edgecolor=col, linewidth=1.6)))
    if annotate:
        for i, name in enumerate(metric_names):
            ax.annotate(name, (emb[i, 0], emb[i, 1]), fontsize=6, alpha=0.75)
    handles = [mlines.Line2D([], [], marker=markers[c], linestyle="", markersize=8, color=color_dict.get(c, P.NEUTRAL),
                             markerfacecolor=color_dict.get(c, P.NEUTRAL) if filled[c] else "none",
                             label=P.model_label(c)) for c in cats]
    ax.legend(handles=handles, title=P.capitalize_first(getattr(labels_in, "name", None) or "group"), bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    _finish(save_path, show,
            name=f"embedding_{'pca' if xlabel.lower().startswith('pc') else 'umap'}"
                 f"_by_{getattr(labels_in, 'name', None) or 'group'}")


def plot_spearman_heatmap(corr, annot=None, numeric_ticks=None, save_path=None, show=False):
    """Heatmap of a Spearman matrix among metrics.

    `annot` defaults on at <= 12 rows -- colour alone cannot be read to two decimals.
    `numeric_ticks` (default on above 12) replaces names with their 1-based position in `corr`,
    coloured by model family; 46 rotated names are unreadable and squeeze the cells. Print the
    number -> metric table with plot_spearman_heatmap_index(corr).
    """
    from analysis.distributions.long import assign_model
    P.apply_plot_style()
    size = max(8, 0.5 * len(corr))
    annot = (len(corr) <= 12) if annot is None else annot
    numeric_ticks = (len(corr) > 12) if numeric_ticks is None else numeric_ticks
    labels = [str(i + 1) for i in range(len(corr))] if numeric_ticks else list(corr.columns)

    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(corr, cmap=P.diverging_cmap(), vmin=-1, vmax=1, center=0, square=True,
                annot=annot, fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"shrink": 0.6, "label": "Spearman's \u03c1"},
                xticklabels=labels, yticklabels=labels, ax=ax)

    if numeric_ticks:
        families = [P.model_family(assign_model(m) or "") for m in corr.columns]
        colours = [P.MODEL_FAMILY.get(f, P.NEUTRAL) for f in families]
        for axis in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, colour in zip(axis, colours):
                tick.set_color(colour)
                tick.set_fontweight("semibold")
        ax.tick_params(axis="x", rotation=0)     # x numbers upright (seaborn rotates them 90 by default)
        ax.tick_params(axis="y", rotation=0)     # upright on BOTH axes (seaborn rotates x 90 by default)
        # family key centred under the x axis -- the numbers carry the colour, so the key is what
        # turns them back into models
        handles = [mpatches.Patch(color=P.MODEL_FAMILY[f], label=P.model_label(f))
                   for f in P.MODEL_FAMILY if f in families]
        ax.legend(handles=handles, title="Model", frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.06), ncol=min(len(handles), 6))
    plt.tight_layout()
    _finish(save_path, show, name="spearman_heatmap")


def plot_spearman_heatmap_index(corr) -> pd.DataFrame:
    """The number -> metric table behind plot_spearman_heatmap's numeric ticks (1-based, with the
    model family each number is coloured by)."""
    from analysis.distributions.long import assign_model
    return pd.DataFrame({"n": range(1, len(corr) + 1), "metric": list(corr.columns),
                         "model": [P.model_label(P.model_family(assign_model(m) or ""))
                                   for m in corr.columns]})


# Class-distribution panels ---> to do: config
DISTRIBUTION_CLASS_ORDER = ["Binder", "Non-Binder", "Mutant+", "Mutant-"]  #  "Design", "Target Shuffle",
DISTRIBUTION_PALETTE = P.BINDER_CLASS


def plot_class_distributions(
    df,
    rankings=None,
    metrics=None,
    top_n=5,
    class_order=None,
    palette=None,
    model_set_only=False,
    save_path=None,
    show=False,
):
    """
    Per-metric class distributions of the raw score, one metric per row: left = violin + strip (individual points), right = shared-bin histogram, both colored by "binder_class".
    """
    class_order = class_order or DISTRIBUTION_CLASS_ORDER
    palette = palette or DISTRIBUTION_PALETTE
    if metrics is None:
        if rankings is None:
            raise ValueError("pass either `metrics` or `rankings`")
        metrics = _model_set(rankings, model_set_only).head(top_n)["metric"].tolist()

    pr = None
    if rankings is not None and "pr_auc" in rankings.columns:
        pr = rankings.set_index("metric")["pr_auc"]

    n = len(metrics)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    axes = np.atleast_2d(axes)

    for i, metric in enumerate(metrics):
        ax_v, ax_h = axes[i, 0], axes[i, 1]
        d = df[[metric, "binder_class"]].dropna()

        # Left: violin + stripplot (individual points)
        sns.violinplot(
            data=d, x="binder_class", y=metric, order=class_order,
            hue="binder_class", palette=palette, legend=False,
            inner=None, cut=0, density_norm="width", ax=ax_v,
        )
        for art in ax_v.collections:
            art.set_alpha(0.30)
        sns.stripplot(
            data=d, x="binder_class", y=metric, order=class_order,
            hue="binder_class", palette=palette, legend=False,
            size=3, jitter=0.15, edgecolor="white", linewidth=0.3, ax=ax_v,
        )
        title = metric
        if pr is not None and metric in pr.index:
            title = f"{metric}  (PR-AUC = {float(pr.loc[metric]):.2f})"
        ax_v.set_xlabel(title, fontsize=10, fontweight="bold")   # metric (+ PR-AUC) names the panel
        ax_v.set_ylabel("raw score")

        # Right: histogram colored by binder_class, on one shared bin grid so the
        vmin, vmax = d[metric].min(), d[metric].max()
        edges = np.linspace(vmin, vmax, 26) if vmax > vmin else 25
        for cls in class_order:
            vals = d.loc[d["binder_class"] == cls, metric]
            if len(vals):
                ax_h.hist(vals, bins=edges, alpha=0.5, color=palette.get(cls), label=cls)
        ax_h.set_xlabel(f"{metric} \u2014 raw score", fontsize=10)
        ax_h.set_ylabel("count")
        ax_h.legend(fontsize=8)

    plt.tight_layout()
    _finish(save_path, show, name="class_distributions")

