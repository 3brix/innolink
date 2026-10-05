"""Ranking figures: the Random Forest (primary ranker), the composite rules, and the
RF-vs-composite comparison that feeds the consensus shortlist.

Reads the stage outputs under evaluation/<dataset>/ranking/, composite/ and consensus/. Shared
bar-chart and saving helpers come from analysis.plot_common; the single-metric benchmark figures
stay in analysis.evaluation.plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

from config import palette as P
from analysis.plot_common import _finish, _bar_axes, _metric_bars, _model_category_legends


def plot_rf_importances(importances, top_n=10, equal_share=True, label=None, save_path=None, show=False):
    """RF impurity (Gini) importances, most important first; same colour/hatch language as
    plot_top_metrics_bar.

    The dashed line is the equal share 1/n_features, taken from the full table so it does not move
    with `top_n`. Impurity importance is training-data and correlation-biased: read it as "what the
    forest used", not "what predicts binding".
    """
    P.apply_plot_style()
    d = importances.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    n_features = len(d)
    top = d.head(top_n)
    x = np.arange(len(top))

    fig, ax, legend_ax = _bar_axes(len(top))
    families, categories = _metric_bars(ax, top["metric"], top["importance"].astype(float))
    if equal_share and n_features:
        share = 1.0 / n_features
        ax.axhline(share, color=P.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
        ax.annotate(f"equal share {share:.3f}", xy=(1.0, share), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=P.INK_SOFT,
                    ha="left", va="center", annotation_clip=False)
    ax.set_xticks(x, top["metric"], rotation=60, ha="right", fontsize=11)
    if label:
        ax.set_xlabel(label, fontweight="bold", labelpad=8)
    ax.set_ylabel("RF importance (Gini)")
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    _model_category_legends(legend_ax, families, categories)
    plt.tight_layout()
    _finish(save_path, show, name="rf_importances")


# The three importance columns of rf_importance_comparison.csv: (column, panel title, zero reference).
IMPORTANCE_SPECS = [("gini_final", "Gini (final model)", None),
                    ("perm_final", "Permutation (final model, in-sample)", 0.0),
                    ("perm_cv", "Permutation (held-out, grouped CV)", 0.0)]


def plot_rf_importance_comparison(comparison, top_n=10, measures=None, sort_by="gini_final",
                                  save_path=None, show=False):
    """One sorted dot panel per importance measure, sharing a y order set by `sort_by`.

    Each panel keeps its OWN x-axis -- Gini sums to 1, the permutation columns are in
    average-precision units. gini_final and perm_final describe the same fitted model and still
    disagree, which is the point. The dashed line on permutation panels is 0.
    """
    from analysis.distributions.long import assign_model
    P.apply_plot_style()
    measures = [m for m in (measures or IMPORTANCE_SPECS) if m[0] in comparison.columns]
    top = comparison.sort_values(sort_by, ascending=False).head(top_n).iloc[::-1]   # best at the TOP
    families = [P.model_family(assign_model(m) or "") for m in top["feature"]]
    colours = [P.MODEL_FAMILY.get(f, P.NEUTRAL) for f in families]
    y = np.arange(len(top))

    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 0.36 * len(top) + 2.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, title, zero) in zip(axes, measures):
        vals = top[col].astype(float)
        origin = 0.0 if zero is None else float(zero)
        ax.hlines(y, origin, vals, color=P.GRID, linewidth=2.0, zorder=1)
        ax.scatter(vals, y, s=46, color=colours, edgecolor=P.SURFACE, linewidth=0.8, zorder=3)
        if zero is not None:
            ax.axvline(zero, color=P.INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        sd = f"{col}_sd"
        if sd in top.columns:   # permutation spread across repeats / folds
            ax.errorbar(vals, y, xerr=top[sd].astype(float), fmt="none", ecolor=P.GRID,
                        elinewidth=1.2, capsize=0, zorder=2)
        ax.set_xlabel(title, fontsize=10)      # panel identity belongs on the axis, not a title
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(top["feature"], fontsize=8)
    handles = [mpatches.Patch(color=P.MODEL_FAMILY[f], label=P.model_label(f))
               for f in P.MODEL_FAMILY if f in families]
    axes[-1].legend(handles=handles, title="Model", fontsize=8, title_fontsize=8, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    _finish(save_path, show, name=f"rf_importance_comparison_by_{sort_by}")


# Composite-vs-RF comparison ------------------------------------------------------------------
# Fixed order, never cycled: RF first, then the three composite rules. Marker shape is a SECONDARY
# encoding, required because THEME slots 3 and 4 sit at the CVD floor (protan dE 7.3).
METHOD_ORDER = ["rf", "composite_weighted", "composite_consensus", "composite_product"]
# ONE spelling per method, used by every figure: a method must not be "Random Forest" in one and
# "rf" in the next. method_display() maps whatever a caller passes onto these.
METHOD_LABEL = {"rf": "RF", "composite_weighted": "composite weighted",
                "composite_consensus": "composite consensus", "composite_product": "composite product"}
METHOD_MARKER = {"rf": "o", "composite_weighted": "s", "composite_consensus": "^", "composite_product": "D"}


def _method_colours():
    return {m: P.THEME[i] for i, m in enumerate(METHOD_ORDER)}


def method_key(label):
    """Normalise however a figure spells a method ('Random Forest', 'composite product',
    'Composite (product)', 'composite_product') onto its METHOD_ORDER key, or None."""
    text = str(label).strip().lower().replace("(", " ").replace(")", " ").replace("-", " ")
    words = text.replace("_", " ").split()
    if words[:1] == ["rf"] or words[:2] == ["random", "forest"]:
        return "rf"
    if words and words[0] == "composite":
        for rule in ("consensus", "weighted", "product"):
            if rule in words:
                return f"composite_{rule}"
    return None


def method_display(label):
    """The canonical label for a ranking method, whatever spelling the caller used."""
    return METHOD_LABEL.get(method_key(label), str(label))


def method_colour(label, default=None):
    """Colour for a ranking method, by ENTITY not by position in the figure. Without this a method
    changes colour depending on which others happen to be plotted beside it."""
    return _method_colours().get(method_key(label), default)


# The shortlist is not a design SET, so it takes no THEME slot -- a dataset colour would read as
# "another design set". This pale green sits outside the categorical slots, clear of germinal0's
# purple and esm0's sky blue.
SELECTED_COLOUR = "#86C79A"
# label 14 sits under the global axes.labelsize (15); the legend matches the global 11. This
# figure's legend labels are the longest in the project (set + n + score range).
DESIGN_LABEL_SIZE, DESIGN_LEGEND_SIZE = 14, 11


def plot_design_scores(scores, score="p_binder", group="dataset", bins=30, threshold=None,
                       selected=None, selected_label="shortlist", xlabel=None, save_path=None, show=False):
    """Design-score histogram, one colour per design set, on one shared bin grid.

    `threshold` draws a cutoff line. `selected` adds the chosen designs as a third histogram in a
    colour no design set uses; a {label: ids} mapping is pooled into one series.
    """
    P.apply_plot_style()
    d = scores.dropna(subset=[score])
    groups = list(dict.fromkeys(d[group])) if group in d.columns else [None]
    edges = np.linspace(float(d[score].min()), float(d[score].max()), bins + 1)

    drawn_selection = False
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for g in groups:
        vals = d[score] if g is None else d.loc[d[group] == g, score]
        label = "Designs" if g is None else f"{P.dataset_label(g)} (n={len(vals)})"
        if threshold is not None:
            label += f" " #  — {int((vals >= threshold).sum())} ≥ {threshold:g}
        ax.hist(vals, bins=edges, alpha=0.55, color=P.DATASET.get(g, P.NEUTRAL), label=label, zorder=2)
    if threshold is not None:
        ax.axvline(threshold, color=P.INK_SOFT, linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)
    if selected is not None:
        # one pooled series: consensus and primary_only are both shortlisted, and the
        # question here is WHERE the shortlist sits, not how it was reached.
        ids = set().union(*(set(v) for v in selected.values())) if isinstance(selected, dict) else set(selected)
        chosen = d[d["sample"].isin(ids)]
        if len(chosen):
            drawn_selection = True
            # SELECTED_COLOUR belongs to no design set, so it cannot be read as one
            ax.hist(chosen[score], bins=edges, color=SELECTED_COLOUR, zorder=5,
                    label=f"{P.capitalize_first(selected_label)} (n={len(chosen)}, "
                          f"{chosen[score].min():.2f}–{chosen[score].max():.2f})")
    ax.set_xlabel(P.capitalize_first(xlabel or score), fontsize=DESIGN_LABEL_SIZE)
    ax.set_ylabel("Designs", fontsize=DESIGN_LABEL_SIZE)
    ax.grid(axis="x", visible=False)
    # just outside the axes: a "best" legend lands in the right tail, over the threshold line.
    # Short handles and no border pad keep the block narrow.
    ax.legend(frameon=False, fontsize=DESIGN_LEGEND_SIZE, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              handlelength=1.3, handletextpad=0.5, borderaxespad=0.0)
    plt.tight_layout()
    _finish(save_path, show,
            name=f"design_scores_{score}{'_with_' + selected_label.replace(' ', '_') if drawn_selection else ''}")


def plot_rf_performance(oof_scores, score="p_binder_oof", label="binder", cv_metrics=None,
                        save_path=None, show=False):
    """RF out-of-fold PR and ROC curves, side by side.

    `oof_scores` is rf_oof_scores.csv -- the same predictions the reported AUCs come from, so the
    two cannot drift apart. Two panels, never one axis: the PR no-skill line is the prevalence,
    the ROC's is the diagonal.
    """
    P.apply_plot_style()
    y = oof_scores[label].astype(int).to_numpy()
    p = oof_scores[score].astype(float).to_numpy()
    prevalence = float(y.mean())
    values = cv_metrics.set_index("metric")["value"].to_dict() if cv_metrics is not None else {}
    pr_auc = float(values.get("pr_auc_oof", average_precision_score(y, p)))
    roc_auc = float(values.get("roc_auc_oof", roc_auc_score(y, p)))

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(10.4, 4.6))

    precision, recall, _ = precision_recall_curve(y, p)
    ax_pr.step(recall, precision, where="post", color=P.THEME[0], linewidth=2, zorder=3,
               label=f"RF (PR-AUC = {pr_auc:.3f})")
    ax_pr.axhline(prevalence, color=P.NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2,
                  label=f"chance = prevalence ({prevalence:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")

    fpr, tpr, _ = roc_curve(y, p)
    ax_roc.plot(fpr, tpr, color=P.THEME[0], linewidth=2, zorder=3, label=f"RF (ROC-AUC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=P.NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2,
                label="chance (0.5)")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")

    for ax in (ax_pr, ax_roc):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    _finish(save_path, show, name="rf_performance_curves")


# RF vs composite: the two views disagree, so both are drawn. Pooled ranks all labelled rows
# together; per-source scores WITHIN each dataset and aggregates, which is the primary measure.
# (panel label, canonical value key, canonical "worst source" key or None)
POOLED_VS_SOURCE_PANELS = [
    ("pooled\nPR-AUC", "pooled_pr_auc", None),
    ("per-source PR-AUC\n(ap_norm, 0 = no-skill)", "ap_norm_mean", "ap_norm_min"),
    ("pooled\nprecision@10%", "pooled_precision", None),
    ("per-source\nprecision@10%", "precision_mean", "precision_min"),
]
# the same quantity is named differently by the two stages, so each canonical key lists its aliases
_VALUE_ALIASES = {
    "pooled_pr_auc":    ("pr_auc_oof", "pr_auc"),
    "ap_norm_mean":     ("ap_norm_ds_mean",),
    "ap_norm_min":      ("ap_norm_ds_min",),
    "pooled_precision": ("precision_at_10pct", "precision_at_pct"),
    "precision_mean":   ("precision_at_10pct_ds_mean", "precision_at_pct_ds_mean"),
    "precision_min":    ("precision_at_10pct_ds_min", "precision_at_pct_ds_min"),
}


def _canonical(values: dict) -> dict:
    """Map one method's metrics onto the canonical keys, whichever alias it happens to use."""
    out = {}
    for key, aliases in _VALUE_ALIASES.items():
        for alias in aliases:
            if alias in values and values[alias] == values[alias]:
                out[key] = float(values[alias])
                break
    return out


def plot_ranking_curves(oof, score="score", truth="binder", model="model", models=None,
                        save_path=None, show=False):
    """PR and ROC curves per ranking method, from their out-of-fold predictions.

    `oof` is a long table (model, binder, score). Both panels are POOLED, and pooled and
    per-source disagree here, so read these beside plot_per_source_comparison, not instead of it.
    """
    P.apply_plot_style()
    d = oof.dropna(subset=[score, truth])
    names = models or list(dict.fromkeys(d[model]))
    prevalence = float(d[truth].astype(int).mean())

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for i, name in enumerate(names):
        sub = d[d[model] == name]
        y, p = sub[truth].astype(int).to_numpy(), sub[score].astype(float).to_numpy()
        if np.unique(y).size < 2:
            continue
        colour = method_colour(name, P.THEME[i % len(P.THEME)])
        precision, recall, _ = precision_recall_curve(y, p)
        shown = method_display(name)
        ax_pr.step(recall, precision, where="post", color=colour, linewidth=2, zorder=3,
                   label=f"{shown} ({average_precision_score(y, p):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, color=colour, linewidth=2, zorder=3,
                    label=f"{shown} ({roc_auc_score(y, p):.3f})")
    ax_pr.axhline(prevalence, color=P.NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2,
                  label=f"chance = prevalence ({prevalence:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_roc.plot([0, 1], [0, 1], color=P.NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2,
                label="chance (0.5)")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("TPR")
    for ax in (ax_pr, ax_roc):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    _finish(save_path, show, name="ranking_curves_" + "_vs_".join(n.replace(" ", "_") for n in names))


def plot_pooled_vs_per_source(rf_metrics, composite_eval, composite_model="composite_product",
                              panels=None, save_path=None, show=False):
    """RF against one composite rule, pooled and per-source side by side.

    The two views rank the methods oppositely here, so showing one alone argues for whichever
    method it favours. A worst-source value is drawn as a marker: a mean can sit above no-skill
    while one source is below it.
    """
    P.apply_plot_style()
    panels = panels or POOLED_VS_SOURCE_PANELS
    rf = _canonical(rf_metrics.set_index("metric")["value"].to_dict())
    comp_row = composite_eval.set_index("model").loc[composite_model].to_dict()
    comp = _canonical(comp_row)
    prevalence = float(comp_row.get("prevalence", float("nan")))
    methods = [(method_display("rf"), rf, method_colour("rf", P.THEME[0])),
               (method_display(composite_model), comp, method_colour(composite_model, P.THEME[1]))]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, value_key, worst_key) in zip(axes, panels):
        for i, (name, values, colour) in enumerate(methods):
            if value_key in values:
                ax.bar(i, values[value_key], width=0.62, color=colour, edgecolor=P.INK,
                       linewidth=0.8, zorder=2)
            if worst_key and worst_key in values:
                ax.plot([i], [values[worst_key]], marker="v", markersize=9, color=P.INK, zorder=4,
                        linestyle="", label="worst source" if (i == 0 and ax is axes[1]) else None)
        # chance: 0 for ap_norm (already prevalence-corrected), the prevalence for the rest
        baseline = 0.0 if value_key.startswith("ap_norm") else prevalence
        if baseline == baseline:
            ax.axhline(baseline, color=P.NEUTRAL, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(len(methods)), [m[0] for m in methods], fontsize=8, rotation=25, ha="right")
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.02))
    plt.tight_layout()
    _finish(save_path, show, name=f"pooled_vs_per_source_rf_vs_{composite_model}")


def plot_per_source_comparison(per_source, metric="ap_norm", models=None, baseline=0.0,
                               xlabel=None, save_path=None, show=False):
    """One group of bars per SOURCE: where each method actually wins.

    `per_source` is a long table (source, model, <metric>) -- the evidence behind the per-source
    aggregate. `baseline` is the metric's no-skill line (0 for ap_norm).
    """
    P.apply_plot_style()
    d = per_source.dropna(subset=[metric])
    names = models or list(dict.fromkeys(d["model"]))
    sources = list(dict.fromkeys(d["source"]))
    x = np.arange(len(sources))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(1.6 * len(sources) + 3.0, 4.4))
    for i, name in enumerate(names):
        sub = d[d["model"] == name].set_index("source")
        values = [float(sub[metric].get(src, np.nan)) for src in sources]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, values, width=width * 0.92,
               color=method_colour(name, P.THEME[i % len(P.THEME)]), edgecolor=P.INK,
               linewidth=0.8, zorder=2, label=method_display(name))
    if baseline is not None:
        ax.axhline(baseline, color=P.NEUTRAL, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x, [P.dataset_label(src) for src in sources], fontsize=9)
    ax.set_xlabel(P.capitalize_first(xlabel or "source"))
    ax.set_ylabel({"ap_norm": "Normalized PR-AUC (0 = no-skill)"}.get(metric, metric))
    ax.grid(axis="x", visible=False)
    # legend OUTSIDE the axes: inside it lands on whichever bar happens to be tallest
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    _finish(save_path, show, name=f"per_source_comparison_{metric}")


def plot_standardisation_sensitivity(sensitivity, measures=("ap_norm_ds_mean", "precision_at_pct_ds_mean"),
                                     labels=None, save_path=None, show=False):
    """How much the composite moves when the standardisation changes, one panel per measure.

    Each panel keeps its own x-axis; the configs share a y order so one can be read across panels.
    """
    P.apply_plot_style()
    measures = [m for m in measures if m in sensitivity.columns]
    d = sensitivity.sort_values(measures[0], ascending=True)
    y = np.arange(len(d))
    titles = labels or {"ap_norm_ds_mean": "Normalized PR-AUC (per-source mean)",
                        "precision_at_pct_ds_mean": "Precision@10% (per-source mean)"}

    fig, axes = plt.subplots(1, len(measures), figsize=(4.6 * len(measures), 0.42 * len(d) + 2.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, measure in zip(axes, measures):
        values = d[measure].astype(float)
        ax.hlines(y, 0, values, color=P.GRID, linewidth=2.0, zorder=1)
        ax.scatter(values, y, s=48, color=P.THEME[0], edgecolor=P.SURFACE, linewidth=0.8, zorder=3)
        ax.set_xlabel(titles.get(measure, measure), fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d[d.columns[0]], fontsize=8)
    plt.tight_layout()
    _finish(save_path, show, name="standardisation_sensitivity")


def plot_fold_pr_auc(by_fold, save_path=None, show=False):
    """Per-fold PR-AUC for every ranking method.

    Folds are ordered by PREVALENCE, the axis the methods separate on, and each tick is labelled
    with its fold number. The dashed line is each fold's prevalence, so a point below it is worse
    than chance; the composites rise with that line and the RF does not.
    """
    P.apply_plot_style()
    colour = _method_colours()
    order = [m for m in METHOD_ORDER if m in set(by_fold["model"])]
    ranked = by_fold.drop_duplicates("fold").sort_values("prevalence")
    folds = ranked["fold"].tolist()
    prev = ranked["prevalence"].astype(float).tolist()
    x = list(range(len(folds)))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, prev, ls="--", lw=1.2, color=P.NEUTRAL, zorder=1)
    ax.annotate("fold prevalence\n(PR-AUC baseline)", (x[-1], prev[-1]), xytext=(7, 0),
                textcoords="offset points", fontsize=8, color=P.INK_SOFT, va="center")
    for m in order:
        g = by_fold[by_fold.model == m].set_index("fold").reindex(folds)
        ax.plot(x, g["pr_auc"], marker=METHOD_MARKER[m], ms=7, lw=2, color=colour[m],
                markeredgecolor=P.SURFACE, markeredgewidth=1.5, label=METHOD_LABEL[m], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}\nprev {p:.2f}" for f, p in zip(folds, prev)], linespacing=1.5)
    ax.set_xlabel("Cross-validation fold, ordered by prevalence", labelpad=8)
    ax.set_ylabel("PR-AUC")
    ax.set_xlim(-0.35, len(folds) - 0.45)
    ax.grid(axis="y", color=P.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, right=0.86)   # room for the two-line ticks and the end label
    _finish(save_path, show, name="fold_pr_auc")
    return fig


def plot_pooled_vs_fold_mean(summary, save_path=None, show=False):
    """The two PR-AUC estimators side by side, per method.

    Pooling weights folds by their positives, the fold mean weights them equally; they disagree
    about which method wins, which is why both are drawn. Whisker is +-1 SD across folds.
    """
    P.apply_plot_style()
    colour = _method_colours()
    s = summary.set_index("model").reindex([m for m in METHOD_ORDER if m in set(summary["model"])])
    y = list(range(len(s)))[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 0.72 * len(s) + 1.9))
    for yi, (m, r) in zip(y, s.iterrows()):
        ax.errorbar(r["pr_auc_fold_mean"], yi, xerr=r.get("pr_auc_fold_std", 0), fmt="none",
                    ecolor=colour[m], elinewidth=2, capsize=4, zorder=2)
        ax.scatter(r["pr_auc_fold_mean"], yi, marker=METHOD_MARKER[m], s=70, color=colour[m],
                   edgecolor=P.SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(r["pr_auc_pooled"], yi, marker="|", s=170, color=colour[m], linewidth=2.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL[m] for m in s.index])
    ax.set_ylim(-0.85, len(s) - 0.4)
    ax.set_xlabel("PR-AUC")
    ax.grid(axis="x", color=P.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[mlines.Line2D([], [], color=P.INK_SOFT, marker="o", ls="none", ms=7, label="fold-mean \u00b1 SD"),
                       mlines.Line2D([], [], color=P.INK_SOFT, marker="|", ls="none", ms=11, mew=2.5, label="pooled OOF")],
              frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    _finish(save_path, show, name="pooled_vs_fold_mean")
    return fig
