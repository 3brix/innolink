"""
Sample-level & outlier views over the scored metrics.

Pure-viz layer (like analysis.distributions.plots): each function takes an
already-loaded dataframe and returns a matplotlib Figure / seaborn ClusterGrid.
Loading + scaling live in the driver notebook (profiles_plots.ipynb).

Figures
-------
plot_sample_clustermap  sample x metric heatmap of STANDARDIZED, aligned values,
                        both axes clustered, binder_type colour strip. Expects a
                        scaled+aligned table (e.g. merged_scaled_aligned).
plot_filter_heatmap     same layout, pass/fail vs a per-metric threshold
                        (default = the metric's median = PLACEHOLDER) + pass-rate strip.
plot_sample_profiles    a few selected samples (top/median/bottom by mean score)
                        across metric FAMILIES, over each group's IQR band.
plot_per_dataset        top separating metrics (rows) x dataset (cols), box+strip by
                        binder_type. Uses RAW values; facets on `dataset` else `source`.
plot_outliers           per-scale strip plot on a symlog axis (| = median) to show
                        whether spread is cross-model SCALE or true outliers, with
                        outlier_table() giving the robust (median/IQR) flag counts.

Notes
-----
- Cross-metric comparability comes from STANDARDIZATION, so clustermap / profiles /
  filter expect the scaled+aligned table. `outlier` + `per_dataset` read RAW values.
- With no Non-Binder rows the separation ranking falls back to variance and the
  Non-Binder band is omitted, so it runs on binder-only / mutant-only sets too.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list,   fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from analysis.distributions.wide import get_numeric_metrics, add_binder_type
from config.analysis import COLORS_MAP
from config.paths import METRIC_YAML, THRESHOLDS_YAML

CLASS_ORDER = ["Binder", "Non-Binder", "Mutant+", "Mutant-", "Target Shuffle", "Design", "Unknown"]


# helpers
    
def usable_metrics(df: pd.DataFrame) -> list[str]:
    """Numeric metrics that are not all-NaN and have some spread (needed for clustering)."""
    m = get_numeric_metrics(df)
    return [c for c in m if df[c].notna().any() and df[c].std(skipna=True) > 0]


def class_colors(bt: pd.Series) -> dict:
    """binder_type -> colour, falling back to grey for any class missing from COLORS_MAP."""
    return {k: (COLORS_MAP[k] if k in COLORS_MAP else "grey") for k in bt.unique()}


def _families(yaml_path=METRIC_YAML) -> dict:
    return {m: d.get("family", m) for m, d in yaml.safe_load(open(yaml_path))["metrics"].items()}


def _scale_of(yaml_path=METRIC_YAML) -> dict:
    return {m: d.get("scale") for m, d in yaml.safe_load(open(yaml_path))["metrics"].items()}


def _legend(bt, cc):
    return [Patch(facecolor=cc[k], label=k) for k in bt.unique()]


#  clustermap

def plot_sample_clustermap(df: pd.DataFrame, metrics: list[str] | None = None,
                           vmin: float = -2.5, vmax: float = 2.5,
                           col_order: str = "cluster", row_order: str = "cluster",
                           show_dendrogram: bool = False, sample_labels: bool = True,
                           yaml_path=METRIC_YAML):
    """
    Sample x metric heatmap of a scaled+aligned table (hand-built with matplotlib for a
    tight layout; no dendrograms drawn).

    vmin/vmax     colour clip (default +/-2.5) so a few extreme cells don't wash the map out.
    col_order     'cluster' (hierarchical leaf order) | 'binder_type' (grouped by class,
                  best mean-score first within class) | 'score' (best mean-score first).
    row_order     'cluster' | 'category' (grouped by metric category).
    sample_labels draw sample names on the x-axis (default True; small at high N).
    show_dendrogram accepted for compatibility but ignored (this layout draws no tree).

    A binder_type strip sits above the columns and a category strip left of the rows;
    black separators mark the class / category groups when ordered that way.
    """
    from matplotlib import colors as mcolors
    metrics = metrics or usable_metrics(df)
    M = df.set_index("sample")[metrics].T                                # metrics x samples (NaN kept)
    bt = df.set_index("sample")["binder_type"].reindex(M.columns)
    cat_map = {m: d.get("category", "?") for m, d in yaml.safe_load(open(yaml_path))["metrics"].items()}

    if col_order == "cluster":
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        M = M.iloc[:, leaves_list(linkage(pdist(M.T.fillna(0.0).values), "average"))]
    elif col_order in ("binder_type", "score"):
        score = df.set_index("sample")[metrics].mean(axis=1)
        if col_order == "binder_type":
            order_cats = [c for c in CLASS_ORDER if c in set(bt)] + [c for c in pd.unique(bt.dropna()) if c not in CLASS_ORDER]
            key = pd.DataFrame({"bt": pd.Categorical(bt, order_cats, ordered=True), "score": score.reindex(bt.index)})
            M = M[key.sort_values(["bt", "score"], ascending=[True, False]).index]
        else:
            M = M[score.reindex(M.columns).sort_values(ascending=False).index]
    else:
        raise ValueError("col_order must be 'cluster', 'binder_type' or 'score'")
    bt = bt.reindex(M.columns)

    if row_order == "cluster":
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        M = M.iloc[leaves_list(linkage(pdist(M.fillna(0.0).values), "average"))]
    elif row_order == "category":
        catser = pd.Series({m: cat_map.get(m, "?") for m in M.index})
        M = M.loc[catser.sort_values(kind="stable").index]
    else:
        raise ValueError("row_order must be 'cluster' or 'category'")
    catser = pd.Series({m: cat_map.get(m, "?") for m in M.index})

    ncols, nrows = M.shape[1], M.shape[0]
    cc = class_colors(bt)
    classes = [c for c in CLASS_ORDER if c in set(bt)] + [c for c in pd.unique(bt.dropna()) if c not in CLASS_ORDER]
    cats = sorted(catser.unique())
    cpal = dict(zip(cats, sns.color_palette("tab20", len(cats))))
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("lightgrey")

    W, H = min(30, max(11, 0.10 * ncols)), min(26, max(8, 0.15 * nrows))
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 45], height_ratios=[1, 45], wspace=0.01, hspace=0.01)
    ax_cls, ax_cat, ax = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    im = ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_cls.imshow(np.array([mcolors.to_rgb(cc[x]) for x in bt])[None, :, :], aspect="auto")
    ax_cat.imshow(np.array([mcolors.to_rgb(cpal[c]) for c in catser])[:, None, :], aspect="auto")
    for a in (ax_cls, ax_cat):
        a.set_xticks([]); a.set_yticks([])

    if col_order == "binder_type":
        b = 0
        for c in classes:
            b += int((bt == c).sum())
            if 0 < b < ncols:
                ax.axvline(b - 0.5, color="k", lw=0.6)
    if row_order == "category":
        r = 0
        for c in catser.drop_duplicates():
            r += int((catser == c).sum())
            if 0 < r < nrows:
                ax.axhline(r - 0.5, color="k", lw=0.6)

    ax.set_yticks(range(nrows)); ax.set_yticklabels(M.index, fontsize=4); ax.yaxis.tick_right()
    if sample_labels:
        ax.set_xticks(range(ncols)); ax.set_xticklabels(M.columns, rotation=90, fontsize=(3 if ncols > 120 else 5))
    else:
        ax.set_xticks([])
    ax.set_xlabel("sample"); ax_cat.set_ylabel("metric")
    ax_cls.set_title(f"Sample x metric heatmap (cols={col_order}, rows={row_order}, {nrows} metrics)",
                     loc="left", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.10 if sample_labels else 0.03)
    cbar.set_label(f"standardized, aligned (clip +/-{vmax})", fontsize=8)
    handles = [Patch(facecolor=cc[k], label=k) for k in classes] + [Patch(facecolor=cpal[c], label=c) for c in cats]
    fig.legend(handles=handles, title="binder_type / category", loc="upper left",
               bbox_to_anchor=(1.0, 0.98), fontsize=7, title_fontsize=8)
    return fig




# filter heatmap

# load_thresholds now lives in preprocessing.metric_meta (canonical); re-exported here.
from preprocessing.metric_meta import load_thresholds


def plot_filter_heatmap(df: pd.DataFrame, thresholds: dict, metrics: list[str] | None = None,
                        yaml_path=METRIC_YAML, col_order: str = "binder_type",
                        mode: str = "binary", clip: float = 3.0, sample_labels: bool = True):
    """
    Pass/fail matrix vs per-FAMILY thresholds on RAW (native-unit) values.

    df          RAW table (merged.csv), native units -- NOT the scaled table.
    thresholds  {family: value}. A value is either a scalar (one-sided cutoff; direction from
                metric_data.yaml) OR a 2-list [lo, hi] (two-sided WINDOW, e.g. net_charge). A
                family with no threshold is skipped; its metrics are dropped and returned.
    mode        'binary'  -> pass (blue) / fail (red) / grey (no data).
                'graded'  -> signed MARGIN from the threshold on a colourblind-safe diverging
                             map (RdBu: blue = better/pass, red = worse/fail), so you see HOW
                             well each cell passes/fails. Margin is normalised per metric by
                             its IQR and clipped to +-`clip`, so rows are comparable.
    Direction (higher/lower better) is taken from metric_data.yaml. Family/direction resolved
    by longest-suffix match (so pred_lddt -> plddt). Missing data -> grey.
    Rows = thresholded metrics (category-ordered); cols = samples (binder_type-ordered, best
    pass-rate first within class), with a pass-rate strip above the columns.
    Returns (fig, report) where report = {"no_threshold": [families skipped],
    "unmatched": [columns that matched no family and were dropped]}.
    """
    from matplotlib import colors as mcolors
    from analysis.evaluation.metrics import load_metric_families, get_family
    from preprocessing.align import load_metric_directions, get_direction
    df = add_binder_type(df) if "binder_type" not in df.columns else df
    metrics = metrics or usable_metrics(df)
    meta = yaml.safe_load(open(yaml_path))["metrics"]
    fam_map, dir_map = load_metric_families(yaml_path), load_metric_directions(yaml_path)
    fam2cat = {}
    for _i in meta.values():
        fam2cat.setdefault(_i.get("family"), _i.get("category", "?"))
    fam = {m: get_family(m, fam_map) for m in metrics}                  # suffix match: pred_lddt -> plddt
    drc = {m: (get_direction(m, dir_map) or 1) for m in metrics}
    unmatched = [m for m in metrics if fam[m] is None]                  # columns with no family -> would vanish
    keep = [m for m in metrics if fam[m] and thresholds.get(fam[m]) is not None]
    skipped = sorted({fam[m] for m in metrics if fam[m] and thresholds.get(fam[m]) is None})
    if unmatched:
        import warnings; warnings.warn(f"plot_filter_heatmap: {len(unmatched)} column(s) match no metric_data family "
                                       f"and are dropped: {', '.join(unmatched)}")
    if not keep:
        raise ValueError("no metric matched a threshold family -- check family names vs metric_data.yaml")

    R = df.set_index("sample")
    def goodness(m):
        t = thresholds[fam[m]]; v = R[m]
        if isinstance(t, (list, tuple)) and len(t) == 2:            # window [lo, hi] (two-sided)
            lo, hi = sorted(t); g = np.minimum(v - lo, hi - v)
        else:
            g = (v - t) if drc[m] > 0 else (t - v)                 # signed margin; + = better side of cutoff
        return g.mask(v.isna())
    G = pd.DataFrame({m: goodness(m) for m in keep}).T             # metrics x samples (signed margin)
    Pass = G.ge(0).astype(float).mask(G.isna())                    # 1 pass / 0 fail / NaN no-data
    bt = R["binder_type"].reindex(G.columns)
    catser = pd.Series({m: fam2cat.get(fam[m], "?") for m in G.index})
    order_rows = catser.sort_values(kind="stable").index
    G, Pass, catser = G.loc[order_rows], Pass.loc[order_rows], catser.reindex(order_rows)

    passrate = Pass.mean(axis=0, skipna=True)                       # per-sample pass fraction (mode-independent)
    classes = [c for c in CLASS_ORDER if c in set(bt)] + [c for c in pd.unique(bt.dropna()) if c not in CLASS_ORDER]
    if col_order == "binder_type":
        key = pd.DataFrame({"bt": pd.Categorical(bt, classes, ordered=True), "pr": passrate.reindex(bt.index)})
        cols = key.sort_values(["bt", "pr"], ascending=[True, False]).index
    elif col_order == "passrate":
        cols = passrate.sort_values(ascending=False).index
    else:
        cols = G.columns
    G, Pass, bt, passrate = G[cols], Pass[cols], bt.reindex(cols), passrate.reindex(cols)

    if mode == "graded":                                            # signed margin, per-metric IQR-normalised
        iqr = R[keep].quantile(.75) - R[keep].quantile(.25); std = R[keep].std()
        scale = iqr.copy(); scale[scale <= 0] = std.reindex(scale.index)[scale <= 0]; scale[scale <= 0] = 1.0
        D = G.div(scale.reindex(G.index), axis=0).clip(-clip, clip)
        cmap = plt.cm.RdBu.copy(); cmap.set_bad("lightgrey"); vmin, vmax = -clip, clip
    elif mode == "binary":
        D = Pass
        cmap = ListedColormap(["#b2182b", "#2166ac"]); cmap.set_bad("lightgrey"); vmin, vmax = 0, 1  # red=fail, blue=pass
    else:
        raise ValueError("mode must be 'binary' or 'graded'")

    ncols, nrows = D.shape[1], D.shape[0]
    cc = class_colors(bt)
    cats = sorted(catser.unique())
    cpal = dict(zip(cats, sns.color_palette("tab20", len(cats))))

    W, H = min(30, max(11, 0.10 * ncols)), min(24, max(6, 0.30 * nrows))
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 45], height_ratios=[1, 1, 30], wspace=0.01, hspace=0.04)
    ax_cls, ax_pr = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_cat, ax = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    im = ax.imshow(D.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_cls.imshow(np.array([mcolors.to_rgb(cc[x]) for x in bt])[None, :, :], aspect="auto")
    ax_pr.imshow(passrate.to_numpy(dtype=float)[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax_cat.imshow(np.array([mcolors.to_rgb(cpal[c]) for c in catser])[:, None, :], aspect="auto")
    for a in (ax_cls, ax_pr, ax_cat):
        a.set_xticks([]); a.set_yticks([])
    ax_cls.set_ylabel("class", rotation=0, ha="right", va="center", fontsize=7)
    ax_pr.set_ylabel("pass rate", rotation=0, ha="right", va="center", fontsize=7)

    if col_order == "binder_type":
        b = 0
        for c in classes:
            b += int((bt == c).sum())
            if 0 < b < ncols: ax.axvline(b - 0.5, color="k", lw=0.6)
    r = 0
    for c in catser.drop_duplicates():
        r += int((catser == c).sum())
        if 0 < r < nrows: ax.axhline(r - 0.5, color="k", lw=0.6)

    ax.set_yticks(range(nrows)); ax.set_yticklabels(D.index, fontsize=4); ax.yaxis.tick_right()
    if sample_labels:
        ax.set_xticks(range(ncols)); ax.set_xticklabels(D.columns, rotation=90, fontsize=(3 if ncols > 120 else 5))
    else:
        ax.set_xticks([])
    ax.set_xlabel("sample"); ax_cat.set_ylabel("metric")
    ax_cls.set_title(f"Filter matrix ({mode}) -- {nrows} thresholded metrics x {ncols} samples"
                     + (f"   (no threshold, skipped: {', '.join(skipped)})" if skipped else ""),
                     loc="left", fontsize=11)
    if mode == "binary":
        status = [Patch(facecolor="#2166ac", label="pass"), Patch(facecolor="#b2182b", label="fail"),
                  Patch(facecolor="lightgrey", label="no data")]
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.12 if sample_labels else 0.04)
        cbar.set_label(f"margin from threshold (IQR units, clip +/-{clip})\nblue = pass / better, red = fail / worse", fontsize=7)
        status = [Patch(facecolor="lightgrey", label="no data")]
    handles = status + [Patch(facecolor=cc[k], label=k) for k in classes] + \
              [Patch(facecolor=cpal[c], label=c) for c in cats]
    fig.legend(handles=handles, title="status  ·  binder_type  ·  category", loc="upper left",
               bbox_to_anchor=(1.0, 0.98), fontsize=7, title_fontsize=8)
    return fig, {"no_threshold": skipped, "unmatched": unmatched}



# sample profiles

def plot_sample_profiles(df: pd.DataFrame, metrics: list[str] | None = None,
                         yaml_path=METRIC_YAML, n: int = 3, samples: list[str] | None = None):
    """Family-level profiles for selected samples (default top/median/bottom by mean score)."""
    metrics = metrics or usable_metrics(df)
    fam = _families(yaml_path)
    zf = df.set_index("sample")[metrics].T.groupby(fam).mean().T   # metrics->families (axis=1 groupby removed in pandas 2.2)
    fams = zf.mean().sort_values().index.tolist()
    bt = df.set_index("sample")["binder_type"]
    score = df.set_index("sample")[metrics].mean(axis=1)
    sel = samples or list(dict.fromkeys([score.idxmax(),
                          score.iloc[(score - score.median()).abs().argsort()].index[0],
                          score.idxmin()]))[:n]
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(fams)), 6))
    for grp in ("Binder", "Non-Binder"):
        idx = bt[bt == grp].index
        if len(idx):
            c = COLORS_MAP.get(grp, "grey")
            q1, med, q3 = zf.loc[idx, fams].quantile(.25), zf.loc[idx, fams].median(), zf.loc[idx, fams].quantile(.75)
            ax.fill_between(range(len(fams)), q1, q3, color=c, alpha=.15, label=f"{grp} IQR")
            ax.plot(range(len(fams)), med, color=c, lw=1, ls="--", alpha=.7)
    for s in sel:
        ax.plot(range(len(fams)), zf.loc[s, fams], marker="o", ms=4, lw=1.8, label=f"{s} [{bt[s]}]")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=90, fontsize=6)
    ax.axhline(0, color="grey", lw=.6); ax.set_ylabel("family-mean standardized (aligned)")
    ax.legend(fontsize=7, ncol=2); ax.set_title("Selected-sample profiles vs group bands")
    fig.tight_layout()
    return fig


# per-dataset distribution

def _separation(raw: pd.DataFrame, m: str) -> float:
    """Standardized mean difference |mean_Binder - mean_NonBinder| / pooled_sd for one metric.
    Assumes both classes are present (guaranteed by the guard in plot_per_dataset)."""
    b = raw.loc[raw["binder_type"] == "Binder", m]
    n = raw.loc[raw["binder_type"] == "Non-Binder", m]
    sd = np.sqrt((b.var() + n.var()) / 2)
    return abs(b.mean() - n.mean()) / sd if sd else 0.0


def plot_per_dataset(raw: pd.DataFrame, yaml_path=METRIC_YAML, top: int = 4,
                     metrics: list[str] | None = None):
    """Top separating metrics (rows) x dataset (cols), box+strip by binder_type. RAW values.

    NOTE (descriptive, not predictive): the metrics shown are chosen by their IN-SAMPLE
    Binder-vs-Non-Binder separation on the very rows being plotted. This is a descriptive
    figure for eyeballing distributions -- it is NOT evidence that these metrics generalize
    or that they are the "best" discriminators out of sample.
    """
    raw = add_binder_type(raw) if "binder_type" not in raw.columns else raw
    # Ranking by separation requires both classes; error rather than silently falling back to
    # variance (which would rank "most variable", not "most separating").
    classes = set(raw["binder_type"].dropna())
    if not {"Binder", "Non-Binder"} <= classes:
        raise ValueError(
            "plot_per_dataset ranks metrics by Binder-vs-Non-Binder separation and needs both "
            f"classes present; got {sorted(classes)}. Pass a labelled set or choose `metrics=` explicitly."
        )
    metrics = metrics or usable_metrics(raw)
    facet_col = "dataset" if "dataset" in raw.columns else "source"
    chosen = sorted(metrics, key=lambda m: _separation(raw, m), reverse=True)[:top]
    facets = raw[facet_col].dropna().unique().tolist()
    order = [k for k in CLASS_ORDER if k in raw["binder_type"].values]
    pal = {k: COLORS_MAP.get(k, "grey") for k in order}
    fig, axes = plt.subplots(len(chosen), len(facets),
                             figsize=(4 * len(facets) + 2, 3 * len(chosen)), squeeze=False)
    for i, m in enumerate(chosen):
        for j, d in enumerate(facets):
            ax = axes[i][j]; sub = raw[raw[facet_col] == d]
            sns.boxplot(data=sub, x="binder_type", y=m, hue="binder_type", order=order,
                        palette=pal, legend=False, ax=ax, fliersize=0, width=.6)
            sns.stripplot(data=sub, x="binder_type", y=m, order=order, ax=ax, color="black", size=4, alpha=.7)
            ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=7, rotation=20)
            ax.set_ylabel(m if j == 0 else "", fontsize=8)
            if i == 0: ax.set_title(f"{facet_col} = {d}", fontsize=10)
    fig.suptitle(f"Per-{facet_col} distribution (top-{top} separating metrics)", y=1.0, fontsize=12)
    fig.tight_layout()
    return fig


# outliers

def outlier_table(raw: pd.DataFrame, scale: str = "energy", yaml_path=METRIC_YAML, iqr_k: float = 3.0):
    """Robust (median/IQR fence) outlier summary for one metric `scale`. Returns a sorted table."""
    scales = _scale_of(yaml_path)
    present = set(get_numeric_metrics(raw))
    metrics = [m for m, sc in scales.items() if sc == scale and m in present]
    rows = []
    for m in metrics:
        s = raw[m]; q1, med, q3 = s.quantile(.25), s.median(), s.quantile(.75); iqr = q3 - q1
        n_out = int(((s > q3 + iqr_k * iqr) | (s < q1 - iqr_k * iqr)).sum())
        rows.append(dict(metric=m, min=round(s.min(), 2), median=round(med, 2), max=round(s.max(), 2),
                         neg=int((s < 0).sum()), n_outliers=n_out))
    return pd.DataFrame(rows).sort_values("max", ascending=False).reset_index(drop=True)


def plot_outliers(raw: pd.DataFrame, scale: str = "energy", yaml_path=METRIC_YAML,
                  highlight: tuple[str, ...] = ("interface_dG", "pyros")):
    """Per-scale strip plot on a symlog axis (| = median). Shows scale-vs-outliers for one `scale`."""
    scales = _scale_of(yaml_path)
    present = set(get_numeric_metrics(raw))
    metrics = sorted([m for m, sc in scales.items() if sc == scale and m in present],
                     key=lambda m: raw[m].median())
    if not metrics:
        raise ValueError(f"no present metrics with scale={scale!r}")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.34 * len(metrics))))
    for i, m in enumerate(metrics):
        v = raw[m].dropna().values
        red = any(h in m for h in highlight)
        ax.scatter(v, np.full_like(v, i, dtype=float), s=18, alpha=.6, color="#eb405d" if red else "#1f3397")
        ax.plot([np.median(v)], [i], marker="|", ms=16, color="black")
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics, fontsize=7)
    ax.set_xscale("symlog"); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel(f"value (symlog)  — red = {'/'.join(highlight)} families")
    ax.set_title(f"{scale} metrics — spread vs outliers (| = median)")
    fig.tight_layout()
    return fig

# cluster association

def cluster_association(df: pd.DataFrame, label_cols=("binder_type", "dataset", "source"),
                        ks=(2, 4, 6), method: str = "average"):
    """
    Does whole-profile sample clustering line up with any label?

    Hierarchically clusters samples on the (scaled) metric matrix and reports the
    adjusted Rand index vs each present label in `label_cols`, for each k in `ks`
    (ARI ~ 0 = clustering unrelated to that label; higher = aligned). Also returns
    the smallest k=2 cluster (the split-off block) with its labels. Duplicate sample
    ids (pooled sets) are handled positionally, so `df` must carry the label columns.

    Returns (ari, block): ari is a DataFrame [k, ARI_<label>...]; block is a DataFrame
    [sample, <present labels>] listing the split-off samples.
    """
    mets = usable_metrics(df)
    # NaN -> 0 treats a missing metric as "average": 0 is the centre of a StandardScaler table
    # (the default scaling) and the median of a RobustScaler table (used only for comparison).
    # Caveat: the unbounded energy metrics are only loosely centred by either scaler, so a 0 fill
    # nudges those samples toward the middle -- a windowed/clipped treatment of energy metrics
    # would make this imputation cleaner (noted, not changed here). Report the imputed count if
    # missingness is high.
    X = df.set_index("sample")[mets].fillna(0.0).values
    Z = linkage(pdist(X), method)
    present = [c for c in label_cols if c in df.columns]
    rows = []
    for k in ks:
        lab = fcluster(Z, k, criterion="maxclust")
        rows.append({"k": k, **{f"ARI_{c}": adjusted_rand_score(df[c].astype(str).values, lab) for c in present}})
    ari = pd.DataFrame(rows)
    lab2 = fcluster(Z, 2, criterion="maxclust")
    small = pd.Series(lab2).value_counts().idxmin()
    block = df.iloc[np.where(lab2 == small)[0]][["sample", *present]].reset_index(drop=True)
    return ari, block
