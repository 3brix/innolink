"""
Scoring Pipeline Preprocessing & EDA
=====================================
Processes raw scores from AF3, ColabFold, ESMFold, Chai, Boltz predictors.
- Keeps A,C and B,C interfaces as separate rows (one row = one sample × interface)
- WT binders vs. alanine-scan mutants parsed from sample name
- Raw vs. non-raw column comparison (report only, no dropping)
- ΔΔScore computed per interface separately (sanity check for alanine scan)
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV  = "scores_raw.csv"       # <-- change to your file path
OUTPUT_DIR = Path("pipeline_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

KEEP_INTERFACES = {"A,C", "B,C"}
ALA_PATTERN = re.compile(r"^(.+?)_([A-Z])_([A-Z]\d+[A-Z])$")

palette = {"binder": "#2196F3", "ala_scan": "#FF5722"}

# ── Step 1: Load & Filter ─────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Load & Filter")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"Raw shape: {df.shape}")

df = df[df["interface"].apply(lambda x: str(x).strip() in KEEP_INTERFACES)].copy()
print(f"After interface filter {KEEP_INTERFACES}: {df.shape}")
print(f"Rows per interface:\n{df['interface'].value_counts().to_string()}")

# ── Step 2: Parse Sample Labels ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Parse Sample Labels")
print("=" * 60)

def parse_sample(sample):
    m = ALA_PATTERN.match(sample)
    if m:
        return {
            "base":      m.group(1),
            "group":     "ala_scan",
            "mut_chain": m.group(2),
            "mutation":  m.group(3),
            "mut_pos":   m.group(3)[:-1],
        }
    return {
        "base":      sample,
        "group":     "binder",
        "mut_chain": None,
        "mutation":  None,
        "mut_pos":   None,
    }

parsed = df["sample"].apply(parse_sample).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

print(df["group"].value_counts().to_string())
print(f"\nUnique base structures: {df['base'].nunique()}")
print(f"Unique mutations:       {df['mutation'].dropna().nunique()}")
print(f"\nSample rows (first 5):")
print(df[["sample", "interface", "group", "base", "mut_chain", "mutation"]].head())

# ── Step 3: Raw vs. Non-Raw Agreement ────────────────────────────────────────
#
# We keep BOTH versions — this step just reports whether they agree.
# "DIFFERENT" cases are the most interesting: post-processing changed something.
# Scatter plots are generated for DIFFERENT pairs so you can see which
# samples are affected and whether the divergence is systematic.

print("\n" + "=" * 60)
print("STEP 3 — Raw vs. Non-Raw Agreement (report only, no columns dropped)")
print("=" * 60)

RAW_ONLY_PREFIXES = ("af3_notpl_raw_", "cf_custom_raw_", "cf_notpl_raw_")

def find_processed_counterpart(raw_col, all_cols):
    if any(raw_col.startswith(p) for p in RAW_ONLY_PREFIXES):
        return None
    m = re.match(r'^(af3|cf)_raw_(.+)$', raw_col)
    if m:
        candidate = f"{m.group(1)}_{m.group(2)}"
        return candidate if candidate in all_cols else None
    return None

all_cols = set(df.columns)
raw_cols = [c for c in df.columns if "_raw_" in c]
print(f"Total columns with '_raw_': {len(raw_cols)}")

pairs = []
for rc in raw_cols:
    proc = find_processed_counterpart(rc, all_cols)
    if proc:
        pairs.append((proc, rc))

raw_only = [c for c in raw_cols if not any(c == r for _, r in pairs)]
print(f"Matched pairs (processed <-> raw): {len(pairs)}")
print(f"Raw-only independent predictors:   {len(raw_only)}")

pair_df_rows = []
for (col, raw_col) in pairs:
    a = pd.to_numeric(df[col],     errors="coerce")
    b = pd.to_numeric(df[raw_col], errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        status = "insufficient_data"
        corr = max_diff = mean_diff = np.nan
    else:
        corr      = a[mask].corr(b[mask])
        diff      = (a[mask] - b[mask]).abs()
        max_diff  = diff.max()
        mean_diff = diff.mean()
        if corr > 0.9999 and max_diff < 1e-8:
            status = "IDENTICAL"
        elif corr > 0.99:
            status = "near_identical"
        else:
            status = "DIFFERENT"
    pair_df_rows.append({
        "col":       col,
        "raw_col":   raw_col,
        "corr":      round(corr, 6)    if not np.isnan(corr)      else np.nan,
        "max_diff":  round(max_diff, 8) if not np.isnan(max_diff) else np.nan,
        "mean_diff": round(mean_diff, 8) if not np.isnan(mean_diff) else np.nan,
        "status":    status,
    })

if pair_df_rows:
    pair_df = pd.DataFrame(pair_df_rows)
    print("\n" + pair_df[["col", "corr", "max_diff", "status"]].to_string(index=False))
    pair_df.to_csv(OUTPUT_DIR / "raw_vs_nonraw_comparison.csv", index=False)
    print(f"\nSaved → raw_vs_nonraw_comparison.csv")

    # Scatter plots for DIFFERENT pairs only
    different_pairs = pair_df[pair_df["status"] == "DIFFERENT"]
    if len(different_pairs) > 0:
        print(f"\nGenerating scatter plots for {len(different_pairs)} DIFFERENT pair(s)...")
        ncols = min(3, len(different_pairs))
        nrows = int(np.ceil(len(different_pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 4), squeeze=False)
        axes = axes.flatten()
        for i, (_, row) in enumerate(different_pairs.iterrows()):
            ax = axes[i]
            col, raw_col = row["col"], row["raw_col"]
            a = pd.to_numeric(df[col],     errors="coerce")
            b = pd.to_numeric(df[raw_col], errors="coerce")
            for grp, color in palette.items():
                mask = df["group"] == grp
                ax.scatter(a[mask], b[mask], c=color, label=grp,
                           alpha=0.6, s=20, edgecolors="none")
            # y = x reference line
            lims = [min(a.min(), b.min()), max(a.max(), b.max())]
            ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel(f"processed: {col}", fontsize=8)
            ax.set_ylabel(f"raw: {raw_col}", fontsize=8)
            ax.set_title(f"r = {row['corr']:.4f}", fontsize=9)
            ax.legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Processed vs Raw — DIFFERENT pairs\n(points on dashed line = identical)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_raw_vs_processed_scatter.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → 03_raw_vs_processed_scatter.png")
    else:
        print("All pairs are IDENTICAL or near_identical — no scatter plots needed.")
else:
    print("No matched raw/non-raw pairs found — check column names.")

# ── Step 4: Select Numeric Score Columns ──────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Select Score Columns")
print("=" * 60)

EXCLUDE = {
    "sample", "binding", "interface", "base", "group",
    "mut_chain", "mutation", "mut_pos",
    # rank/ordinal columns
    "af3_rank", "af3_raw_rank", "af3_rank0_masif", "af3_raw_rank0_masif",
    "af3_notpl_raw_rank", "af3_notpl_raw_rank0_masif",
    "cf_rank", "cf_raw_rank", "cf_rank0_masif", "cf_raw_rank0_masif",
    "cf_custom_raw_rank", "cf_custom_raw_rank0_masif",
    "cf_notpl_raw_rank", "cf_notpl_raw_rank0_masif",
    "af3_ranking_score", "af3_raw_ranking_score", "af3_notpl_raw_ranking_score",
    # list-valued Chai columns
    "chai_unconstrained_per_chain_ptm", "chai_unconstrained_per_chain_pair_iptm",
    "chai_constrained_per_chain_ptm", "chai_constrained_per_chain_pair_iptm",
}
path_cols = {c for c in df.columns if "path" in c.lower()}
EXCLUDE.update(path_cols)

score_cols = [
    c for c in df.columns
    if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
]
print(f"Score columns retained: {len(score_cols)}")

# ── Step 5: ΔΔScore per Interface ─────────────────────────────────────────────
#
# Key insight: each alanine mutation is on a specific chain.
#   e.g. 5ig7_abc_cut_A_H52A mutates chain A
#        -> A,C interface score should change  (expected_affected = True)
#        -> B,C interface score should be ~0   (expected_affected = False)
#
# Sanity check: control interface deltas should be near zero.
# If they're not, your pipeline has a problem worth investigating.

print("\n" + "=" * 60)
print("STEP 5 — ΔΔScore per Interface (sanity check)")
print("=" * 60)

wt_df = (
    df[df["group"] == "binder"]
    .set_index(["base", "interface"])[score_cols]
)

mut_df = df[df["group"] == "ala_scan"].copy()
mut_df = mut_df[mut_df["base"].isin(df[df["group"] == "binder"]["base"])]

delta_rows = []
for _, row in mut_df.iterrows():
    key = (row["base"], row["interface"])
    if key not in wt_df.index:
        continue
    wt_scores = wt_df.loc[key]
    delta = {c: row[c] - wt_scores[c] for c in score_cols
             if pd.notna(row[c]) and pd.notna(wt_scores[c])}
    delta["sample"]            = row["sample"]
    delta["base"]              = row["base"]
    delta["interface"]         = row["interface"]
    delta["mutation"]          = row["mutation"]
    delta["mut_chain"]         = row["mut_chain"]
    delta["mut_pos"]           = row["mut_pos"]
    delta["expected_affected"] = row["mut_chain"] in str(row["interface"])
    delta_rows.append(delta)

if delta_rows:
    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(OUTPUT_DIR / "scores_delta.csv", index=False)
    print(f"ΔΔScore rows: {len(df_delta)}")
    print(f"  Expected-affected interface rows:    {df_delta['expected_affected'].sum()}")
    print(f"  Control (unaffected) interface rows: {(~df_delta['expected_affected']).sum()}")
    print(f"Saved → scores_delta.csv")
else:
    df_delta = None
    print("No matched WT–mutant pairs found. Check base name parsing.")

df.to_csv(OUTPUT_DIR / "scores_clean.csv", index=False)
print(f"\nClean per-row scores saved → scores_clean.csv")

# ── Step 6: EDA Plots ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 — EDA Plots")
print("=" * 60)

KEY_METRICS = [
    c for c in score_cols if any(k in c for k in [
        "iptm", "plddt", "interface_pae", "interface_dG",
        "ipsae", "confidence_score", "pyros_binder_score"
    ])
][:20]

# 6a — Distributions per metric, one panel per interface
for iface in ["A,C", "B,C"]:
    sub = df[df["interface"] == iface]
    n = len(KEY_METRICS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(KEY_METRICS):
        ax = axes[i]
        for grp, color in palette.items():
            vals = sub.loc[sub["group"] == grp, col].dropna()
            if len(vals) > 1:
                vals.plot.kde(ax=ax, label=grp, color=color, linewidth=2)
            elif len(vals) == 1:
                ax.axvline(vals.iloc[0], color=color, label=grp, linewidth=2)
        ax.set_title(col, fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    iface_tag = iface.replace(",", "")
    fig.suptitle(f"Distributions: Binder vs Alanine Scan  [{iface} interface]",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"01_distributions_{iface_tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fname.name}")

# 6b — Correlation heatmap (binders only)
binder_data = df.loc[df["group"] == "binder", KEY_METRICS].dropna(axis=1, how="all")
if binder_data.shape[0] > 1:
    corr_mat = binder_data.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr_mat) * 0.55),
                                     max(6, len(corr_mat) * 0.55)))
    sns.heatmap(corr_mat, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=len(corr_mat) <= 15, fmt=".2f", annot_kws={"size": 7},
                xticklabels=corr_mat.columns, yticklabels=corr_mat.index)
    ax.set_title("Metric Correlation (Binders, all interfaces)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → 02_correlation_heatmap.png")

# 6c — ΔΔScore heatmaps: affected vs control interface
if df_delta is not None and len(df_delta) > 0:
    delta_score_cols = [c for c in KEY_METRICS if c in df_delta.columns]
    if delta_score_cols:
        for affected, label in [(True, "affected"), (False, "control")]:
            sub = df_delta[df_delta["expected_affected"] == affected]
            if len(sub) == 0:
                continue
            pivot = sub.set_index("mutation")[delta_score_cols]
            fig, ax = plt.subplots(figsize=(max(8, len(delta_score_cols) * 0.6),
                                             max(4, len(pivot) * 0.35)))
            sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                        xticklabels=pivot.columns, yticklabels=pivot.index)
            title = (f"ΔΔScore — {'mutated' if affected else 'control (unaffected)'} "
                     f"interface  (mutant − WT)")
            ax.set_title(title, fontsize=11)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            fname = OUTPUT_DIR / f"04_delta_heatmap_{label}_interface.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved → {fname.name}")

# 6d — Separability per metric (Mann-Whitney), split by interface
sep_rows = []
for iface in ["A,C", "B,C"]:
    sub = df[df["interface"] == iface]
    for col in score_cols:
        g1 = sub.loc[sub["group"] == "binder",   col].dropna()
        g2 = sub.loc[sub["group"] == "ala_scan",  col].dropna()
        if len(g1) < 2 or len(g2) < 2:
            continue
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        effect = 1 - (2 * u_stat) / (len(g1) * len(g2))
        sep_rows.append({
            "metric":          col,
            "interface":       iface,
            "p_value":         p_val,
            "effect_size_rbc": effect,
            "n_binder":        len(g1),
            "n_ala":           len(g2),
        })

sep_df = pd.DataFrame(sep_rows)
sep_df.to_csv(OUTPUT_DIR / "05_separability.csv", index=False)

top = (sep_df.assign(abs_effect=sep_df["effect_size_rbc"].abs())
             .sort_values("abs_effect", ascending=False)
             .head(30))

fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.38)))
iface_colors = {"A,C": "#1976D2", "B,C": "#388E3C"}
for i, (_, row) in enumerate(top[::-1].iterrows()):
    ax.barh(i, row["effect_size_rbc"],
            color=iface_colors[row["interface"]], alpha=0.85)
    ax.text(row["effect_size_rbc"] + 0.01 * np.sign(row["effect_size_rbc"]),
            i, row["interface"], va="center", fontsize=7,
            color=iface_colors[row["interface"]])
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top[::-1]["metric"].tolist(), fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Rank-Biserial Correlation (effect size)\n+ = higher in binders")
ax.set_title("Top Metrics Separating Binder vs. Alanine Scan\n(by interface)", fontsize=12)
for iface, color in iface_colors.items():
    ax.bar(0, 0, color=color, label=iface)
ax.legend(title="Interface", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_separability.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → 05_separability.png")

print("\n" + "=" * 60)
print("DONE. All outputs in:", OUTPUT_DIR)
print("=" * 60)
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")

# ── Step 7: Metric Evaluation vs Curated Binding Labels ───────────────────────

print("\n" + "=" * 60)
print("STEP 7 — Metric Evaluation (curated binding labels)")
print("=" * 60)

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# ── Directionality ────────────────────────────────────────────────────────────
# +1 = higher score means better binder (e.g. iptm, plddt, confidence)
# -1 = lower score means better binder (e.g. interface_dG, pae, pde)
# Auto-populated by keyword; add manual overrides below if needed.

def infer_direction(col):
    col_l = col.lower()
    if any(k in col_l for k in ["dg", "pae", "pde", "ipde", "pyros"]):
        return -1
    return +1  # iptm, ptm, plddt, ipsae, confidence, etc.

DIRECTION_OVERRIDES = {
    # "some_col": -1,  # add manual overrides here if needed
}

if "binding" not in df.columns:
    print("No 'binding' column found — skipping Step 7.")
else:
    y_all = pd.to_numeric(df["binding"], errors="coerce")
    if y_all.nunique() < 2:
        print("Binding column has <2 unique values — skipping evaluation.")
    else:
        for iface in ["A,C", "B,C", "all"]:
            if iface == "all":
                sub   = df.copy()
                y_sub = y_all.copy()
                tag   = "all_interfaces"
            else:
                mask_iface = df["interface"] == iface
                sub        = df[mask_iface].copy()
                y_sub      = y_all[mask_iface].copy()
                tag        = iface.replace(",", "")

            if y_sub.nunique() < 2:
                print(f"[{iface}] Binding column has <2 classes — skipping.")
                continue

            eval_rows = []
            for col in score_cols:
                direction = DIRECTION_OVERRIDES.get(col, infer_direction(col))
                scores    = pd.to_numeric(sub[col], errors="coerce") * direction
                mask      = scores.notna() & y_sub.notna()
                if mask.sum() < 10:
                    continue
                y = y_sub[mask].astype(int).values
                s = scores[mask].values
                if len(np.unique(y)) < 2:
                    continue
                try:
                    roc = roc_auc_score(y, s)
                    pr  = average_precision_score(y, s)
                except Exception:
                    continue
                precision, recall, thresholds = precision_recall_curve(y, s)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx  = np.argmax(f1_scores)
                best_f1   = f1_scores[best_idx]
                best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else np.nan
                k         = min(10, len(s))
                top_idx   = np.argsort(s)[-k:][::-1]
                eval_rows.append({
                    "metric":           col,
                    "direction":        direction,
                    "roc_auc":          round(roc, 4),
                    "pr_auc":           round(pr,  4),
                    "best_f1":          round(best_f1, 4),
                    "best_threshold":   round(best_thr * direction, 4) if not np.isnan(best_thr) else np.nan,
                    "top10_precision":  round(y[top_idx].mean(), 4),
                    "n":                int(mask.sum()),
                })

            if not eval_rows:
                print(f"[{iface}] No valid metrics for evaluation.")
                continue

            eval_df = (pd.DataFrame(eval_rows)
                         .sort_values("pr_auc", ascending=False))
            fname_csv = OUTPUT_DIR / f"06_metric_evaluation_{tag}.csv"
            eval_df.to_csv(fname_csv, index=False)

            print(f"\n[{iface}] Top metrics by PR-AUC:")
            print(eval_df[["metric", "pr_auc", "roc_auc",
                            "top10_precision", "best_f1"]]
                  .head(15).to_string(index=False))

            # ── Plot: PR-AUC + ROC-AUC side by side ──────────────────────────
            top = eval_df.head(20)[::-1]
            fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(top) * 0.4)),
                                     sharey=True)

            for ax, metric_col, xlabel, color in [
                (axes[0], "pr_auc",  "PR-AUC",  "#1976D2"),
                (axes[1], "roc_auc", "ROC-AUC", "#388E3C"),
            ]:
                bars = ax.barh(range(len(top)), top[metric_col],
                               color=color, alpha=0.8)
                ax.axvline(0.5, color="gray", linewidth=0.8,
                           linestyle="--", label="random (0.5)")
                ax.set_xlim(0, 1)
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_title(xlabel, fontsize=11)
                ax.legend(fontsize=8)
                # Value labels
                for bar, val in zip(bars, top[metric_col]):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center", fontsize=7)

            axes[0].set_yticks(range(len(top)))
            axes[0].set_yticklabels(top["metric"].tolist(), fontsize=8)
            fig.suptitle(f"Metric Evaluation vs Binding Labels [{iface} interface]",
                         fontsize=12)
            plt.tight_layout()
            fname_png = OUTPUT_DIR / f"06_metric_evaluation_{tag}.png"
            plt.savefig(fname_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved → {fname_png.name}")
            print(f"Saved → {fname_csv.name}")
