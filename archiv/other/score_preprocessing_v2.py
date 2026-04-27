"""
Scoring Pipeline Preprocessing & EDA
=====================================
Processes raw scores from AF3, ColabFold, ESMFold, Chai, Boltz predictors.

Key design decisions:
  - `binding` column is ground truth for binder/mutant classification
  - Base name extracted by stripping both single-point (_A_K110A) and
    all-interface (_all_A) mutation suffixes
  - Direction (higher=better vs lower=better) is inferred automatically
    from ROC-AUC: if raw ROC-AUC < 0.5, the metric is flipped
  - ΔΔScore computed per interface via vectorized pandas merge
  - No external config files or yaml dependency
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV  = "scores_raw.csv"       # <-- change to your file path
OUTPUT_DIR = Path("pipeline_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

KEEP_INTERFACES = {"A,C", "B,C"}

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

# ── Step 2: Labels & Base Extraction ─────────────────────────────────────────
#
# Ground truth comes from the `binding` column (True/False), not from
# parsing the sample name. This correctly handles all mutant types:
#   - Single-point:  5ig7_abc_cut_A_H52A   -> ala_scan
#   - All-interface: 5ifj_abc_cut_all_A    -> ala_scan
#   - Wild-type:     5ig7_abc_cut          -> binder
#
# Base name is extracted by stripping known mutation suffixes so that
# each mutant can be paired with its parent WT for ΔΔScore calculation.
#
# Suffix patterns stripped (in order, first match wins):
#   _all_A          e.g. 5ifj_abc_cut_all_A      -> 5ifj_abc_cut
#   _CHAIN_MutCode  e.g. 5ig7_abc_cut_A_H52A     -> 5ig7_abc_cut

print("\n" + "=" * 60)
print("STEP 2 — Labels & Base Extraction")
print("=" * 60)

# Regex patterns for suffix stripping (order matters — most specific first)
_SUFFIX_PATTERNS = [
    re.compile(r"_all_[A-Z]+$"),                    # _all_A
    re.compile(r"_[A-Z]_[A-Z]\d+[A-Z]$"),           # _A_H52A
]

def extract_base(sample):
    for pat in _SUFFIX_PATTERNS:
        m = pat.search(sample)
        if m:
            return sample[:m.start()]
    return sample  # no suffix found -> is WT

def extract_mutation(sample):
    for pat in _SUFFIX_PATTERNS:
        m = pat.search(sample)
        if m:
            return m.group().lstrip("_")
    return None

def extract_mut_chain(sample):
    # only defined for single-point mutations
    m = re.search(r"_([A-Z])_[A-Z]\d+[A-Z]$", sample)
    return m.group(1) if m else None

# Use binding column as ground truth
df["binding"] = df["binding"].map(
    {True: True, False: False, "True": True, "False": False,
     1: True, 0: False, "1": True, "0": False}
)
df["group"]     = df["binding"].map({True: "binder", False: "ala_scan"})
df["base"]      = df["sample"].apply(extract_base)
df["mutation"]  = df["sample"].apply(extract_mutation)
df["mut_chain"] = df["sample"].apply(extract_mut_chain)

print(df["group"].value_counts().to_string())
print(f"\nUnique base structures: {df['base'].nunique()}")
print(f"Unique mutations:       {df['mutation'].dropna().nunique()}")
print(f"\nSample parsing (first 8 rows):")
print(df[["sample", "interface", "group", "base",
          "mut_chain", "mutation"]].head(8).to_string(index=False))

# ── Step 3: Raw vs. Non-Raw Agreement ────────────────────────────────────────
# (report only — no columns dropped)

print("\n" + "=" * 60)
print("STEP 3 — Raw vs. Non-Raw Agreement (report only)")
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
pairs = []
for rc in raw_cols:
    proc = find_processed_counterpart(rc, all_cols)
    if proc:
        pairs.append((proc, rc))

print(f"Matched raw/non-raw pairs: {len(pairs)}")

pair_df_rows = []
for col, raw_col in pairs:
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
    pair_df_rows.append({"col": col, "raw_col": raw_col,
                          "corr": round(corr, 6) if not np.isnan(corr) else np.nan,
                          "max_diff": round(max_diff, 8) if not np.isnan(max_diff) else np.nan,
                          "status": status})

if pair_df_rows:
    pair_df = pd.DataFrame(pair_df_rows)
    print(pair_df[["col", "corr", "max_diff", "status"]].to_string(index=False))
    pair_df.to_csv(OUTPUT_DIR / "raw_vs_nonraw_comparison.csv", index=False)

    different = pair_df[pair_df["status"] == "DIFFERENT"]
    if len(different) > 0:
        ncols = min(3, len(different))
        nrows = int(np.ceil(len(different) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 4), squeeze=False)
        axes = axes.flatten()
        for i, (_, row) in enumerate(different.iterrows()):
            ax = axes[i]
            a  = pd.to_numeric(df[row["col"]],     errors="coerce")
            b  = pd.to_numeric(df[row["raw_col"]], errors="coerce")
            for grp, color in palette.items():
                mask = df["group"] == grp
                ax.scatter(a[mask], b[mask], c=color, label=grp,
                           alpha=0.6, s=20, edgecolors="none")
            lims = [min(a.min(), b.min()), max(a.max(), b.max())]
            ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel(f"processed: {row['col']}", fontsize=8)
            ax.set_ylabel(f"raw: {row['raw_col']}", fontsize=8)
            ax.set_title(f"r = {row['corr']:.4f}", fontsize=9)
            ax.legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Processed vs Raw — DIFFERENT pairs", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_raw_vs_processed_scatter.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → 03_raw_vs_processed_scatter.png")
else:
    print("No matched raw/non-raw pairs found.")

# ── Step 4: Select Numeric Score Columns ──────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Select Score Columns")
print("=" * 60)

EXCLUDE = {
    "sample", "binding", "interface", "base", "group",
    "mut_chain", "mutation",
    "af3_rank", "af3_raw_rank", "af3_rank0_masif", "af3_raw_rank0_masif",
    "af3_notpl_raw_rank", "af3_notpl_raw_rank0_masif",
    "cf_rank", "cf_raw_rank", "cf_rank0_masif", "cf_raw_rank0_masif",
    "cf_custom_raw_rank", "cf_custom_raw_rank0_masif",
    "cf_notpl_raw_rank", "cf_notpl_raw_rank0_masif",
    "af3_ranking_score", "af3_raw_ranking_score", "af3_notpl_raw_ranking_score",
    "chai_unconstrained_per_chain_ptm", "chai_unconstrained_per_chain_pair_iptm",
    "chai_constrained_per_chain_ptm",   "chai_constrained_per_chain_pair_iptm",
}
path_cols = {c for c in df.columns if "path" in c.lower()}
EXCLUDE.update(path_cols)

score_cols = [
    c for c in df.columns
    if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
]
print(f"Score columns retained: {len(score_cols)}")

# ── Step 4.5: Auto-Direction via ROC-AUC ──────────────────────────────────────
#
# Instead of a hardcoded config, direction is inferred from the data:
#   - Compute raw ROC-AUC for each metric against binding labels
#   - If ROC-AUC >= 0.5: higher = better (keep as-is, direction = +1)
#   - If ROC-AUC <  0.5: lower  = better (flip sign,  direction = -1)
#
# This requires binding labels to be present. If a metric has fewer than
# MIN_SAMPLES labeled rows it is skipped and kept as-is.
#
# Caveat: direction is computed from your labeled data, not domain knowledge.
# With clean, balanced labels this is robust. With noisy or heavily imbalanced
# labels a low ROC-AUC might incorrectly flip a well-oriented metric —
# always sanity-check the direction table in 04_auto_directions.csv.
#
# NOTE — normalization for ML (future):
#   Scale does not matter for ROC-AUC but will matter for ML models.
#   At that point use QuantileTransformer fit on training data only:
#       qt.fit(X_train[score_cols])
#       X_train_norm = qt.transform(X_train[score_cols])
#       X_test_norm  = qt.transform(X_test[score_cols])  # NOT fit_transform!

print("\n" + "=" * 60)
print("STEP 4.5 — Auto-Direction via ROC-AUC")
print("=" * 60)

MIN_SAMPLES = 10

y_all = pd.to_numeric(df["binding"].map({True: 1, False: 0}), errors="coerce")

direction_rows = []
DIRECTIONS = {}

for col in score_cols:
    scores = pd.to_numeric(df[col], errors="coerce")
    mask   = scores.notna() & y_all.notna()
    if mask.sum() < MIN_SAMPLES or y_all[mask].nunique() < 2:
        DIRECTIONS[col] = 1   # not enough data — keep as-is
        direction_rows.append({"col": col, "raw_roc": np.nan,
                                "direction": 1, "note": "insufficient_data"})
        continue
    raw_roc = roc_auc_score(y_all[mask], scores[mask])
    direction = 1 if raw_roc >= 0.5 else -1
    DIRECTIONS[col] = direction
    direction_rows.append({"col": col, "raw_roc": round(raw_roc, 4),
                            "direction": direction,
                            "note": "flipped" if direction == -1 else "kept"})

dir_df = pd.DataFrame(direction_rows)
dir_df.to_csv(OUTPUT_DIR / "04_auto_directions.csv", index=False)
print(dir_df[["col", "raw_roc", "direction", "note"]].to_string(index=False))
print(f"\nFlipped: {(dir_df['direction'] == -1).sum()} / {len(dir_df)} metrics")
print("Saved → 04_auto_directions.csv")

# Apply directions
df_aligned = df.copy()
for col in score_cols:
    if DIRECTIONS[col] == -1:
        df_aligned[col] = df_aligned[col] * -1

df_aligned.to_csv(OUTPUT_DIR / "scores_clean.csv", index=False)
print("Saved direction-aligned scores → scores_clean.csv")

# ── Step 5: ΔΔScore per Interface (vectorized) ────────────────────────────────
#
# For each mutant row, subtract its parent WT scores on the same interface.
# Pairing key: (base, interface) — works for all mutant types:
#   single-point (_A_H52A)  -> base stripped, both interfaces compared
#   all-interface (_all_A)  -> base stripped, both interfaces compared
#
# expected_affected flags whether the mutated chain contacts the target (C)
# on this interface — useful sanity check for single-point mutants.
# For _all_A mutants both interfaces are expected to be affected.
#
# ΔΔScore is computed on direction-ALIGNED scores (df_aligned) so that
# a negative delta always means "mutation hurt predicted binding."

print("\n" + "=" * 60)
print("STEP 5 — ΔΔScore per Interface (vectorized)")
print("=" * 60)

meta_cols  = ["sample", "interface", "base", "group", "mutation",
              "mut_chain", "binding"]

wt_scores  = (df_aligned[df_aligned["group"] == "binder"]
              .set_index(["base", "interface"])[score_cols])

mut_scores = (df_aligned[df_aligned["group"] == "ala_scan"]
              .set_index(["base", "interface"])[score_cols])

# Vectorized subtraction — only rows with matching (base, interface) WT
common_idx = mut_scores.index.intersection(wt_scores.index)
delta      = mut_scores.loc[common_idx] - wt_scores.loc[common_idx]

# Re-attach metadata
meta = (df_aligned[df_aligned["group"] == "ala_scan"]
        .set_index(["base", "interface"])[
            ["sample", "mutation", "mut_chain", "binding"]]
        .loc[common_idx])

df_delta = pd.concat([meta, delta], axis=1).reset_index()

# Sanity flag: for single-point mutants, is the mutated chain in this interface?
def is_expected_affected(row):
    if pd.isna(row["mut_chain"]):
        return True   # _all_A — all interfaces expected to be affected
    return row["mut_chain"] in str(row["interface"])

df_delta["expected_affected"] = df_delta.apply(is_expected_affected, axis=1)

df_delta.to_csv(OUTPUT_DIR / "scores_delta.csv", index=False)
print(f"ΔΔScore rows:                    {len(df_delta)}")
print(f"  Expected-affected interfaces:  {df_delta['expected_affected'].sum()}")
print(f"  Control interfaces:            {(~df_delta['expected_affected']).sum()}")
print("Saved → scores_delta.csv")

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

# 6a — Distributions per interface
for iface in ["A,C", "B,C"]:
    sub   = df_aligned[df_aligned["interface"] == iface]
    n     = len(KEY_METRICS)
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
    fig.suptitle(f"Distributions: Binder vs Alanine Scan  [{iface}]",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fname = OUTPUT_DIR / f"01_distributions_{iface_tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fname.name}")

# 6b — Correlation heatmap (binders only, direction-aligned)
binder_data = df_aligned.loc[df_aligned["group"] == "binder",
                              KEY_METRICS].dropna(axis=1, how="all")
if binder_data.shape[0] > 1:
    corr_mat = binder_data.corr()
    fig, ax  = plt.subplots(figsize=(max(8, len(corr_mat) * 0.55),
                                      max(6, len(corr_mat) * 0.55)))
    sns.heatmap(corr_mat, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=len(corr_mat) <= 15, fmt=".2f", annot_kws={"size": 7},
                xticklabels=corr_mat.columns, yticklabels=corr_mat.index)
    ax.set_title("Metric Correlation (Binders, direction-aligned)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → 02_correlation_heatmap.png")

# 6c — ΔΔScore heatmaps: affected vs control interface
delta_score_cols = [c for c in KEY_METRICS if c in df_delta.columns]
if delta_score_cols and len(df_delta) > 0:
    for affected, label in [(True, "affected"), (False, "control")]:
        sub = df_delta[df_delta["expected_affected"] == affected]
        if len(sub) == 0:
            continue
        pivot = sub.set_index("mutation")[delta_score_cols]
        fig, ax = plt.subplots(figsize=(max(8, len(delta_score_cols) * 0.6),
                                         max(4, len(pivot) * 0.35)))
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                    xticklabels=pivot.columns, yticklabels=pivot.index)
        ax.set_title(
            f"ΔΔScore — {'mutated' if affected else 'control (unaffected)'} "
            f"interface  (mutant − WT, direction-aligned)", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        fname = OUTPUT_DIR / f"05_delta_heatmap_{label}_interface.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {fname.name}")

# ── Step 7: Metric Benchmarking (ROC-AUC, PR-AUC, F1) ────────────────────────
#
# Evaluates each metric's ability to separate binders from alanine-scan mutants
# using the direction-aligned scores (so higher always = better binder).
# Run per interface and combined.

print("\n" + "=" * 60)
print("STEP 7 — Metric Benchmarking")
print("=" * 60)

y_all_aligned = pd.to_numeric(
    df_aligned["binding"].map({True: 1, False: 0}), errors="coerce")

for iface in ["A,C", "B,C", "all"]:
    if iface == "all":
        sub   = df_aligned.copy()
        y_sub = y_all_aligned.copy()
        tag   = "all_interfaces"
    else:
        mask  = df_aligned["interface"] == iface
        sub   = df_aligned[mask].copy()
        y_sub = y_all_aligned[mask].copy()
        tag   = iface.replace(",", "")

    if y_sub.nunique() < 2:
        print(f"[{iface}] Binding column has <2 classes — skipping.")
        continue

    eval_rows = []
    for col in score_cols:
        s    = pd.to_numeric(sub[col], errors="coerce")
        mask = s.notna() & y_sub.notna()
        if mask.sum() < MIN_SAMPLES or y_sub[mask].nunique() < 2:
            continue
        y = y_sub[mask].astype(int).values
        s = s[mask].values
        try:
            roc = roc_auc_score(y, s)
            pr  = average_precision_score(y, s)
        except Exception:
            continue
        precision, recall, thresholds = precision_recall_curve(y, s)
        f1       = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1)
        eval_rows.append({
            "metric":          col,
            "roc_auc":         round(roc, 4),
            "pr_auc":          round(pr,  4),
            "best_f1":         round(f1[best_idx], 4),
            "best_threshold":  round(thresholds[best_idx], 4)
                           if best_idx < len(thresholds) else np.nan,
            "top10_precision": round(y[np.argsort(s)[-10:][::-1]].mean(), 4),
            "n":               int(mask.sum()),
        })

    if not eval_rows:
        print(f"[{iface}] No valid metrics.")
        continue

    eval_df = pd.DataFrame(eval_rows).sort_values("roc_auc", ascending=False)
    fname_csv = OUTPUT_DIR / f"06_benchmarking_{tag}.csv"
    eval_df.to_csv(fname_csv, index=False)

    print(f"\n[{iface}] Top 15 by ROC-AUC:")
    print(eval_df[["metric", "roc_auc", "pr_auc", "best_f1", "top10_precision"]]
          .head(15).to_string(index=False))

    top  = eval_df.head(20)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(top) * 0.4)),
                             sharey=True)
    for ax, metric_col, xlabel, color in [
        (axes[0], "roc_auc", "ROC-AUC",  "#1976D2"),
        (axes[1], "pr_auc",  "PR-AUC",   "#388E3C"),
    ]:
        bars = ax.barh(range(len(top)), top[metric_col],
                       color=color, alpha=0.8)
        ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--",
                   label="random (0.5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(xlabel, fontsize=11)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, top[metric_col]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top["metric"].tolist(), fontsize=8)
    fig.suptitle(f"Metric Benchmarking [{iface}]", fontsize=12)
    plt.tight_layout()
    fname_png = OUTPUT_DIR / f"06_benchmarking_{tag}.png"
    plt.savefig(fname_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fname_png.name}  |  {fname_csv.name}")

print("\n" + "=" * 60)
print("DONE. All outputs in:", OUTPUT_DIR)
print("=" * 60)
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
