"""
Threshold report (REPORT-ONLY).

For every one-sided metric, compare two candidate cutoffs on the labelled
benchmark (eval.csv):

  - DERIVED : the data-driven, F1-optimal cutoff found on the benchmark
              (via analysis.evaluation.calculate_all_metrics), in native units;
  - LITERATURE : the per-family value in config/thresholds.yaml.

For each it reports precision, recall and N-passing, plus the metric's PR-AUC /
ROC-AUC / effect direction and the class balance. This lets you SEE, side by
side, how a benchmark-derived cutoff would behave versus the literature value,
per metric and per category (confidence / interface / sequence vs
developability / energy).

IMPORTANT: this stage writes reports only. It does NOT change thresholds.yaml
and does NOT change what run_filter.py uses. As agreed, developability/energy
gates stay literature-based unless you decide, after reading this report, to
adopt a derived cutoff. Developability/energy metrics are expected to separate
binders near-randomly (PR-AUC ~ base rate); that non-separation is itself a
reported result.

Two-sided WINDOW families (net_charge, surface_hydrophobicity) are excluded from
this one-sided report and listed separately.

Outputs (under EVALUATION_DIR/<dataset>/thresholds/):
  threshold_report.csv            per metric, derived vs literature, pooled
  threshold_report_by_category.csv    per-category summary
  threshold_report_by_dataset.csv     per metric x dataset (pooled sets only)
"""

import logging
import os

import numpy as np
import pandas as pd


from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from config.analysis import WINDOW_FAMILIES, PRECISION_TARGET as _PT_DEFAULT, N_MIN as _NMIN_DEFAULT
from preprocessing.align import load_metric_directions, get_direction
from preprocessing.metric_meta import load_categories, get_category, load_thresholds
from analysis.distributions import get_numeric_metrics
from analysis.io import load_eval, rankings_for
from analysis.evaluation import select_threshold
from analysis.evaluation import calculate_all_metrics, load_metric_families, get_family, precision_recall_at

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Precision-first threshold selection (Q1). CONFIGURABLE and intentionally NOT final:
# choose the operating values after inspecting threshold_sweep.csv on the full data.
PRECISION_TARGET = float(os.getenv("PRECISION_TARGET", _PT_DEFAULT))   # default in config.analysis
N_MIN = int(os.getenv("N_MIN", _NMIN_DEFAULT))                          # default in config.analysis
SWEEP_TARGETS = [0.6, 0.7, 0.8, 0.9, 0.95]                       # for the parameter-choice sweep








def build_report(df: pd.DataFrame, metrics: list[str], directions: dict,
                 families: dict, categories: dict, lit_thresholds: dict,
                 derived: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per metric: derived (F1-opt) vs literature cutoff, each with prec/recall/N.
    `derived` = a precomputed rankings table (canonical, pooled); if None it is computed
    here (used for the per-dataset breakdown, which is genuinely per-group)."""
    if derived is None:
        derived = calculate_all_metrics(df, metrics, directions)
    derived = derived.set_index("metric")

    y_true = df["binder"]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    base_rate = round(n_pos / (n_pos + n_neg), 4) if (n_pos + n_neg) else np.nan

    rows = []
    for m in metrics:
        if m not in derived.index:
            continue
        d = derived.loc[m]
        direction = get_direction(m, directions)
        family = get_family(m, families)
        category = get_category(m, categories)

        derived_cut = d.get("opt_threshold_raw", np.nan)
        der = precision_recall_at(df[m], y_true, derived_cut, direction)

        lit_cut = lit_thresholds.get(family) if family else None
        # only scalar literature cutoffs apply here (windows/null handled elsewhere)
        lit_cut = lit_cut if isinstance(lit_cut, (int, float)) else None
        lit = precision_recall_at(df[m], y_true, lit_cut, direction)

        pr_auc = float(d.get("pr_auc", np.nan))

        # target-precision + N-floor operating point (precision-first filtering, Q1)
        aligned = df[m] * direction
        sel = select_threshold(y_true, aligned, PRECISION_TARGET, N_MIN)
        tp_cut_native = round(sel["threshold"] * direction, 4) if np.isfinite(sel["threshold"]) else np.nan
        # enrichment: PRECISION enrichment (threshold-dependent) vs PR-AUC-vs-baseline
        # (threshold-INDEPENDENT). Named distinctly on purpose -- these are NOT the same thing.
        prec_enr = round(sel["precision"] / base_rate, 3) if (sel["precision"] == sel["precision"] and base_rate) else np.nan
        pr_auc_vs_base = round(pr_auc / base_rate, 3) if (np.isfinite(pr_auc) and base_rate) else np.nan

        rows.append({
            "metric": m, "family": family, "category": category, "direction": direction,
            "n_pos": n_pos, "n_neg": n_neg, "base_rate": base_rate,
            "pr_auc": pr_auc, "roc_auc": float(d.get("aligned_roc", np.nan)),
            "pr_auc_vs_baseline": pr_auc_vs_base,          # threshold-independent ranking enrichment
            "above_baseline": bool(pr_auc > base_rate) if np.isfinite(pr_auc) else False,
            # F1-optimal cutoff (kept as REFERENCE)
            "f1_cutoff": derived_cut,
            "f1_precision": der["precision"], "f1_recall": der["recall"], "f1_n_pass": der["n_pass"],
            # target-precision + N-floor cutoff (precision-first; the filtering criterion)
            "tp_cutoff": tp_cut_native,
            "tp_precision": sel["precision"], "tp_recall": sel["recall"],
            "tp_n_pass": sel["n_pass"], "tp_n_pos_retained": sel["n_pos_retained"],
            "tp_precision_enrichment": prec_enr,           # threshold-DEPENDENT precision enrichment
            "tp_status": sel["status"],
            # literature cutoff (for comparison)
            "lit_cutoff": lit_cut,
            "lit_precision": lit["precision"], "lit_recall": lit["recall"], "lit_n_pass": lit["n_pass"],
            "n_eval": der["n_eval"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("pr_auc", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ run
base = RAW_DATA_DIR / cfg.name
output_dir = EVALUATION_DIR / cfg.name / "thresholds"
output_dir.mkdir(parents=True, exist_ok=True)

df = load_eval(cfg)
logger.info("Threshold report on %d labelled rows", len(df))

directions = load_metric_directions(METRIC_YAML)
families = load_metric_families(METRIC_YAML)
categories = load_categories(METRIC_YAML)
lit_thresholds = load_thresholds()

# candidate metrics: numeric, minus two-sided window families (one-sided report)
all_metrics = get_numeric_metrics(df)
window_cols = [m for m in all_metrics if get_family(m, families) in WINDOW_FAMILIES]
metrics = [m for m in all_metrics if m not in window_cols]
if window_cols:
    logger.info("Excluded %d two-sided window metric(s) from one-sided report: %s",
                len(window_cols), window_cols)

report = build_report(df, metrics, directions, families, categories, lit_thresholds,
                      derived=rankings_for(cfg, df, metrics, directions))
report.to_csv(output_dir / "threshold_report.csv", index=False)

# per-category summary: how well each category separates binders, derived vs literature
by_cat = (report.groupby("category")
          .agg(n_metrics=("metric", "count"),
               mean_pr_auc=("pr_auc", "mean"),
               mean_roc_auc=("roc_auc", "mean"),
               n_above_baseline=("above_baseline", "sum"),
               mean_tp_precision=("tp_precision", "mean"),
               mean_lit_precision=("lit_precision", "mean"))
          .round(4).reset_index()
          .sort_values("mean_pr_auc", ascending=False))
by_cat.to_csv(output_dir / "threshold_report_by_category.csv", index=False)

# ---------------------------------------------------------------------
# Precision-target SWEEP (for choosing PRECISION_TARGET / N_MIN).
# For each confidence/interface metric and each candidate precision target, show the
# best-retaining cutoff (n_min=1) so you can see how many samples survive at each
# precision. This is the table to inspect before fixing the operating parameters.
# ---------------------------------------------------------------------
_directions = directions
sweep_rows = []
for m in metrics:
    if get_category(m, categories) not in ("confidence", "interface"):
        continue
    direction = get_direction(m, _directions)
    aligned = df[m] * direction
    for pt in SWEEP_TARGETS:
        s = select_threshold(df["binder"], aligned, precision_target=pt, n_min=1)
        sweep_rows.append({
            "metric": m, "family": get_family(m, families), "precision_target": pt,
            "cutoff": round(s["threshold"] * direction, 4) if s["threshold"] == s["threshold"] else float("nan"),
            "achieved_precision": s["precision"], "recall": s["recall"],
            "n_pass": s["n_pass"], "n_pos_retained": s["n_pos_retained"], "status": s["status"],
        })
if sweep_rows:
    pd.DataFrame(sweep_rows).to_csv(output_dir / "threshold_sweep.csv", index=False)
    logger.info("wrote threshold_sweep.csv (%d confidence/interface metrics x %d targets)",
                len(sweep_rows) // len(SWEEP_TARGETS), len(SWEEP_TARGETS))

# per-dataset breakdown for pooled sets (benchmark-fit numbers, reported per source)
if cfg.is_set and "dataset" in df.columns:
    parts = []
    for ds, g in df.groupby("dataset"):
        if g["binder"].nunique() < 2:
            logger.warning("skipping %s: single binder class", ds)
            continue
        r = build_report(g, metrics, directions, families, categories, lit_thresholds)
        r.insert(0, "dataset", ds)
        parts.append(r)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            output_dir / "threshold_report_by_dataset.csv", index=False)

print(f"[{cfg.name}] threshold report -> {output_dir}")
print(f"  metrics reported: {len(report)}  (window metrics excluded: {len(window_cols)})")
print("  per-category separation (mean PR-AUC, derived vs literature precision):")
print(by_cat.to_string(index=False))
