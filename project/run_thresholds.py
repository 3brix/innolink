"""
Threshold report (REPORT-ONLY).

For every one-sided metric, compare two candidate cutoffs on the labelled
benchmark (eval.csv):

  - DERIVED : the data-driven, F1-optimal cutoff found on the benchmark
              (via analysis.evaluation.calculate_all_metrics), in native units;
  - REFERENCE  : the metric's 'reference:' value in config/metric_data.yaml -- the
              conventional operating point for that metric.

For each it reports precision, recall and N-passing, plus the metric's PR-AUC /
ROC-AUC / effect direction and the class balance. The point of the comparison is to show
whether the conventional cutoffs transfer to this data (several do not: the esm3dg_dg_a reference
value retains 1 of 133 samples), which is what justifies the distribution-based gate values
used in config/thresholds.yaml.

Reports only -- nothing downstream reads these files.

Two-sided WINDOW families (e.g. net_charge, surface_hydrophobicity) are excluded from
this one-sided report and listed separately.

Outputs (under EVALUATION_DIR/<dataset>/thresholds/):
  threshold_report.csv            per metric, derived vs reference, pooled
  threshold_report_by_category.csv    per-category summary
  threshold_report_by_dataset.csv     per metric x dataset (pooled sets only)
"""

import logging

import numpy as np
import pandas as pd


from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from preprocessing.metric_meta import load_window_families
from preprocessing.align import load_metric_directions, get_direction
from preprocessing.metric_meta import load_categories, get_category, load_reference_values, get_reference_value
from analysis.distributions import get_all_numeric_metrics
from analysis.io import load_eval, rankings_for
from analysis.evaluation import calculate_all_metrics, load_metric_families, get_family, precision_recall_at

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)









def build_report(df: pd.DataFrame, metrics: list[str], directions: dict,
                 families: dict, categories: dict, reference_values: dict,
                 derived: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per metric: derived (F1-opt) vs conventional reference cutoff, each with prec/recall/N."""
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

        ref_cut = get_reference_value(m, reference_values)
        # only scalar reference cutoffs apply here (windows/null handled elsewhere)
        ref_cut = ref_cut if isinstance(ref_cut, (int, float)) else None
        ref = precision_recall_at(df[m], y_true, ref_cut, direction)

        pr_auc = float(d.get("pr_auc", np.nan))

        # PR-AUC relative to the positive-class baseline: threshold-INDEPENDENT enrichment
        pr_auc_vs_base = round(pr_auc / base_rate, 3) if (np.isfinite(pr_auc) and base_rate) else np.nan

        rows.append({
            "metric": m, "family": family, "category": category, "direction": direction,
            "n_pos": n_pos, "n_neg": n_neg, "base_rate": base_rate,
            "pr_auc": pr_auc, "roc_auc": float(d.get("aligned_roc", np.nan)),
            "pr_auc_vs_baseline": pr_auc_vs_base,          # threshold-independent ranking enrichment
            "above_baseline": bool(pr_auc > base_rate) if np.isfinite(pr_auc) else False,
            # F1-optimal cutoff (kept as reference)
            "f1_cutoff": derived_cut,
            "f1_precision": der["precision"], "f1_recall": der["recall"], "f1_n_pass": der["n_pass"],
            # conventional reference cutoff (for comparison)
            "ref_cutoff": ref_cut,
            "ref_precision": ref["precision"], "ref_recall": ref["recall"], "ref_n_pass": ref["n_pass"],
            "n_eval": der["n_eval"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("pr_auc", ascending=False).reset_index(drop=True)


# run
base = RAW_DATA_DIR / cfg.name
output_dir = EVALUATION_DIR / cfg.name / "thresholds"
output_dir.mkdir(parents=True, exist_ok=True)

df = load_eval(cfg)
logger.info("Threshold report on %d labelled rows", len(df))

directions = load_metric_directions(METRIC_YAML)
families = load_metric_families(METRIC_YAML)
categories = load_categories(METRIC_YAML)
reference_values = load_reference_values()

# FULL set minus the two-sided window families: this report is descriptive, and the
# developability/energy families are exactly the ones thresholds.yaml documents.
all_metrics = get_all_numeric_metrics(df)
window_cols = [m for m in all_metrics if get_family(m, families) in load_window_families()]
metrics = [m for m in all_metrics if m not in window_cols]
if window_cols:
    logger.info("Excluded %d two-sided window metric(s) from one-sided report: %s",
                len(window_cols), window_cols)

report = build_report(df, metrics, directions, families, categories, reference_values,
                      derived=rankings_for(cfg, df, metrics, directions))
report.to_csv(output_dir / "threshold_report.csv", index=False)

# per-category summary: how well each category separates binders, derived vs reference
by_cat = (report.groupby("category")
          .agg(n_metrics=("metric", "count"),
               mean_pr_auc=("pr_auc", "mean"),
               mean_roc_auc=("roc_auc", "mean"),
               n_above_baseline=("above_baseline", "sum"),
               mean_ref_precision=("ref_precision", "mean"))
          .round(4).reset_index()
          .sort_values("mean_pr_auc", ascending=False))
by_cat.to_csv(output_dir / "threshold_report_by_category.csv", index=False)



# per-dataset breakdown for pooled sets
if cfg.is_set and "dataset" in df.columns:
    parts = []
    for ds, g in df.groupby("dataset"):
        if g["binder"].nunique() < 2:
            logger.warning("skipping %s: single binder class", ds)
            continue
        r = build_report(g, metrics, directions, families, categories, reference_values)
        r.insert(0, "dataset", ds)
        parts.append(r)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            output_dir / "threshold_report_by_dataset.csv", index=False)

print(f"[{cfg.name}] threshold report -> {output_dir}")
print(f"  metrics reported: {len(report)}  (window metrics excluded: {len(window_cols)})")
print("  per-category separation (mean PR-AUC, derived vs reference precision):")
print(by_cat.to_string(index=False))