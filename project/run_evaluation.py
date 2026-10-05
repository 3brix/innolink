import logging
import os

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from preprocessing.align import load_metric_directions, align_dataframe
from preprocessing.metric_meta import load_window_families, load_categories, get_category
from analysis.distributions import get_numeric_metrics, get_all_numeric_metrics
from analysis.io import load_eval
from analysis.evaluation import (
    calculate_all_metrics,
    compute_auroc_pvalue,
    cliffs_delta,
    cohens_d,
    spearman_correlation,
    load_metric_families,
    get_family,
)
from config.analysis import PRECISION_AT_PERCENT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



base = RAW_DATA_DIR / cfg.name

output_dir = EVALUATION_DIR / cfg.name
output_dir.mkdir(parents=True, exist_ok=True)

df = load_eval(cfg)
logger.info("Evaluation (BENCHMARK) on %d labelled rows", len(df))

# Benchmarking compares binders vs non-binders, so it needs BOTH classes present.
_classes = pd.to_numeric(df["binder"], errors="coerce").dropna().unique()
if len(_classes) < 2:
    logger.warning(
        "Only one binder class present (%s) in %d labelled rows -- cannot benchmark "
        "binder vs non-binder. Skipping evaluation. Check that predictions and the "
        "mastertable share sample IDs across both classes.", sorted(_classes), len(df))
    raise SystemExit(0)

# FULL metric set: this benchmark is the evidence for which metrics to carry forward, so
# evaluating only the already-kept predictors would be
# circular. 'in_model_set' below marks the ones the composite / RF actually use.
metrics = get_all_numeric_metrics(df)
model_set = set(get_numeric_metrics(df))
directions = load_metric_directions(METRIC_YAML)
categories = load_categories(METRIC_YAML)
logger.info("Benchmarking %d metrics (%d of them in the predictor set)", len(metrics), len(model_set))

# window families are dropped: "higher is better" is meaningless for a two-sided metric
# screened as windows in feasibility instead.
_families = load_metric_families(METRIC_YAML)
_window_cols = [m for m in metrics if get_family(m, _families) in load_window_families()]
if _window_cols:
    metrics = [m for m in metrics if m not in _window_cols]
    logger.info("Excluded %d window-metric column(s) from monotonic evaluation: %s",
                len(_window_cols), _window_cols)

# pooled / single-dataset ranking + effect sizes + redundancy
_pct = float(os.getenv("PRECISION_AT_PERCENT", PRECISION_AT_PERCENT))
logger.info("Benchmark: precision@%.0f%% + PR-AUC point estimates (pooled + per-dataset + per-mol_type)", _pct*100)
rankings = calculate_all_metrics(df, metrics, directions, pct=_pct)
compute_auroc_pvalue(df, metrics, directions).to_csv(output_dir / "auroc_pvalue.csv", index=False)
cliffs_delta(df, metrics, directions).to_csv(output_dir / "cliffs_delta.csv", index=False)
cohens_d(df, metrics, directions).to_csv(output_dir / "cohens_d.csv", index=False)


aligned_path = base / "eval_aligned.csv"
aligned = pd.read_csv(aligned_path) if aligned_path.exists() else align_dataframe(df, METRIC_YAML)
spearman_correlation(aligned).to_csv(output_dir / "spearman_correlation.csv")   # model-set metrics only (window families included)

# per-dataset breakdown (pooled sets)
if cfg.is_set and "dataset" in df.columns:
    rankings_by, cliffs_by, cohens_by = [], [], []
    for ds, group in df.groupby("dataset"):
        if group["binder"].nunique() < 2:
            logger.warning("skipping %s: single binder class (%d rows)", ds, len(group))
            continue
        r = calculate_all_metrics(group, metrics, directions, pct=_pct); r.insert(0, "dataset", ds); rankings_by.append(r)
        c = cliffs_delta(group, metrics, directions);          c.insert(0, "dataset", ds); cliffs_by.append(c)
        d = cohens_d(group, metrics, directions);              d.insert(0, "dataset", ds); cohens_by.append(d)
    by_dataset = pd.concat(rankings_by, ignore_index=True)
    by_dataset.to_csv(output_dir / "rankings_by_dataset.csv", index=False)
    pd.concat(cliffs_by, ignore_index=True).to_csv(output_dir / "cliffs_delta_by_dataset.csv", index=False)
    pd.concat(cohens_by, ignore_index=True).to_csv(output_dir / "cohens_d_by_dataset.csv", index=False)
    print(f"Set '{cfg.name}': wrote pooled + per-dataset breakdown ({df['dataset'].nunique()} datasets)")

    # A metric can look like no-skill POOLED while separating inside every dataset (Simpson's
    # paradox), so cross-source consistency belongs next to the pooled number.
    #
    # PR-AUC's baseline is each dataset's prevalence (0.09-0.74 here),
    # so raw per-dataset PR-AUCs are not averageable. Three views are recorded:
    #   pr_auc_ds_*   raw per-dataset PR-AUC (read together with that dataset's prevalence)
    #   ap_norm_ds_*  (PR-AUC - prevalence) / (1 - prevalence): 0 = no-skill, 1 = perfect, so it
    #                 IS comparable across datasets -- the prevalence-corrected primary measure
    #   all_datasets_above_baseline  PR-AUC > prevalence in EVERY dataset
    # ROC-AUC is kept as the secondary, prevalence-independent cross-check.
    per_ds = by_dataset.copy()
    per_ds["ap_norm"] = (per_ds["pr_auc"] - per_ds["prevalence"]) / (1 - per_ds["prevalence"])
    per_ds["above_baseline"] = per_ds["pr_auc"] > per_ds["prevalence"]
    agg = (per_ds.groupby("metric")
           .agg(n_datasets=("pr_auc", "count"),
                pr_auc_ds_mean=("pr_auc", "mean"), pr_auc_ds_min=("pr_auc", "min"),
                ap_norm_ds_mean=("ap_norm", "mean"), ap_norm_ds_min=("ap_norm", "min"),
                roc_ds_mean=("aligned_roc", "mean"), roc_ds_min=("aligned_roc", "min"),
                n_datasets_above_baseline=("above_baseline", "sum"))
           .round(4).reset_index())
    agg["all_datasets_above_baseline"] = agg["n_datasets_above_baseline"] == agg["n_datasets"]
    agg["all_datasets_above_chance"] = agg["roc_ds_min"] > 0.5
    rankings = rankings.merge(agg, on="metric", how="left")

# rankings.csv is written LAST so it can carry the per-dataset consistency columns.
rankings.insert(1, "category", [get_category(m, categories) for m in rankings["metric"]])
rankings.insert(2, "in_model_set", rankings["metric"].isin(model_set))
rankings.to_csv(output_dir / "rankings.csv", index=False)
print(f"rankings.csv: {len(rankings)} metrics ({int(rankings['in_model_set'].sum())} in the predictor set)")

# per-molecule-type breakdown (nanobody vs antibody), when the set mixes types
if "mol_type" in df.columns and df["mol_type"].nunique() > 1:
    parts = []
    for mt, group in df.groupby("mol_type"):
        if group["binder"].nunique() < 2:
            logger.warning("skipping mol_type %s: single binder class (%d rows)", mt, len(group))
            continue
        r = calculate_all_metrics(group, metrics, directions, pct=_pct)
        r.insert(0, "mol_type", mt)
        parts.append(r)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(output_dir / "rankings_by_mol_type.csv", index=False)
        print(f"Set '{cfg.name}': wrote per-mol_type breakdown ({df['mol_type'].nunique()} types)")