"""Random Forest design ranking -- the PRIMARY ranking method.

Model, hyperparameters, features, folds and seed are unchanged from the 'PIPELINE STAGE' cell of
notebooks/ranking_rf_final.ipynb, which is kept for provenance only.

Two models, kept distinct: CV models refit per lineage-grouped fold, used only to estimate
performance; the final model refit on ALL labelled rows, used only to score designs.

Reported on POOLED out-of-fold predictions, the same estimator the composite uses:
  pr_auc_oof / precision_at_10pct  primary
  roc_auc_oof                      secondary
  pr_auc_fold_mean                 spread only -- a different quantity, not comparable

Outputs (EVALUATION_DIR/<dataset>/ranking/):
  rf_design_scores.csv  sample[, dataset], p_binder -- consumed by run_consensus.py
  rf_cv_metrics.csv     the metrics above, plus within-fold mean +- std
  cv_by_fold.csv        per fold: n, prevalence, pr_auc, enrichment, roc_auc, composition
  rf_per_source.csv     per-source PR-AUC / ap_norm / precision@pct
  rf_oof_scores.csv     the per-sample out-of-fold predictions the AUCs come from
  rf_importances.csv    impurity importances of the FINAL model

Usage:  DATASET=rf PYTHONPATH=. python run_ranking_rf.py
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR
from config.analysis import PRECISION_AT_PERCENT
from analysis.distributions import get_numeric_metrics          # leakage-safe feature selector
from analysis.evaluation.ranking_metrics import precision_at_percent
from analysis.evaluation.metrics import per_fold_metrics, fold_summary
from analysis.composite import per_source_metrics   # same per-source measure the composite reports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base = RAW_DATA_DIR / cfg.name
out = EVALUATION_DIR / cfg.name / "ranking"
out.mkdir(parents=True, exist_ok=True)

eval_df = pd.read_csv(base / "eval.csv")
design_df = pd.read_csv(base / "design.csv") if (base / "design.csv").exists() else None

# get_numeric_metrics drops meta + experimental readouts and the filter-only categories, so
# nothing label-adjacent can enter as a feature.
features = get_numeric_metrics(eval_df)
X_train = eval_df[features]
y_train = eval_df["binder"].astype(int)
assert not ({"binder", "KD[M]", "EC50[M]", "iteration", "binder_class", "dataset"} & set(features)), \
    "leak: a meta/experimental column entered the feature set"
print(f"[{cfg.name}] train: {len(X_train)} labelled rows, {X_train.shape[1]} features, "
      f"classes={y_train.value_counts().to_dict()}")

# RF hyperparameters: unchanged from the notebook.
rf = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_split=8,
    min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample",
    bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)

# --- CV evaluation: grouped by parent lineage, so scan variants never split across folds
groups = eval_df["sample"].astype(str).str.split("_").str[0]
n_splits = int(min(5, y_train.value_counts().min(), groups.nunique()))
cv_rows = []
if n_splits >= 2:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = cross_validate(rf, X_train, y_train, groups=groups, cv=cv, scoring=["average_precision"])

    # pooled out-of-fold probabilities: every row predicted by a model that never saw it
    oof = cross_val_predict(rf, X_train, y_train, groups=groups, cv=cv, method="predict_proba")[:, 1]
    p_at_pct, n_pos_at_pct, k_used = precision_at_percent(y_train, oof, PRECISION_AT_PERCENT)

    # per-fold detail from the same splits: folds are near leave-one-source-out, so the spread
    # across them is cross-dataset transfer
    fold_ids = np.full(len(y_train), -1, dtype=int)
    for f, (_, test_idx) in enumerate(cv.split(X_train, y_train, groups)):
        fold_ids[test_idx] = f
    by_fold = per_fold_metrics(y_train, oof, fold_ids, meta=eval_df, model="rf")
    by_fold.to_csv(out / "cv_by_fold.csv", index=False)
    summary = fold_summary(by_fold)

    # kept so the PR / ROC curves come from the reported predictions, not a refit that may drift
    oof_cols = {c: eval_df[c] for c in ("sample", "dataset") if c in eval_df.columns}
    pd.DataFrame({**oof_cols, "binder": y_train, "fold": fold_ids, "p_binder_oof": oof}) \
        .to_csv(out / "rf_oof_scores.csv", index=False)

    cv_rows = [
        {"metric": "pr_auc_oof", "value": round(float(average_precision_score(y_train, oof)), 4),
         "std": np.nan, "n": len(y_train), "n_splits": n_splits,
         "note": "pooled out-of-fold; comparable to the composite and single-metric PR-AUC"},
        {"metric": f"precision_at_{int(PRECISION_AT_PERCENT * 100)}pct",
         "value": round(float(p_at_pct), 4) if p_at_pct == p_at_pct else np.nan,
         "std": np.nan, "n": k_used, "n_splits": n_splits,
         "note": f"pooled out-of-fold; {n_pos_at_pct}/{k_used} binders in the top {k_used}"},
        {"metric": "roc_auc_oof", "value": round(float(roc_auc_score(y_train, oof)), 4),
         "std": np.nan, "n": len(y_train), "n_splits": n_splits,
         "note": "pooled out-of-fold"},
        {"metric": "roc_auc_fold_mean", "value": summary["roc_auc_fold_mean"],
         "std": summary["roc_auc_fold_std"], "n": len(y_train), "n_splits": n_splits,
         "note": "mean +- std of WITHIN-fold ROC-AUC; prevalence-independent and invariant to "
                 "per-fold rescaling, so this is the fairest cross-method comparison"},
        {"metric": "pr_auc_enrichment_fold_mean", "value": summary["pr_auc_enrichment_fold_mean"],
         "std": summary["pr_auc_enrichment_fold_std"], "n": len(y_train), "n_splits": n_splits,
         "note": "mean +- std of within-fold PR-AUC / that fold's prevalence"},
        {"metric": "pr_auc_fold_mean", "value": round(float(res["test_average_precision"].mean()), 4),
         "std": round(float(res["test_average_precision"].std()), 4), "n": len(y_train),
         "n_splits": n_splits, "note": "mean +- std across folds; folds differ in prevalence, so "
                                       "NOT comparable to pr_auc_oof or across methods"},
    ]

    # PER-SOURCE view from the same helper the composite uses, so both are on the same primary
    # measure. The pooled rows above are diagnostics (CLAUDE.md).
    if "dataset" in eval_df.columns:
        ps = per_source_metrics(y_train.to_numpy(), oof, np.isfinite(oof),
                                eval_df["dataset"].to_numpy(), PRECISION_AT_PERCENT, detail=True)
        # the rows the aggregates come from, for a source-by-source comparison with the composite
        pd.DataFrame(ps.pop("per_source")).assign(model="rf").to_csv(out / "rf_per_source.csv", index=False)
        pctl = int(PRECISION_AT_PERCENT * 100)
        cv_rows += [
            {"metric": "ap_norm_ds_mean", "value": ps["ap_norm_ds_mean"], "std": np.nan,
             "n": ps["n_datasets_scored"], "n_splits": n_splits,
             "note": "PRIMARY: (PR-AUC - prevalence)/(1 - prevalence) within each source, averaged; "
                     "0 = no-skill, comparable across sources"},
            {"metric": "ap_norm_ds_min", "value": ps["ap_norm_ds_min"], "std": np.nan,
             "n": ps["n_datasets_scored"], "n_splits": n_splits,
             "note": "robustness: worst source. Negative = below no-skill on some source"},
            {"metric": f"precision_at_{pctl}pct_ds_mean", "value": ps["precision_at_pct_ds_mean"],
             "std": np.nan, "n": ps["n_datasets_scored"], "n_splits": n_splits,
             "note": f"secondary: top-{pctl}% taken WITHIN each source, then averaged"},
            {"metric": f"precision_at_{pctl}pct_ds_min", "value": ps["precision_at_pct_ds_min"],
             "std": np.nan, "n": ps["n_datasets_scored"], "n_splits": n_splits,
             "note": "robustness: worst source"},
            {"metric": "score_spread_ds", "value": ps["score_spread_ds"], "std": np.nan,
             "n": ps["n_datasets_scored"], "n_splits": n_splits,
             "note": "descriptive: max - min of the mean out-of-fold score across sources. Large "
                     "means the pooled ranking largely sorts sources rather than samples"},
        ]
    print("  CV:", {r["metric"]: r["value"] for r in cv_rows})
else:
    print("  CV skipped: too few samples/groups per class for grouped folds")
pd.DataFrame(cv_rows).to_csv(out / "rf_cv_metrics.csv", index=False)

# --- final model: fit on ALL labelled data, then score the designs ---
rf.fit(X_train, y_train)
print("  OOB score:", round(float(rf.oob_score_), 4))
pd.DataFrame({"metric": X_train.columns, "importance": rf.feature_importances_}) \
    .sort_values("importance", ascending=False).to_csv(out / "rf_importances.csv", index=False)

if design_df is not None and len(design_df):
    X_design = design_df.reindex(columns=features)   # same feature order; missing cols -> NaN
    missing = [c for c in features if X_design[c].isna().all()]
    if missing:
        logger.warning("%d of %d features are entirely missing for the designs, so the design "
                       "scores rest on fewer inputs than the CV estimate: %s",
                       len(missing), len(features), missing)
    scores = design_df[[c for c in ("sample", "dataset") if c in design_df.columns]].copy()
    scores["p_binder"] = rf.predict_proba(X_design)[:, 1]
    scores = scores.sort_values("p_binder", ascending=False)
    scores.to_csv(out / "rf_design_scores.csv", index=False)
    print(f"  wrote rf_design_scores.csv ({len(scores)} designs) + rf_cv_metrics.csv "
          f"+ rf_importances.csv -> {out}")
else:
    print(f"  no design.csv for '{cfg.name}'; wrote rf_cv_metrics.csv + rf_importances.csv only")

# No record_stage() here: run_all.sh records every stage uniformly, and self-recording would
# double-count this one in manifest.json.
