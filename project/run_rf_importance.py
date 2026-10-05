"""
RF feature-importance comparison -- a separate REPORT, not a pipeline stage.

Three importances for the SAME feature set (whatever get_numeric_metrics selects, 46 for 'rf') and
the SAME hyperparameters/seed as run_ranking_rf.py, so any difference between the columns is the
MEASUREMENT method, not the model or the data:

  gini_final   impurity (Gini) decrease of the FINAL model (fit on all labelled rows).
               Structural: how much impurity a feature removed WHEN it was chosen for a split.
               Normalised to sum to 1. IN-SAMPLE.
  perm_final   permutation importance of that same final model, scored on the rows it was fit on.
               Functional: AP lost when the column is scrambled. IN-SAMPLE too -- the final model
               has no held-out data by construction, so this is NOT an estimate of generalisation.
  perm_cv      permutation importance on lineage-grouped CV: fit on the train folds, scored on the
               HELD-OUT fold, averaged over folds (+- sd). The only out-of-sample column.

Why all three: gini_final and perm_final disagree (Spearman ~0.27) although they describe one
model on one dataset -- among ~46 correlated metrics, which one wins a split is near a lottery, so
Gini rewards the winner while permutation reads ~0 because a substitute covers for it.

Output (under EVALUATION_DIR/<dataset>/ranking/):
  rf_importance_comparison.csv   feature, gini_final, perm_final(+sd), perm_cv(+sd), rank_* per column

Usage:  DATASET=rf PYTHONPATH=. python run_rf_importance.py
        N_REPEATS=10 N_JOBS=4 ...   (defaults 50 / 2; the permutation passes dominate the runtime)
"""

import os

# Set BEFORE the sklearn import: the warning is raised inside joblib workers, which inherit the
# environment but not warnings.filterwarnings().
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR
from analysis.distributions import get_numeric_metrics      # the selector run_ranking_rf.py uses

N_REPEATS = int(os.getenv("N_REPEATS", "50"))
# Modest by default: each worker copies X, and n_jobs=-1 emits one sklearn warning per delayed
# call (a 50-repeat run once produced a 200 MB log). Raise on a bigger machine.
N_JOBS = int(os.getenv("N_JOBS", "2"))
SEED = 42
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

base = RAW_DATA_DIR / cfg.name
out = EVALUATION_DIR / cfg.name / "ranking"
out.mkdir(parents=True, exist_ok=True)

eval_df = pd.read_csv(base / "eval.csv")
features = get_numeric_metrics(eval_df)
X, y = eval_df[features], eval_df["binder"].astype(int)
groups = eval_df["sample"].astype(str).str.split("_").str[0]
print(f"[{cfg.name}] {len(X)} labelled rows, {len(features)} features, "
      f"classes={y.value_counts().to_dict()}, n_repeats={N_REPEATS}")

# hyperparameters: unchanged from run_ranking_rf.py's final model
rf = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_split=8,
    min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample",
    bootstrap=True, random_state=SEED, n_jobs=N_JOBS)

# --- the final model: both in-sample measures come from this one fit
final = clone(rf).fit(X, y)
perm_final = permutation_importance(final, X, y, scoring="average_precision",
                                    n_repeats=N_REPEATS, random_state=SEED, n_jobs=N_JOBS)

# --- grouped CV: permutation scored on the held-out fold (the only out-of-sample column)
n_splits = int(min(5, y.value_counts().min(), groups.nunique()))
perm_cv = np.zeros((n_splits, len(features)))
if n_splits >= 2:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = clone(rf).fit(X.iloc[tr], y.iloc[tr])
        perm_cv[k] = permutation_importance(m, X.iloc[te], y.iloc[te], scoring="average_precision",
                                            n_repeats=N_REPEATS, random_state=SEED, n_jobs=N_JOBS).importances_mean
else:
    perm_cv[:] = np.nan
    print("  grouped CV skipped: too few samples/groups per class")

res = pd.DataFrame({"feature": features,
                    "gini_final": final.feature_importances_,
                    "perm_final": perm_final.importances_mean,
                    "perm_final_sd": perm_final.importances_std,
                    "perm_cv": perm_cv.mean(0),
                    "perm_cv_sd": perm_cv.std(0)})
for c in ("gini_final", "perm_final", "perm_cv"):
    res[f"rank_{c}"] = res[c].rank(ascending=False).astype("Int64")
res = res.sort_values("gini_final", ascending=False).reset_index(drop=True)
res.to_csv(out / "rf_importance_comparison.csv", index=False)

spearman = res[["gini_final", "perm_final", "perm_cv"]].corr(method="spearman")
print(f"  Spearman gini_final vs perm_final: {spearman.loc['gini_final', 'perm_final']:.3f}  "
      f"(both IN-SAMPLE, same model -- the gap is the measurement method)")
print(f"  Spearman gini_final vs perm_cv:    {spearman.loc['gini_final', 'perm_cv']:.3f}")
print(f"  <= 0:  perm_final {int((res.perm_final <= 0).sum())}/{len(res)}   "
      f"perm_cv {int((res.perm_cv <= 0).sum())}/{len(res)}   (held-out is the honest one)")
print(f"  sums:  gini {res.gini_final.sum():.3f} (normalised)   perm_final {res.perm_final.sum():.3f}   "
      f"perm_cv {res.perm_cv.sum():.3f}   -- different units, never one shared axis")
print(f"-> {out / 'rf_importance_comparison.csv'}")
