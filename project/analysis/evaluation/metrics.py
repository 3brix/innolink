from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from preprocessing.metric_meta import get_direction, get_family, load_families as load_metric_families  # re-exported via analysis.evaluation.__init__
from analysis.evaluation.ranking_metrics import precision_at_percent
from config.analysis import PRECISION_AT_PERCENT

def per_fold_metrics(y_true, scores, fold_ids, meta: pd.DataFrame | None = None,
                     group_col: str = "dataset", model: str | None = None) -> pd.DataFrame:
    """One row per CV fold: n, n_pos, prevalence, pr_auc, pr_auc_enrichment, roc_auc.

    PR-AUC's baseline is the fold's prevalence, so folds are not on a common scale --
    pr_auc_enrichment (pr_auc / prevalence) removes that, and ROC-AUC is prevalence-independent.
    Pass `meta` to record each fold's composition. NaN scores and fold_id < 0 are dropped.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    folds = np.asarray(fold_ids)
    rows = []
    for f in sorted({int(v) for v in folds if v >= 0}):
        m = (folds == f) & ~np.isnan(s) & ~np.isnan(y)
        if m.sum() == 0:
            continue
        yf, sf = y[m].astype(int), s[m]
        prevalence = float(yf.mean())
        two_class = len(np.unique(yf)) > 1
        pr_auc = float(average_precision_score(yf, sf)) if two_class else np.nan
        row = {"fold": f, "n": int(m.sum()), "n_pos": int(yf.sum()),
               "prevalence": round(prevalence, 4),
               "pr_auc": round(pr_auc, 4) if pr_auc == pr_auc else np.nan,
               "pr_auc_enrichment": round(pr_auc / prevalence, 4) if (pr_auc == pr_auc and prevalence) else np.nan,
               "roc_auc": round(float(roc_auc_score(yf, sf)), 4) if two_class else np.nan}
        if model is not None:
            row = {"model": model, **row}
        if meta is not None and group_col in meta.columns:
            counts = meta.loc[m, group_col].value_counts()
            row["composition"] = "; ".join(f"{k}:{v}" for k, v in counts.items())
        rows.append(row)
    return pd.DataFrame(rows)


def fold_summary(by_fold: pd.DataFrame) -> dict:
    """mean +- std across folds for the per-fold columns of `per_fold_metrics`."""
    out = {}
    for col in ("roc_auc", "pr_auc_enrichment", "pr_auc"):
        if col in by_fold.columns:
            vals = by_fold[col].dropna()
            out[f"{col}_fold_mean"] = round(float(vals.mean()), 4) if len(vals) else np.nan
            out[f"{col}_fold_std"] = round(float(vals.std(ddof=0)), 4) if len(vals) else np.nan
    return out


def pass_mask(values: pd.Series, cutoff: float, direction: int) -> pd.Series:
    """Boolean mask: True where a value is on the better side of 'cutoff' for its
    direction (>= when higher-is-better, <= when lower-is-better). NaN -> False."""
    aligned = values * direction
    return (aligned >= cutoff * direction).fillna(False)


def precision_recall_at(values: pd.Series, y_true: pd.Series, cutoff: float,
                        direction: int) -> dict:
    """Precision / recall / N-passing when 'values' is thresholded at 'cutoff'
    (direction-aware). Missing values do not pass."""
    mask = values.notna() & y_true.notna()
    if mask.sum() == 0 or cutoff is None or not np.isfinite(cutoff):
        return {"precision": np.nan, "recall": np.nan, "n_pass": 0, "n_eval": int(mask.sum())}
    passed = pass_mask(values[mask], cutoff, direction)
    y = y_true[mask].astype(int)
    n_pass = int(passed.sum())
    tp = int((passed & (y == 1)).sum())
    n_pos = int((y == 1).sum())
    return {"precision": round(tp / n_pass, 4) if n_pass else np.nan,
            "recall": round(tp / n_pos, 4) if n_pos else np.nan,
            "n_pass": n_pass, "n_eval": int(mask.sum())}


def calculate_all_metrics(df: pd.DataFrame, metrics: list[str], directions: dict[str, int],
                          pct: float = PRECISION_AT_PERCENT) -> pd.DataFrame:
    """Per-metric benchmark: ROC-AUC, PR-AUC (primary), precision@pct, and precision/recall at
    the F1-optimal threshold. In-sample, point estimates only."""
    results = []

    for col in metrics:
        if df[col].notna().sum() < 2 or df[col].nunique() < 2:
            continue

        try:
            clean = df[[col, "binder"]].dropna()
            if clean["binder"].nunique() < 2:
                continue

            y_true = clean["binder"].astype(int)
            y_raw = clean[col].values
            direction = get_direction(col, directions)
            aligned = y_raw * direction

            raw_roc = roc_auc_score(y_true, y_raw)
            aligned_roc = roc_auc_score(y_true, aligned)
            # TO TEST: lineage-grouped (cluster) bootstrap CI for PR-AUC --> resample parent
            # lineages (sample.split('_')[0]), not rows, since DMS variants are non-independent
            pr_auc = average_precision_score(y_true, aligned)

            # Each quantity is computed on THIS metric's observed rows, so n_eval and prevalence
            # vary between metrics and PR-AUC is not directly comparable across them -- always
            # report n_eval alongside pr_auc.
            prevalence = round(float((y_true == 1).mean()), 4)
            p_at_pct, n_pos_at_pct, k_used = precision_at_percent(y_true, aligned, pct)
            _rank_cols = {
                "n_eval": int(len(clean)), "prevalence": prevalence, "pct": pct,
                "precision_at_pct": round(p_at_pct, 4) if p_at_pct == p_at_pct else np.nan,
                "n_pos_at_pct": n_pos_at_pct, "k": k_used,
                "pr_auc_vs_prevalence": round(float(pr_auc / prevalence), 4) if prevalence else np.nan,
            }

            precisions, recalls, thresholds = precision_recall_curve(y_true, aligned)
            # precisions/recalls carry an extra (0, 1) endpoint -> use [:-1]
            f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

            if f1_scores.max() == 0:
                results.append({
                    "metric": col, "raw_roc": round(raw_roc, 4), "aligned_roc": round(aligned_roc, 4),
                    "pr_auc": round(pr_auc, 4), **_rank_cols, "precision": 0.0, "recall": 0.0,
                    "opt_threshold": np.nan, "opt_threshold_raw": np.nan,
                    "direction": direction, "note": "degenerate_f1",
                })
                continue

            best = int(np.argmax(f1_scores))
            best_thr = thresholds[best]

            # F1 and MCC are deliberately absent here: threshold-dependent at a point the ranking
            # task never uses. MCC lives in the RF stage only.
            results.append({
                "metric": col,
                "raw_roc": round(raw_roc, 4),
                "aligned_roc": round(aligned_roc, 4),
                "pr_auc": round(pr_auc, 4), **_rank_cols,
                "precision": round(float(precisions[:-1][best]), 4),
                "recall": round(float(recalls[:-1][best]), 4),
                "opt_threshold": round(float(best_thr), 4),
                # the same cutoff in NATIVE units, read by run_thresholds.py
                # (F1 and MCC themselves stay out -- see the note above.)
                "opt_threshold_raw": round(float(best_thr * direction), 4),
                "direction": direction,
                "note": "",
            })

        except Exception as e:
            results.append({"metric": col, "note": f"error: {e}"})
            continue

    out = pd.DataFrame(results)
    return out.sort_values("pr_auc", ascending=False) if "pr_auc" in out.columns else out



def top_metrics_distinct_family(
    rankings: pd.DataFrame,
    families: dict[str, str],
    top_n: int = 5,
    metric_col: str = "metric",
    sort_col: str = "pr_auc",
) -> pd.DataFrame:
    """Top `top_n` metrics by separation, one per family (unknown family -> its own name).
    """
    ranked = rankings.sort_values(sort_col, ascending=False) if sort_col in rankings.columns else rankings

    seen: set[str] = set()
    keep_idx = []
    for idx, metric in ranked[metric_col].items():
        family = get_family(metric, families) or metric
        if family in seen:
            continue
        seen.add(family)
        keep_idx.append(idx)
        if len(keep_idx) >= top_n:
            break

    return ranked.loc[keep_idx].copy()