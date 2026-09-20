from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score, matthews_corrcoef
from preprocessing.metric_meta import get_direction, get_family, load_families as load_metric_families

def pass_mask(values: pd.Series, cutoff: float, direction: int) -> pd.Series:
    """Boolean mask: True where a value is on the better side of `cutoff` for its
    direction (>= when higher-is-better, <= when lower-is-better). NaN -> False."""
    aligned = values * direction
    return (aligned >= cutoff * direction).fillna(False)


def precision_recall_at(values: pd.Series, y_true: pd.Series, cutoff: float,
                        direction: int) -> dict:
    """Precision / recall / N-passing when `values` is thresholded at `cutoff`
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


def _pr_counts(aligned, y, cutoff):
    """precision, recall, n_pass, n_pos_retained (TP) for predict-positive if aligned >= cutoff."""
    passed = aligned >= cutoff
    n_pass = int(passed.sum())
    tp = int((passed & (y == 1)).sum())
    n_pos = int((y == 1).sum())
    prec = tp / n_pass if n_pass else float("nan")
    rec = tp / n_pos if n_pos else float("nan")
    return prec, rec, n_pass, tp


def select_threshold(y_true, aligned_scores, precision_target: float, n_min: int) -> dict:
    """Choose a cutoff on ALIGNED scores (higher = better) for precision-first filtering.

    Rule: among candidate cutoffs (the observed score values) that reach
    precision >= `precision_target` AND retain >= `n_min` samples, pick the one with the
    HIGHEST recall (i.e. the most-retaining cutoff that still meets the target).
    Fallbacks (reported via `status`):
      - "ok"              : a cutoff met both constraints.
      - "no_target"       : none met the precision target with n_min retained; the
                            highest-precision cutoff among those with >= n_min is used.
      - "no_nmin"         : fewer than n_min samples exist; the highest-precision cutoff
                            overall is used.
    Returns {threshold(aligned), precision, recall, n_pass, n_pos_retained, status}.
    Missing scores are dropped (they never pass).
    """
    import numpy as np
    y = np.asarray(y_true).astype(int)
    a = np.asarray(aligned_scores, dtype=float)
    keep = ~np.isnan(a)
    a, y = a[keep], y[keep]
    if a.size == 0:
        return {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "n_pass": 0, "n_pos_retained": 0, "status": "empty"}

    cutoffs = np.unique(a)
    scan = [(c, *_pr_counts(a, y, c)) for c in cutoffs]   # (cutoff, prec, rec, n_pass, tp)

    def pack(row, status):
        c, prec, rec, n_pass, tp = row
        return {"threshold": float(c), "precision": round(prec, 4) if n_pass else float("nan"),
                "recall": round(rec, 4), "n_pass": n_pass, "n_pos_retained": tp, "status": status}

    meets = [r for r in scan if r[3] >= n_min and not np.isnan(r[1]) and r[1] >= precision_target]
    if meets:
        # highest recall; tie-break on higher precision then fewer passed (tighter)
        best = max(meets, key=lambda r: (r[2], r[1], -r[3]))
        return pack(best, "ok")
    with_nmin = [r for r in scan if r[3] >= n_min]
    if with_nmin:
        best = max(with_nmin, key=lambda r: (r[1] if not np.isnan(r[1]) else -1, r[2]))
        return pack(best, "no_target")
    best = max(scan, key=lambda r: (r[1] if not np.isnan(r[1]) else -1, r[2]))
    return pack(best, "no_nmin")


def calculate_all_metrics(df: pd.DataFrame, metrics: list[str], directions: dict[str, int]) -> pd.DataFrame:
    """Per-metric ROC-AUC, PR-AUC, and F1 at the PR-curve-optimal threshold."""
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
            pr_auc = average_precision_score(y_true, aligned)

            precisions, recalls, thresholds = precision_recall_curve(y_true, aligned)
            # precisions/recalls carry an extra (0, 1) endpoint -> use [:-1]
            f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

            if f1_scores.max() == 0:
                results.append({
                    "metric": col, "raw_roc": round(raw_roc, 4), "aligned_roc": round(aligned_roc, 4),
                    "pr_auc": round(pr_auc, 4), "f1": 0.0, "precision": 0.0, "recall": 0.0,
                    "mcc": np.nan,
                    "opt_threshold": np.nan, "opt_threshold_raw": np.nan,
                    "direction": direction, "note": "degenerate_f1",
                })
                continue

            best = int(np.argmax(f1_scores))
            best_thr = thresholds[best]
            y_pred = (aligned >= best_thr).astype(int)

            # MCC at the same F1-optimal threshold (supplementary, balanced measure).
            # Undefined when predictions or labels are single-class -> NaN (documented).
            mcc = (matthews_corrcoef(y_true, y_pred)
                   if y_true.nunique() > 1 and len(np.unique(y_pred)) > 1 else np.nan)

            results.append({
                "metric": col,
                "raw_roc": round(raw_roc, 4),
                "aligned_roc": round(aligned_roc, 4),
                "pr_auc": round(pr_auc, 4),
                "f1": round(f1_score(y_true, y_pred), 4),
                "mcc": round(float(mcc), 4) if mcc == mcc else np.nan,
                "precision": round(float(precisions[:-1][best]), 4),
                "recall": round(float(recalls[:-1][best]), 4),
                "opt_threshold": round(float(best_thr), 4),
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
    """
    Picks the top "top_n" ranked metrics by separation, one per YAML "family".
    Metrics with an unknown family assigned to their own name as the family key.
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