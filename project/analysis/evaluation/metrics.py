from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from preprocessing.align import get_direction

def load_metric_families(yaml_path) -> dict[str, str]:
    """Load each metric's "family" from the YAML, sorted longest-key-first."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    raw = {name: info.get("family") for name, info in data["metrics"].items()}
    return dict(sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True))
 
 
def get_family(col: str, families: dict[str, str]) -> str | None:
    """Return a column's family via exact / longest-suffix match. None if unknown."""
    for base, family in families.items():
        if col == base or col.endswith(f"_{base}"):
            return family
    return None


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
                    "opt_threshold": np.nan, "opt_threshold_raw": np.nan,
                    "direction": direction, "note": "degenerate_f1",
                })
                continue

            best = int(np.argmax(f1_scores))
            best_thr = thresholds[best]
            y_pred = (aligned >= best_thr).astype(int)

            results.append({
                "metric": col,
                "raw_roc": round(raw_roc, 4),
                "aligned_roc": round(aligned_roc, 4),
                "pr_auc": round(pr_auc, 4),
                "f1": round(f1_score(y_true, y_pred), 4),
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