from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from preprocessing.align import get_direction


def _pos_neg(df: pd.DataFrame, col: str, directions: dict):
    """Direction-aligned (positives, negatives) for one metric, or None if unusable."""
    clean = df[[col, "binder"]].dropna()
    if clean["binder"].nunique() < 2:
        return None
    score = clean[col].values * get_direction(col, directions)
    y = clean["binder"].astype(int).values
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return pos, neg


def _hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Deterministic SE of the AUROC (Hanley & McNeil, 1982)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def responsiveness(df: pd.DataFrame, metrics: list[str], directions: dict, group_col: str = "dataset") -> pd.DataFrame:
    """
    One row per (group, metric): cliffs_delta (+95% CI), auroc, n_pos, n_neg, sign.
    Groups by `group_col` if present (pooled SET), else treats the whole frame as one
    series ("all"). Low-n groups still report -- the CI shows how wide the uncertainty is.
    """
    groups = df.groupby(group_col) if group_col in df.columns else [("all", df)]
    rows = []
    for grp, g in groups:
        for col in metrics:
            pn = _pos_neg(g, col, directions)
            if pn is None:
                continue
            pos, neg = pn; n_pos, n_neg = len(pos), len(neg)
            u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
            auc = u / (n_pos * n_neg)
            delta = 2 * auc - 1
            se_d = 2 * _hanley_mcneil_se(auc, n_pos, n_neg)
            rows.append({group_col: grp, "metric": col,
                         "cliffs_delta": round(delta, 4),
                         "ci_low": round(max(-1.0, delta - 1.96 * se_d), 4),
                         "ci_high": round(min(1.0, delta + 1.96 * se_d), 4),
                         "auroc": round(auc, 4), "n_pos": n_pos, "n_neg": n_neg,
                         "sign": int(np.sign(round(delta, 4)))})
    return pd.DataFrame(rows).sort_values(["metric", group_col]) if rows else pd.DataFrame(
        columns=[group_col, "metric", "cliffs_delta", "ci_low", "ci_high", "auroc", "n_pos", "n_neg", "sign"])


def consistency(resp: pd.DataFrame, group_col: str = "dataset") -> pd.DataFrame:
    """
    Cross-series summary per metric: how many systems it was measured in, the mean /
    min / max Cliff's delta, and whether the SIGN is consistent across all systems
    (min > 0 or max < 0). `consistent_sign` + a decent |mean_delta| is the "trustworthy
    as a gate" signal; a metric that flips sign between systems is not.
    """
    if resp.empty:
        return pd.DataFrame(columns=["metric", "n_groups", "mean_delta", "min_delta", "max_delta", "consistent_sign"])
    s = resp.groupby("metric").agg(n_groups=(group_col, "nunique"),
                                   mean_delta=("cliffs_delta", "mean"),
                                   min_delta=("cliffs_delta", "min"),
                                   max_delta=("cliffs_delta", "max")).reset_index()
    s["consistent_sign"] = (s["min_delta"] > 0) | (s["max_delta"] < 0)
    for c in ["mean_delta", "min_delta", "max_delta"]:
        s[c] = s[c].round(4)
    return s.sort_values("mean_delta", key=lambda x: x.abs(), ascending=False)
