
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing.align import get_direction
from analysis.evaluation.metrics import get_family
from analysis.evaluation.ranking_metrics import precision_at_percent


def _aligned_matrix(df: pd.DataFrame, cols: list[str], directions: dict) -> np.ndarray:
    """(n x len(cols)) direction-aligned values (higher = better)."""
    signs = np.array([get_direction(c, directions) for c in cols], dtype=float)
    return df[cols].to_numpy(dtype=float) * signs


def _lineage_groups(df: pd.DataFrame) -> np.ndarray:
    """Parent-lineage id = first '_'-token of the sample id,
    Falls back to one group per row (i.e. ungrouped) when there is no 'sample' column."""
    if "sample" in df.columns:
        return df["sample"].astype(str).str.split("_").str[0].to_numpy()
    return np.arange(len(df))


def _effective_splits(y, groups, k: int) -> int:
    """n_splits capped by the minority-class count AND the number of lineage groups
    (StratifiedGroupKFold needs n_splits <= #groups). <2 means grouped CV isn't possible."""
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return 0
    min_class = int(np.bincount(y).min())
    n_groups = int(np.unique(groups).size)
    return min(int(k), min_class, n_groups)


def best_per_family(prauc: dict[str, float], families: dict) -> list[str]:
    """Keep the highest-PR-AUC metric within each YAML family (unknown family -> its own name)."""
    best: dict[str, str] = {}
    for m, p in prauc.items():
        fam = get_family(m, families) or m
        if fam not in best or p > prauc[best[fam]]:
            best[fam] = m
    return list(best.values())


def stability_select(df, y, candidate_cols, directions, families, k=5, seed=42, groups=None) -> list[str]:
    """
    Stability selection with lineage-grouped folds (StratifiedGroupKFold): within each fold,
    pick the best-in-family metric by PR-AUC, then keep only metrics chosen in a majority
    (> n_splits//2) of folds. Falls back to a full in-sample best-per-family when the
    vote is empty or grouped CV isn't possible.
    """
    y = np.asarray(y)
    if groups is None:
        groups = _lineage_groups(df)
    n_splits = _effective_splits(y, groups, k)
    votes: Counter = Counter()
    if n_splits >= 2:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, _ in sgkf.split(df[candidate_cols].to_numpy(), y, groups):
            if np.unique(y[tr]).size < 2:            # grouped folds can be class-pure -> skip
                continue
            A = _aligned_matrix(df.iloc[tr], candidate_cols, directions)
            prauc = {c: average_precision_score(y[tr], A[:, i]) for i, c in enumerate(candidate_cols)}
            for m in best_per_family(prauc, families):
                votes[m] += 1
    chosen = sorted(m for m, v in votes.items() if v > n_splits // 2)
    if not chosen:  # degenerate -> fall back to a single in-sample best-per-family
        A = _aligned_matrix(df, candidate_cols, directions)
        prauc = {c: average_precision_score(y, A[:, i]) for i, c in enumerate(candidate_cols)}
        chosen = sorted(best_per_family(prauc, families))
    return chosen


def cv_composite_scores(df, y, cols, directions, method="consensus", k=5, seed=42, groups=None) -> np.ndarray:
    """
    Out-of-fold composite score per sample, using lineage-grouped folds (StratifiedGroupKFold)
    so mutational-scan variants of one parent never split across train/test. Standardization
    (and PR-AUC weights for method='weighted') are fit on the train folds only, so no label
    leaks into the held-out score. NaNs are ignored per row. 
    Returns all-NaN when grouped CV isn't possible (too few groups/samples).
    """
    y = np.asarray(y)
    A = _aligned_matrix(df, cols, directions)
    n = len(df)
    oof = np.full(n, np.nan)
    if groups is None:
        groups = _lineage_groups(df)
    n_splits = _effective_splits(y, groups, k)
    if n_splits < 2:
        return oof
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(A, y, groups):
        mu = np.nanmean(A[tr], axis=0)
        sd = np.nanstd(A[tr], axis=0)
        sd = np.where(sd > 0, sd, 1.0)          # zero-variance -> leave centred
        Z = (A[te] - mu) / sd
        if method == "weighted" and np.unique(y[tr]).size > 1:
            w = np.array([average_precision_score(y[tr], A[tr][:, i]) for i in range(len(cols))])
            if w.sum() == 0:
                w = np.ones(len(cols))
        else:  # consensus (equal weight); also the fallback for a class-pure train fold
            w = np.ones(len(cols))
        present = ~np.isnan(Z)
        denom = (present * w).sum(axis=1)
        num = np.nansum(np.where(present, Z, 0.0) * w, axis=1)
        oof[te] = np.divide(num, denom, out=np.full(len(te), np.nan), where=denom > 0)
    return oof


def evaluate_composites(df, y, candidate_cols, directions, families,
                        k=5, seed=42, selection="stability", fixed_cols=None,
                        pct=0.10) -> tuple[pd.DataFrame, list[str]]:
    """
    Composite comparison with lineage-grouped out-of-fold CV. Returns (table,
    selected_cols). Table has one row per method: composite_consensus (primary),
    composite_weighted (secondary), best_single (baseline) with out-of-fold AUROC, PR-AUC, and precision@pct
    Folds are grouped by parent lineage (StratifiedGroupKFold)
    Single-metric AUROC involves no fitted parameters (optimistic)
    selection : 'stability' (default; majority best-in-family across grouped folds) or 'fixed'
                (use 'fixed_cols', a domain-chosen metric set).

    """
    y = np.asarray(y)
    groups = _lineage_groups(df)
    n_splits = _effective_splits(y, groups, k)
    if selection == "fixed":
        if not fixed_cols:
            raise ValueError("selection='fixed' requires fixed_cols")
        cols = [c for c in fixed_cols if c in df.columns]
    else:
        cols = stability_select(df, y, candidate_cols, directions, families, k, seed, groups=groups)
    if not cols:
        raise ValueError("no metrics selected for the composite")

    def _rank_cols(ys, ss):
        # ys/ss are OUT-OF-FOLD scores for composites (leakage-safe)
        # precision@pct + PR-AUC, reported as point estimates.
        pak, npak, kused = precision_at_percent(ys, ss, pct)
        return {"prevalence": round(float((ys == 1).mean()), 4), "pct": pct,
                "precision_at_pct": round(pak, 4) if pak == pak else np.nan,
                "n_pos_at_pct": npak, "k": kused}

    eval_label = (f"{n_splits}-fold grouped CV (out-of-fold, lineage)" if n_splits >= 2
                  else "grouped CV not possible (too few lineage groups / samples)")
    rows = []
    for method in ("consensus", "weighted"):
        s = cv_composite_scores(df, y, cols, directions, method, k, seed, groups=groups)
        m = ~np.isnan(s)
        if m.sum() == 0 or np.unique(y[m]).size < 2:     # grouped CV degenerate
            continue
        rows.append({"model": f"composite_{method}",
                     "auroc": round(float(roc_auc_score(y[m], s[m])), 4),
                     "pr_auc": round(float(average_precision_score(y[m], s[m])), 4),
                     **_rank_cols(y[m], s[m]),
                     "n_metrics": len(cols), "evaluation": eval_label})

    # best single metric baseline (aligned, no fitting -> full-set AUROC is unbiased per metric)
    A = _aligned_matrix(df, candidate_cols, directions)
    singles = {c: roc_auc_score(y, A[:, i]) for i, c in enumerate(candidate_cols)}
    top = max(singles, key=singles.get)
    _top_scores = A[:, candidate_cols.index(top)]
    rows.append({"model": f"best_single:{top}",
                 "auroc": round(float(singles[top]), 4),
                 "pr_auc": round(float(average_precision_score(y, _top_scores)), 4),
                 **_rank_cols(y, _top_scores),
                 "n_metrics": 1, "evaluation": "full-set (single metric, no fit)"})

    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True), cols
