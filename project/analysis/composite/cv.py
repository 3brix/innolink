
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing.align import get_direction
from analysis.evaluation.metrics import get_family, per_fold_metrics, fold_summary
from analysis.evaluation.ranking_metrics import precision_at_percent


def _aligned_matrix(df: pd.DataFrame, cols: list[str], directions: dict) -> np.ndarray:
    """(n x len(cols)) direction-aligned values (higher = better)."""
    signs = np.array([get_direction(c, directions) for c in cols], dtype=float)
    return df[cols].to_numpy(dtype=float) * signs


def _lineage_groups(df: pd.DataFrame) -> np.ndarray:
    """Lineage id = first '_'-token of the sample id; one group per row if there is no 'sample'."""
    if "sample" in df.columns:
        return df["sample"].astype(str).str.split("_").str[0].to_numpy()
    return np.arange(len(df))


def _effective_splits(y, groups, k: int) -> int:
    """n_splits capped by the minority class and the group count; <2 means grouped CV is impossible."""
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return 0
    min_class = int(np.bincount(y).min())
    n_groups = int(np.unique(groups).size)
    return min(int(k), min_class, n_groups)


def _finite(y, s, min_n: int = 5):
    """(y, s) on finite scores, or None when unusable (< min_n rows, or one class left)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    m = np.isfinite(s)
    if m.sum() < min_n or np.unique(y[m]).size < 2:
        return None
    return y[m], s[m]


def average_precision(y, s, min_n: int = 5) -> float:
    """PR-AUC on a metric's observed values only (its own complete-case subset); NaN if unusable."""
    f = _finite(y, s, min_n)
    return float("nan") if f is None else float(average_precision_score(*f))


def roc_auc(y, s, min_n: int = 5) -> float:
    """ROC-AUC over a metric's observed values only; NaN when unusable."""
    f = _finite(y, s, min_n)
    return float("nan") if f is None else float(roc_auc_score(*f))


def best_per_family(prauc: dict[str, float], families: dict) -> list[str]:
    """Highest-PR-AUC metric per family (unknown family -> its own name); NaN PR-AUC skipped."""
    best: dict[str, str] = {}
    for m, p in {k: v for k, v in prauc.items() if v == v}.items():
        fam = get_family(m, families) or m
        if fam not in best or p > prauc[best[fam]]:
            best[fam] = m
    return list(best.values())


def dataset_aware_select(df, y, candidate_cols, directions, families, dataset_col="dataset",
                         tie_margin=0.05, return_table=False):
    """Best-in-family metric by ap_norm = (PR-AUC - prevalence) / (1 - prevalence), averaged
    across SOURCE DATASETS. Deterministic: no folds, no seed.

    Raw per-dataset PR-AUC is not averageable (its baseline is that dataset's prevalence,
    0.09-0.74 here), so ap_norm is what makes sources comparable.

    Ranking is on the MEAN, with one override: a family's leader is replaced only when it is
    below no-skill somewhere AND a metric within `tie_margin` of its mean is not. Ranking every
    near-tie on the min instead was rejected -- see Appendix_composite_selection.md.

    return_table=True also returns the ap_norm table, with 'margin' (best-to-runner-up gap) and
    'decided_by' ('mean' or 'min').
    """
    y = np.asarray(y)
    ds = df[dataset_col].astype(str).to_numpy() if dataset_col in df.columns else np.full(len(df), "pooled")
    A = _aligned_matrix(df, candidate_cols, directions)
    rows = []
    for i, c in enumerate(candidate_cols):
        vals = []
        for d in np.unique(ds):
            m = ds == d
            if np.unique(y[m]).size < 2:              # single-class source -> PR-AUC undefined
                continue
            ap, prev = average_precision(y[m], A[m, i]), float(y[m].mean())
            if ap == ap and prev < 1:
                vals.append((ap - prev) / (1 - prev))
        if vals:
            rows.append({"metric": c, "family": get_family(c, families) or c,
                         "ap_norm_ds_mean": round(float(np.mean(vals)), 4),
                         "ap_norm_ds_min": round(float(np.min(vals)), 4), "n_datasets": len(vals)})
    table = pd.DataFrame(rows)
    if table.empty:
        return ([], table) if return_table else []
    chosen, margins, decided = [], {}, {}
    for _, g in table.groupby("family"):
        g = g.sort_values("ap_norm_ds_mean", ascending=False)
        best = float(g.iloc[0]["ap_norm_ds_mean"])
        margins_gap = round(best - float(g.iloc[1]["ap_norm_ds_mean"]), 4) if len(g) > 1 else float("nan")
        lead = g.iloc[0]
        tied = g[(g["ap_norm_ds_mean"] >= best - tie_margin) & (g["metric"] != lead["metric"])]
        rescue = tied[tied["ap_norm_ds_min"] >= 0]                   # near-tied and never below no-skill
        if lead["ap_norm_ds_min"] < 0 and len(rescue):
            win = rescue.sort_values("ap_norm_ds_min", ascending=False).iloc[0]["metric"]
            decided[win] = "min (leader below no-skill)"
        else:
            win = lead["metric"]
            decided[win] = "mean"
        chosen.append(win)
        margins[win] = margins_gap
    table["chosen"] = table["metric"].isin(chosen)
    table["margin"] = table["metric"].map(margins)     # family's best-to-runner-up gap in mean ap_norm
    table["decided_by"] = table["metric"].map(decided)
    table = table.sort_values(["chosen", "ap_norm_ds_mean"], ascending=[False, False]).reset_index(drop=True)
    return (sorted(chosen), table) if return_table else sorted(chosen)


def stability_select(df, y, candidate_cols, directions, families, k=5, seed=42, groups=None,
                     return_votes=False):
    """Per-fold best-in-family vote over lineage-grouped folds, keeping majority winners.

    NOT the default -- dataset_aware_select is. It ranks on PR-AUC pooled within a training fold,
    which lets the largest source decide the family. Kept only so earlier numbers reproduce via
    selection='stability'; see CLAUDE.md. Falls back to in-sample best-per-family if the vote is
    empty. return_votes=True also returns (votes, n_splits).
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
            prauc = {c: average_precision(y[tr], A[:, i]) for i, c in enumerate(candidate_cols)}
            for m in best_per_family(prauc, families):
                votes[m] += 1
    chosen = sorted(m for m, v in votes.items() if v > n_splits // 2)
    if not chosen:  # degenerate -> fall back to a single in-sample best-per-family
        A = _aligned_matrix(df, candidate_cols, directions)
        prauc = {c: average_precision(y, A[:, i]) for i, c in enumerate(candidate_cols)}
        chosen = sorted(best_per_family(prauc, families))
    return (chosen, votes, n_splits) if return_votes else chosen


def _ap_norm_weights(y, A, ds, tr):
    """Per-metric mean ap_norm across the sources in a training fold (the weighting counterpart
    of dataset_aware_select). Negative averages clip to 0."""
    w = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        vals = []
        for src in np.unique(ds[tr]):
            sel = tr[ds[tr] == src]
            if np.unique(y[sel]).size < 2:
                continue
            ap, prev = average_precision(y[sel], A[sel, i]), float(y[sel].mean())
            if ap == ap and prev < 1:
                vals.append((ap - prev) / (1 - prev))
        w[i] = max(float(np.mean(vals)), 0.0) if vals else 0.0
    return w


def cv_composite_scores(df, y, cols, directions, method="consensus", k=5, seed=42, groups=None,
                        return_folds=False):
    """Out-of-fold composite score per sample over lineage-grouped folds, so scan variants of one
    parent never split across train/test.

    Standardisation and the weights are fit on the train folds only. Choosing 'cols' is NOT
    fold-local -- the caller selects the metric set once on all labelled rows. NaNs ignored per
    row; all-NaN when grouped CV is impossible.

    method: 'consensus' (equal-weight z mean), 'weighted' (by train-fold per-source ap_norm),
    'product' (geometric mean of Phi(z); the OOF counterpart of combine.product_score).

    return_folds=True also returns the per-row fold index (-1 where unscored).
    """
    y = np.asarray(y)
    A = _aligned_matrix(df, cols, directions)
    ds = df["dataset"].to_numpy() if "dataset" in df.columns else np.zeros(len(df), dtype=int)
    n = len(df)
    oof = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    if groups is None:
        groups = _lineage_groups(df)
    n_splits = _effective_splits(y, groups, k)
    if n_splits < 2:
        return (oof, fold_id) if return_folds else oof
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(sgkf.split(A, y, groups)):
        mu = np.nanmean(A[tr], axis=0)
        sd = np.nanstd(A[tr], axis=0)
        sd = np.where(sd > 0, sd, 1.0)          # zero-variance -> leave centred
        Z = (A[te] - mu) / sd
        fold_id[te] = f
        if method == "product":
            p = norm.cdf(Z)
            logp = np.log(np.clip(p, 1e-9, 1.0))
            with np.errstate(invalid="ignore"):
                oof[te] = np.exp(np.nanmean(logp, axis=1))   # NaN metrics ignored per row
            continue
        if method == "weighted" and np.unique(y[tr]).size > 1:
            w = _ap_norm_weights(y, A, ds, tr)   # per-source mean ap_norm, fit on the train fold
            if w.sum() == 0:
                w = np.ones(len(cols))
        else:  # consensus (equal weight); also the fallback for a class-pure train fold
            w = np.ones(len(cols))
        present = ~np.isnan(Z)
        denom = (present * w).sum(axis=1)
        num = np.nansum(np.where(present, Z, 0.0) * w, axis=1)
        oof[te] = np.divide(num, denom, out=np.full(len(te), np.nan), where=denom > 0)
    return (oof, fold_id) if return_folds else oof


def per_source_metrics(y, s, mask, ds, pct, detail=False):
    """Per-SOURCE view of a composite score -- the PRIMARY measures (CLAUDE.md); the pooled ones
    are contaminated by between-source score offsets.

      ap_norm_ds_*           per-source (PR-AUC - prevalence) / (1 - prevalence), 0 = no-skill
      precision_at_pct_ds_*  top-pct taken WITHIN each source, then aggregated
      score_spread_ds        max - min of the per-source mean score. Large = the pooled ranking
                             is mostly sorting sources, so read the pooled columns as suspect.

    detail=True also returns "per_source": one row per source, the values these aggregate.
    """
    if ds is None:
        return {}
    aps, precs, means, rows = [], [], [], []
    for src in np.unique(ds[mask]):
        k_ = mask & (ds == src)
        ys, ss = y[k_], s[k_]
        means.append(float(np.mean(ss)))
        row = {"source": src, "n": int(len(ys)), "n_pos": int(ys.sum()),
               "prevalence": round(float(ys.mean()), 4), "mean_score": round(float(np.mean(ss)), 4)}
        if np.unique(ys).size < 2:
            rows.append(row)
            continue
        prev = float(ys.mean())
        if prev < 1:
            ap = float(average_precision_score(ys, ss))
            aps.append((ap - prev) / (1 - prev))
            row |= {"pr_auc": round(ap, 4), "ap_norm": round(aps[-1], 4)}
        pak, _, _ = precision_at_percent(ys, ss, pct)
        if pak == pak:
            precs.append(float(pak))
            row["precision_at_pct"] = round(float(pak), 4)
        rows.append(row)
    r = lambda v: round(float(v), 4)
    return {"ap_norm_ds_mean": r(np.mean(aps)) if aps else np.nan,
            "ap_norm_ds_min": r(np.min(aps)) if aps else np.nan,
            "precision_at_pct_ds_mean": r(np.mean(precs)) if precs else np.nan,
            "precision_at_pct_ds_min": r(np.min(precs)) if precs else np.nan,
            "n_datasets_scored": len(aps),
            "score_spread_ds": r(max(means) - min(means)) if len(means) > 1 else np.nan,
            **({"per_source": rows} if detail else {})}


def evaluate_composites(df, y, candidate_cols, directions, families,
                        k=5, seed=42, selection="dataset", fixed_cols=None,
                        pct=0.10, return_votes=False) -> tuple[pd.DataFrame, list[str]]:
    """Composite comparison with lineage-grouped CV -> (table, selected_cols, by_fold, oof).

    One row per rule (consensus / weighted / product) plus a best_single baseline, which is
    in-sample and therefore optimistic.

    WHAT THE CV NUMBERS ARE: out-of-fold GIVEN THE METRIC SET. Standardisation and weights are
    fit on training folds, but the metric set was chosen once on all labelled rows, so these
    measure stability across lineages and sources, NOT generalisation to an unseen dataset. The
    'evaluation' column says so on every row.

    selection : 'dataset' (default, dataset_aware_select), 'stability' (the old fold vote) or
                'fixed' (use `fixed_cols`).
    return_votes : also return the selection record -- the ap_norm table for 'dataset', the
                per-fold ballot for 'stability', empty for 'fixed'.

    Every composite row also carries the per-source columns; prefer those over the pooled ones.
    """
    y = np.asarray(y)
    groups = _lineage_groups(df)
    ds = df["dataset"].to_numpy() if "dataset" in df.columns else None
    n_splits = _effective_splits(y, groups, k)
    votes: Counter = Counter()
    sel_table = pd.DataFrame()
    if selection == "fixed":
        if not fixed_cols:
            raise ValueError("selection='fixed' requires fixed_cols")
        cols = [c for c in fixed_cols if c in df.columns]
    elif selection == "dataset":
        cols, sel_table = dataset_aware_select(df, y, candidate_cols, directions, families,
                                               return_table=True)
        # Reported beside the ap_norm columns, never selected on: lineage groups nest inside
        # datasets, so a fold vote is a vote about one source. See CLAUDE.md. Few votes + a narrow
        # margin flags a thin pick.
        _, fold_votes, n_folds = stability_select(df, y, candidate_cols, directions, families, k, seed,
                                                  groups=groups, return_votes=True)
        sel_table["fold_votes"] = sel_table["metric"].map(lambda m: int(fold_votes.get(m, 0)))
        sel_table["n_folds"] = n_folds
    else:
        cols, votes, n_splits = stability_select(df, y, candidate_cols, directions, families, k, seed,
                                                 groups=groups, return_votes=True)
    if not cols:
        raise ValueError("no metrics selected for the composite")

    def _rank_cols(ys, ss):
        # point estimates, no bootstrap
        pak, npak, kused = precision_at_percent(ys, ss, pct)
        return {"prevalence": round(float((ys == 1).mean()), 4), "pct": pct,
                "precision_at_pct": round(pak, 4) if pak == pak else np.nan,
                "n_pos_at_pct": npak, "k": kused}

    eval_label = (f"{n_splits}-fold lineage-grouped CV, robustness analysis (out-of-fold given the "
                  f"metric set; the set was selected on all labelled rows, so this is not an "
                  f"independent estimate)" if n_splits >= 2
                  else "grouped CV not possible (too few lineage groups / samples)")
    rows, by_fold, oof_rows = [], [], []
    for method in ("consensus", "weighted", "product"):
        s, folds = cv_composite_scores(df, y, cols, directions, method, k, seed, groups=groups,
                                       return_folds=True)
        m = ~np.isnan(s)
        if m.sum() == 0 or np.unique(y[m]).size < 2:     # grouped CV degenerate
            continue
        model = f"composite_{method}"
        # kept so PR / ROC curves come from the reported predictions, not a recomputation
        oof_rows.append(pd.DataFrame({**{c: df[c].to_numpy() for c in ("sample", "dataset") if c in df},
                                      "binder": y, "fold": folds, "model": model, "score": s}))
        folds_df = per_fold_metrics(y, s, folds, meta=df, model=model)
        by_fold.append(folds_df)
        rows.append({"model": model,
                     "auroc": round(float(roc_auc_score(y[m], s[m])), 4),
                     "pr_auc": round(float(average_precision_score(y[m], s[m])), 4),
                     **_rank_cols(y[m], s[m]),
                     **per_source_metrics(y, s, m, ds, pct),
                     **fold_summary(folds_df),
                     "n_metrics": len(cols), "evaluation": eval_label})

    # best-single baseline: selected on the same rows it is scored on, so OPTIMISTIC and not
    # directly comparable to the out-of-fold composites above.
    A = _aligned_matrix(df, candidate_cols, directions)
    singles = {c: roc_auc(y, A[:, i]) for i, c in enumerate(candidate_cols)}
    singles = {c: v for c, v in singles.items() if v == v}
    if singles:
        top = max(singles, key=singles.get)
        _col = A[:, candidate_cols.index(top)]
        _y, _top_scores = _finite(y, _col)
        rows.append({"model": f"best_single:{top}",
                     "auroc": round(float(singles[top]), 4),
                     "pr_auc": round(float(average_precision(_y, _top_scores)), 4),
                     **_rank_cols(_y, _top_scores),
                     **per_source_metrics(y, _col, np.isfinite(_col), ds, pct),
                     "n_metrics": 1,
                     "evaluation": f"in-sample, selected over {len(candidate_cols)} metrics (optimistic)"})

    by_fold_df = pd.concat(by_fold, ignore_index=True) if by_fold else pd.DataFrame()
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    out = (pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True),
           cols, by_fold_df, oof_df)
    if not return_votes:
        return out
    record = sel_table if selection == "dataset" else selection_votes(votes, cols, families, n_splits)
    return out + (record,)


def selection_votes(votes, chosen, families, n_splits) -> pd.DataFrame:
    """The fold ballot as a table: one row per metric that won a fold, plus vote-less chosen ones."""
    metrics = sorted(set(votes) | set(chosen))
    rows = [{"metric": m, "family": get_family(m, families), "votes": int(votes.get(m, 0)),
             "n_splits": int(n_splits), "chosen": m in set(chosen)} for m in metrics]
    if not rows:
        return pd.DataFrame(columns=["metric", "family", "votes", "n_splits", "chosen"])
    return pd.DataFrame(rows).sort_values(["chosen", "votes", "metric"],
                                          ascending=[False, False, True]).reset_index(drop=True)
