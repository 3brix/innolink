"""Profiling summary tables (not plots) for run_profiling.py.

Every function groups by whichever of dataset / mol_type is present, so the same call works for
a single dataset and for a pooled set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.metric_meta import get_all_metric_columns


def _present(df: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if c in df.columns]


def class_distribution(df: pd.DataFrame, by=("dataset", "mol_type")) -> pd.DataFrame:
    """
    Counts of 'binder_class' overall. Derives it when absent and 'type' is available,
    else falls back to counting the raw 'binder' label.
    """
    if "binder_class" not in df.columns:
        if "type" in df.columns and "binder" in df.columns:
            from preprocessing.metadata import add_binder_class
            df = add_binder_class(df)
        else:
            out = (df["binder"].astype(str).value_counts().rename("n")
                   .rename_axis("binder").reset_index())
            return out
    group = _present(df, by)
    if group:
        out = (df.groupby(group)["binder_class"].value_counts()
               .rename("n").reset_index())
    else:
        out = (df["binder_class"].value_counts().rename("n")
               .rename_axis("binder_class").reset_index())
    return out


def composition(df: pd.DataFrame, by=("dataset", "mol_type")) -> pd.DataFrame:
    """Per (dataset / mol_type sample counts and positive-class prevalence."""
    def _agg(sub: pd.DataFrame) -> pd.Series:
        b = pd.to_numeric(sub["binder"], errors="coerce")
        n_bind = int((b == 1).sum())
        n_non = int((b == 0).sum())
        n_design = int((sub["binder"].astype(str).str.strip() == "?").sum())
        lab = n_bind + n_non
        return pd.Series({
            "n_samples": len(sub), "n_binder": n_bind, "n_nonbinder": n_non,
            "n_design": n_design,
            "prevalence": round(n_bind / lab, 4) if lab else np.nan,
        })

    group = _present(df, by)
    if group:
        return df.groupby(group).apply(_agg).reset_index()
    return _agg(df).to_frame().T.reset_index(drop=True)


def missingness_summary(df: pd.DataFrame, group_col: str = "dataset") -> pd.DataFrame:
    """Per-metric missing fraction overall, plus per-'group_col' columns if present.

    Uses the FULL metric set (incl. developability/energy/sequence): missingness is one of the
    main reasons a metric gets excluded, so it must be reported for every metric."""
    metrics = get_all_metric_columns(df)
    out = (df[metrics].isna().mean().round(4)
           .rename("missing_frac_overall").rename_axis("metric").reset_index())
    if group_col in df.columns:
        per = df.groupby(group_col)[metrics].apply(lambda g: g.isna().mean()).round(4)
        per = per.T  # metric x group
        per.columns = [f"missing_{group_col}_{c}" for c in per.columns]
        out = out.merge(per.reset_index().rename(columns={"index": "metric"}), on="metric", how="left")
    return out

# TO DO: migrate function (?)
def metric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric range/summary (min/max/mean/std/median + missing), via QC's describe."""
    from analysis.distributions.qc import metric_description
    return metric_description(df)


def benchmark_structure(df: pd.DataFrame, k: int = 5, seed: int = 42):
    """Lineage-group sizes and CV fold composition for the LABELLED rows.

    Reuses the composite/RF grouping helpers so the folds reported here are the folds actually
    scored downstream (StratifiedGroupKFold on parent lineages, n_splits capped by the rarer class
    and the group count). Returns (groups, folds):
      groups  lineage, dataset, n, n_binder, prevalence      -- one row per lineage, largest first
      folds   fold, n, prevalence, n_groups + one count column per dataset
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from analysis.composite.cv import _lineage_groups, _effective_splits

    lab = df[pd.to_numeric(df["binder"], errors="coerce").isin([0, 1])].copy()
    lab["lineage"] = _lineage_groups(lab)
    y = pd.to_numeric(lab["binder"], errors="coerce").astype(int)

    groups = (lab.assign(_y=y).groupby("lineage")
              .agg(dataset=("dataset", lambda s: s.iloc[0]), n=("lineage", "size"), n_binder=("_y", "sum"))
              .reset_index())
    groups["prevalence"] = (groups["n_binder"] / groups["n"]).round(3)
    groups = groups.sort_values("n", ascending=False).reset_index(drop=True)

    n_splits = _effective_splits(y, lab["lineage"].to_numpy(), k)
    rows = []
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (_, test) in enumerate(cv.split(lab, y, lab["lineage"]), start=1):
            t = lab.iloc[test]
            row = {"fold": i, "n": len(test), "prevalence": round(float(y.iloc[test].mean()), 3),
                   "n_groups": int(t["lineage"].nunique())}
            row.update(t["dataset"].value_counts().to_dict())
            rows.append(row)
    folds = pd.DataFrame(rows).fillna(0)
    for c in folds.columns:
        if c != "prevalence":
            folds[c] = folds[c].astype(int)
    return groups, folds
