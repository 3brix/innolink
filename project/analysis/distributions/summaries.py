"""
Lightweight profiling summaries (tables, not plots).

Reusable functions that produce the reproducible profiling tables written by
run_profiling.py. The figure notebooks import these too, so the computation lives
here once (no duplication between notebook and script).

All functions gracefully group by whichever of `dataset` / `mol_type` columns are
present, so they work for a single dataset and for pooled sets, and pick up
`mol_type` automatically once it is added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.metric_meta import get_metric_columns


def _present(df: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if c in df.columns]


def class_distribution(df: pd.DataFrame, by=("dataset", "mol_type")) -> pd.DataFrame:
    """Counts of `binder_type` overall and grouped by whichever `by` columns exist.
    Derives `binder_type` if absent (via add_binder_type) when `type` is available,
    else falls back to counting the raw `binder` label."""
    if "binder_type" not in df.columns:
        if "type" in df.columns and "binder" in df.columns:
            from preprocessing.metadata import add_binder_type
            df = add_binder_type(df)
        else:
            out = (df["binder"].astype(str).value_counts().rename("n")
                   .rename_axis("binder").reset_index())
            return out
    group = _present(df, by)
    if group:
        out = (df.groupby(group)["binder_type"].value_counts()
               .rename("n").reset_index())
    else:
        out = (df["binder_type"].value_counts().rename("n")
               .rename_axis("binder_type").reset_index())
    return out


def composition(df: pd.DataFrame, by=("dataset", "mol_type")) -> pd.DataFrame:
    """Per (dataset[/mol_type]) sample counts and positive-class prevalence."""
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
    """Per-metric missing fraction overall, plus per-`group_col` columns if present."""
    metrics = get_metric_columns(df)
    out = (df[metrics].isna().mean().round(4)
           .rename("missing_frac_overall").rename_axis("metric").reset_index())
    if group_col in df.columns:
        per = df.groupby(group_col)[metrics].apply(lambda g: g.isna().mean()).round(4)
        per = per.T  # metric x group
        per.columns = [f"missing_{group_col}_{c}" for c in per.columns]
        out = out.merge(per.reset_index().rename(columns={"index": "metric"}), on="metric", how="left")
    return out


def metric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric range/summary (min/max/mean/std/median + missing), via QC's describe."""
    from analysis.distributions.qc import metric_description
    return metric_description(df)
