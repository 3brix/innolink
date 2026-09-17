import logging
import numpy as np
import pandas as pd

from config.analysis import EXCLUDE_COLUMNS
from preprocessing.metadata import add_binder_type


logger = logging.getLogger(__name__)



def check_basic_integrity(df: pd.DataFrame) -> dict:

    report = {}

    # Dataset size
    report["n_rows"] = len(df)
    report["n_samples"] = (df["sample"].nunique())

    # Duplicates
    report["duplicate_samples"] = (df["sample"].duplicated().sum())

    # Metadata integrity
    report["missing_binder"] = (df["binder"].isna().sum())
    report["binder_values"] = (df["binder"].value_counts().to_dict())

    # label-consistency: check if decoy (target shuffle, binder = 0) is mislabeled as 1
    if "type" in df.columns:
        _t = df["type"].astype(str).str.strip().str.lower()
        _b = pd.to_numeric(df["binder"], errors="coerce")
        report["unknown_type"] = int(((_t == "target_shuffle") & (_b != 0)).sum())

    # Metric overview
    metric_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    report["n_metrics"] = len(metric_columns)

    metric_missing = (df[metric_columns].isna().sum().sort_values(ascending=False))
    report["missing_metrics"] = (metric_missing[metric_missing > 0].to_dict())

    # Infinities (invalid; must be caught before they poison correlations / scaling)
    numeric = df[metric_columns].select_dtypes(include="number")
    inf_counts = pd.Series(np.isinf(numeric.to_numpy()).sum(axis=0), index=numeric.columns).sort_values(ascending=False)
    report["infinite_metrics"] = (inf_counts[inf_counts > 0].to_dict())

    logger.info("QC: %d rows, %d samples, %d metrics", report["n_rows"], report["n_samples"], report["n_metrics"])

    return report

# TO DO: generate the report only if there is at least one non-finite value in the dataset
def nonfinite_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric non-finite counts (NaN / +inf / -inf); one row per metric that has any, worst first."""
    numeric = df.select_dtypes(include="number")
    metrics = [c for c in numeric.columns if c not in EXCLUDE_COLUMNS]
    X = numeric[metrics]
    Xv = X.to_numpy()

    rep = pd.DataFrame({
        "metric": metrics,
        "n_nan": X.isna().sum().to_numpy(),
        "n_posinf": np.isposinf(Xv).sum(axis=0),
        "n_neginf": np.isneginf(Xv).sum(axis=0),
    })
    rep["n_invalid"] = rep["n_nan"] + rep["n_posinf"] + rep["n_neginf"]
    return rep[rep["n_invalid"] > 0].sort_values("n_invalid", ascending=False).reset_index(drop=True)



def missing_metrics_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample breakdown of which metrics are missing. Header only when no missing values."""

    metric_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    meta_cols = [c for c in ("sample", "dataset", "source", "type", "binder") if c in df.columns]

    isna = df[metric_columns].isna()
    n_missing = isna.sum(axis=1)

    rows = []
    for idx in df.index[n_missing > 0]:
        row = {c: df.at[idx, c] for c in meta_cols}
        row["n_missing_metrics"] = int(n_missing.at[idx])
        row["missing_metrics"] = "; ".join(isna.columns[isna.loc[idx]])
        rows.append(row)

    columns = meta_cols + ["n_missing_metrics", "missing_metrics"]
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("n_missing_metrics", ascending=False).reset_index(drop=True)


def metric_description(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric summary statistics """

    numeric = df.select_dtypes(include="number").columns
    metrics = [c for c in numeric if c not in EXCLUDE_COLUMNS]

    summary = (df[metrics].describe().T.reset_index().rename(columns={"index": "metric"}))
    summary["missing"] = len(df) - summary["count"]

    return summary