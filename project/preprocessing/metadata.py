"""Sample-level metadata labels derived from the merged table."""

from __future__ import annotations

import pandas as pd


def add_binder_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add `binder_class` from `binder` + `type`. The single source of truth for that column.

    Binder / Non-Binder (type original), Mutant+ / Mutant- (dms, alanine_scan), Target Shuffle,
    Design (binder == "?"), else Unknown. `binder` is coerced to numeric so "1"/1.0 both work.

    TODO: Design keys on the literal "?" only -- decide whether NaN should count too.
    """
    df = df.copy()
    binder = pd.to_numeric(df["binder"], errors="coerce")
    sample_type = df["type"].astype(str).str.strip().str.lower()
    is_design = df["binder"].astype(str).str.strip() == "?"

    label = pd.Series("Unknown", index=df.index)
    label[(binder == 1) & (sample_type.isin(["dms", "alanine_scan"]))] = "Mutant+"
    label[(binder == 0) & (sample_type.isin(["dms", "alanine_scan"]))] = "Mutant-"
    label[(binder == 0) & (sample_type == "target_shuffle")] = "Target Shuffle"
    label[(binder == 0) & (sample_type == "original")] = "Non-Binder"
    label[(binder == 1) & (sample_type == "original")] = "Binder"
    label[is_design] = "Design"          # unlabeled

    df["binder_class"] = label
    return df


def split_eval_design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the merged table into (evaluation, design) rows."""
    is_design = df["binder"].astype(str).str.strip() == "?"
    return df[~is_design].copy(), df[is_design].copy()