"""
Sample-level metadata labels derived from the merged table.
"""

from __future__ import annotations

import pandas as pd


def add_binder_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the human-readable binder class column, driven by "binder" + "type".

    Precedence:
      Non-Binder:     binder == 0 and type == "original"
      Mutant-:         binder == 0 and type in {"dms", "alanine_scan"}
      Binder:         binder == 1
      Mutant +:         type in {"dms", "alanine_scan"}
      Target Shuffle: type == "target_shuffle"
      Design:         binder == "?"
      Unknown:        anything else

    "binder" is coerced to numeric so a column read as strings ("1"/"0") or floats (1.0) still classifies as Binder/Non-Binder.
    Design is keyed on the unlabeled marker binder == "?" --> NaN too? check
    This is the single source of truth for "binder_type" (preprocessing, metadata for plots).
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

    df["binder_type"] = label
    return df


def split_eval_design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the merged table into evaluation and design rows.
    """
    is_design = df["binder"].astype(str).str.strip() == "?"
    return df[~is_design].copy(), df[is_design].copy()