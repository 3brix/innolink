"""
Config-driven quality filtering.

Thresholds live as "{metric: value}" in config. Because the thresholds are raw-score cutoffs (e.g. af3_plddt < 80), the filter runs on the raw merged table.
"""

from __future__ import annotations

import pandas as pd

from preprocessing.align import get_direction


def quality_filter(df: pd.DataFrame, thresholds: dict, directions: dict, mode: str = "any") -> pd.DataFrame:
    """
    Confidence QUALITY screen -- one of three distinct "filter" concepts in this project:
      1. quality_filter (here)            -- flags low-confidence samples
      2. analysis.feasibility.apply_gates -- literature / custom developability / energy / error gates;
      3. run_filter.py                    -- the design filter stage combining 1 + 2

    Add a "quality_flag" column ("low_confidence" / "ok").
    The flag side is taken from each metric's direction. 
    mode="any" flags a sample if any rule matches, "all" only if all match. 
    Missing columns are skipped --> NaN values do not match a rule (they pass).
    """
    df = df.copy()
    masks = []
    for col, thr in thresholds.items():
        if col not in df.columns:
            continue
        if get_direction(col, directions) >= 0:
            masks.append(df[col] < thr)   # higher = better -> low value is bad
        else:
            masks.append(df[col] > thr)   # lower = better  -> high value is bad

    if not masks:
        df["quality_flag"] = "ok"
        return df

    combined = pd.concat(masks, axis=1)
    flagged = combined.any(axis=1) if mode == "any" else combined.all(axis=1)
    df["quality_flag"] = flagged.map({True: "low_confidence", False: "ok"})
    return df
