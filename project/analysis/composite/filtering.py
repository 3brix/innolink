"""Config-driven quality filtering (S1). Cutoffs are raw-score, so this runs on the raw table."""

from __future__ import annotations

import pandas as pd

from preprocessing.align import get_direction


def quality_filter(df: pd.DataFrame, thresholds: dict, directions: dict, mode: str = "any") -> pd.DataFrame:
    """Add a `quality_flag` column ("low_confidence" / "ok") -- the S1 confidence screen.

    Which side flags is taken from each metric's direction. mode="any" flags on any matching
    rule, "all" only when all match. Missing columns are skipped and NaN passes.
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
