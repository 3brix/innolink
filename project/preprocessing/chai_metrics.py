import ast

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)



# Chai array columns
CHAI_PTM_COLUMNS = ["chai_unconstrained_per_chain_ptm", "chai_constrained_per_chain_ptm"]

CHAI_PAIR_IPTM_COLUMNS = ["chai_unconstrained_per_chain_pair_iptm", "chai_constrained_per_chain_pair_iptm"]

CHAI_ARRAY_COLUMNS = (CHAI_PTM_COLUMNS + CHAI_PAIR_IPTM_COLUMNS)



# Helpers
def parse_array(value):
    """Parse a stringified Chai array into a numpy array."""

    if pd.isna(value):
        return None
    try:
        return np.array(ast.literal_eval(value)).squeeze()
    except Exception:
        return None


def extract_ptm(arr):
    """Return the mean pTM over all chains."""

    if arr is None or arr.ndim != 1:
        return np.nan
    return arr.mean()


def extract_pair_iptm(matrix):
    """
    Extract interface iPTM.
    Two-chain complexes: minimum(A,B)
    Three-chain complexes: minimum(A,C,B,C)
    """

    if matrix is None or matrix.ndim != 2:
        return np.nan
    if matrix.shape[0] == 2:
        return min(matrix[0, 1], matrix[1, 0])
    if matrix.shape[0] == 3:
        ac = min(matrix[0, 2], matrix[2, 0])
        bc = min(matrix[1, 2], matrix[2, 1])
        return min(ac, bc)
    return np.nan



# Feature extraction
def extract_chai_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Chai array columns into numeric features."""
    available = [c for c in CHAI_ARRAY_COLUMNS if c in df.columns]

    missing = set(CHAI_ARRAY_COLUMNS) - set(available)

    if missing:
        logger.warning("Missing Chai columns: %s", sorted(missing))

    if not available:
        logger.warning("No Chai array columns found. Skipping Chai processing.")
        return df[["sample"]].copy()

    chai = df[["sample"] + available].copy()

    # Parse arrays
    for col in available:
        chai[f"{col}_parsed"] = chai[col].apply(parse_array)

    # Extract pTM
    for col in CHAI_PTM_COLUMNS:
        if col in available:
            chai[col] = (chai[f"{col}_parsed"].apply(extract_ptm))

    # Extract pair iPTM
    for col in CHAI_PAIR_IPTM_COLUMNS:
        if col in available:
            chai[col] = (chai[f"{col}_parsed"].apply(extract_pair_iptm))

    return chai[["sample"] + [c for c in CHAI_ARRAY_COLUMNS if c in chai.columns]].copy()



# Public API
def process_chai_metrics(df):

    chai_numeric = extract_chai_metrics(df)

    df = df.drop(columns=CHAI_ARRAY_COLUMNS, errors="ignore")

    df = df.merge(chai_numeric, on="sample", how="left", validate="one_to_one",)

    logger.info("Dropped raw CHAI array columns: %s", [c for c in CHAI_ARRAY_COLUMNS if c in df.columns])
    logger.info("Processed CHAI metrics for %d samples. Added columns: %s", len(df), [c for c in chai_numeric.columns if c != "sample"])

    return df