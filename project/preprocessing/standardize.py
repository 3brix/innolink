import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_sample_column(df: pd.DataFrame, require_unique: bool = True,) -> None:

    if "sample" not in df.columns: 
        raise ValueError("Mastertable has no sample column.")
    if df["sample"].isna().any(): 
        raise ValueError("Mastertable contains missing sample IDs." )
    duplicates = df["sample"].duplicated().sum()
    if duplicates and require_unique:
        raise ValueError(f"Mastertable contains {duplicates} duplicate sample IDs.")


def ensure_sample_column(df: pd.DataFrame, strategy: str,) -> pd.DataFrame:

    df = df.copy()

    if "sample" in df.columns and df["sample"].notna().all():
        logger.info("Using existing sample column.")
        return df
    if strategy == "nanobody":
        return add_nanobody_sample_id(df)
    elif strategy == "antibody":
        return add_antibody_sample_id(df)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def add_nanobody_sample_id(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Create sample IDs for nanobody mastertables. Naming convention: vhh_target
    """

    df = df.copy()

    df["sample"] = (df["vhh"].astype(str) + "_" + df["target"].astype(str))
    logger.info("Created sample IDs for %d nanobody rows.", len(df),)
    return df


def add_antibody_sample_id(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Create sample IDs for antibody mastertables. Naming convention: fab_target
    """

    df = df.copy()

    df["sample"] = (df["fab"].astype(str) + "_" + df["target"].astype(str))
    logger.info("Created sample IDs for %d antibody rows.", len(df),)
    return df


def clean_nanobody_mastertable(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Keep only columns required downstream.
    """

    keep_cols = ["source", "type", "binder", "sample",]

    df = df[keep_cols].copy()
    logger.info("Nanobody mastertable reduced to %d columns.", len(df.columns),)
    return df


def clean_antibody_mastertable(df: pd.DataFrame,) -> pd.DataFrame:


    keep_cols = ["source", "type", "binder", "sample",]

    df = df[keep_cols].copy()
    df = df.drop_duplicates(keep="last")
    logger.info("Antibody mastertable reduced to %d columns.", len(df.columns),)
    return df


def standardize_nanobody_mastertable(df: pd.DataFrame,) -> pd.DataFrame:

    df = ensure_sample_column(df,strategy="nanobody",)
    df = clean_nanobody_mastertable(df)
    validate_sample_column(df)
    return df


def standardize_antibody_mastertable(df: pd.DataFrame,) -> pd.DataFrame:

    df = ensure_sample_column(df,strategy="antibody",)
    df = clean_antibody_mastertable(df)
    validate_sample_column(df)
    return df


def standardize_mastertable(df: pd.DataFrame,strategy: str,) -> pd.DataFrame:

    if strategy == "nanobody":
        return standardize_nanobody_mastertable(df)
    elif strategy == "antibody":
        return standardize_antibody_mastertable(df)
    else:
        raise ValueError(f"Unknown mastertable strategy: {strategy}")