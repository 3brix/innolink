# Load predictions and master table, drop unnecessary columns, merge on "sample"

from pathlib import Path
import logging

import pandas as pd

from config.prep import COLS_TO_DROP
from config.analysis import EXCLUDE_COLS
from preprocessing.chai_metrics import process_chai_metrics
from preprocessing.metadata import add_binder_type, split_eval_design
from preprocessing.metric_meta import load_categories, get_category
from config.analysis import FILTER_ONLY_CATEGORIES



logger = logging.getLogger(__name__)


def warn_experimental_columns(df: pd.DataFrame) -> list[str]:       
    """
    Leakage guard: flag experimental affinity config.analysis.EXCLUDE_COLS (EXCLUDE_COLS = KD[M], EC50[M]),
    Returns the offending column names (+ logged as a warning).
    """
    present = [c for c in EXCLUDE_COLS if c in df.columns]
    if present:
        logger.warning(
            "merged table retains experimental readout column(s) %s -- exclude from any model "
            "(use get_numeric_metrics); they are labels, not features.", present,
        )
    return present


def load_predictions(
    predictions_path: str | Path,
    target_shuffle_path: str | Path | None = None,
    keep_interfaces: set[str] | None = None,
) -> pd.DataFrame:
    """
    Load prediction CSVs and optionally filter interfaces.

    predictions_path: path to predictions CSV.
    target_shuffle_path: optional path to target shuffle predictions.  ->> maybe drop?
    keep_interfaces: interfaces to retain; None applies no interface filtering.
    """
    df = pd.read_csv(predictions_path)
    if target_shuffle_path is not None:
        ts = pd.read_csv(target_shuffle_path)
        df = pd.concat([df, ts], ignore_index=True)
    if keep_interfaces is not None:
        df = df[df["interface"].str.strip().isin(keep_interfaces)].copy()
    logger.info("Loaded %d prediction rows.", len(df))
    return df


def aggregate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predictions to one row per sample (numeric mean, other types first)."""
    numeric_cols = df.select_dtypes(include="number").columns
    other_cols = [c for c in df.columns if c not in numeric_cols and c != "sample"]
    agg = {c: "max" for c in numeric_cols} # was mean first decided to change it to max or weighted mean
    agg.update({c: "first" for c in other_cols})
    return df.groupby("sample").agg(agg).copy().reset_index()


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove columns that are not needed downstream."""
    return df.drop(columns=columns, errors="ignore")


# plddt for af3,cf(0-100) and boltz, esmfold(0-1) are on different scales, so we need to normalize them to be comparable
# normalize plddt to 0-1 scale for columns "af3_plddt","af3_iplddt" and "cf_plddt", cf "iplddt" by dividing by 100
# TO DO: pae /pde?
def normalize_plddt(df):
    df = df.copy()
    for col in ["af3_plddt", "af3_iplddt", "cf_plddt", "cf_iplddt"]:  # + esmfold_plddt, esmfold_iplddt --> esmfold currently excluded from analysis
        if col in df.columns:
            df[col] = df[col] / 100.0   
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: set[str], name: str = "dataframe") -> None:
    """Validate that required columns exist."""
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def merge_with_mastertable(predictions: pd.DataFrame, mastertable: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions with mastertable by sample."""
    validate_dataframe(mastertable, {"sample"}, name="mastertable")
    merged = mastertable.merge(predictions, on="sample", how="inner")
    logger.info("Merged dataset contains %d samples.", len(merged))
    return merged


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s", path)


def save_eval_design(merged: pd.DataFrame, output_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split "merged" into evaluation / design base tables and save both.
    design.csv is only written when design rows exist. Returns (eval_df, design_df).
    """
    base = Path(output_path)
    eval_df, design_df = split_eval_design(merged)

    save_dataframe(eval_df, base.with_name("eval.csv"))
    if len(design_df):
        save_dataframe(design_df, base.with_name("design.csv"))

    return eval_df, design_df


def prepare_dataset(
    predictions_path: str | Path,
    mastertable: pd.DataFrame,
    output_path: str | Path,
    keep_interfaces: set[str] | None = None,
    target_shuffle_path: str | Path | None = None,
    mol_type: str | None = None,
):
    """
    Build tables: merged.csv + eval.csv + design.csv.
    """
    predictions = load_predictions(predictions_path, target_shuffle_path, keep_interfaces)
    predictions = aggregate_predictions(predictions)
    # keep developability/energy columns 
    _cats = load_categories()
    _drop = [c for c in COLS_TO_DROP if get_category(c, _cats) not in FILTER_ONLY_CATEGORIES]
    predictions = drop_columns(predictions, _drop)
    predictions = normalize_plddt(predictions)

    validate_dataframe(mastertable, {"sample"}, name="mastertable")
    merged = merge_with_mastertable(predictions, mastertable)
    merged = process_chai_metrics(merged)
    merged = add_binder_type(merged)
    if mol_type is not None:
        merged["mol_type"] = mol_type          # nanobody / antibody, persisted for stratified reporting
    warn_experimental_columns(merged)

    save_dataframe(merged, output_path)
    save_eval_design(merged, output_path)

    return merged