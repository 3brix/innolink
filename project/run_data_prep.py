"""Create the 3 base tables for the selected DATASET: merged.csv, eval.csv, design.csv."""

import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR
from preprocessing.standardize import standardize_mastertable
from preprocessing.data_prep import prepare_dataset, save_dataframe, save_eval_design, warn_experimental_columns
from analysis.io import load_processed_datasets


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


output_path = RAW_DATA_DIR / cfg.name / "merged.csv"


if cfg.is_set:
    # pooled set: concatenate member datasets, then split into eval / design
    logger.info("Pooling set '%s' from members: %s", cfg.name, ", ".join(cfg.members))

    pooled = load_processed_datasets(cfg.members, RAW_DATA_DIR)
    logger.info(
        "Pooled %d rows across %d datasets",
        len(pooled), pooled["dataset"].nunique(),
    )

    warn_experimental_columns(pooled)               # leakage guard           
    save_dataframe(pooled, output_path)             # pooled merged.csv
    save_eval_design(pooled, output_path)           # eval.csv / design.csv split


    logger.info("Wrote pooled base tables to %s", output_path.parent)
    print(f"Pooled '{cfg.name}' ({len(pooled)} rows) -> {output_path.parent}")

else:
    # single dataset: standardize mastertable, merge with predictions
    logger.info("Dataset: %s", cfg.name)
    logger.info("Predictions: %s", cfg.predictions_path)
    logger.info("Mastertable: %s", cfg.mastertable_path)
    logger.info("Interfaces: %s", cfg.keep_interfaces)

    mastertable = pd.read_csv(cfg.mastertable_path)
    mastertable = standardize_mastertable(mastertable, strategy=cfg.mastertable_strategy)

    merged = prepare_dataset(
        predictions_path=cfg.predictions_path,
        mastertable=mastertable,
        output_path=output_path,
        keep_interfaces=cfg.keep_interfaces,
    )

    logger.info("Finished. Base tables for %d rows written to %s", len(merged), output_path.parent)
    print(merged.head())