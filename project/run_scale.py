"""Scale the selected dataset / pooled set -> merged_scaled_aligned.csv.

SCALER env var picks 'standard' (default) or 'robust'. ONE scaler is fit on the LABELLED rows of
merged_aligned.csv and applied to every row, so a design's scaled value is a fixed function of
the benchmark rather than of which designs share its file (CLAUDE.md, scaler default).

Only merged is written: the eval / design subsets are just this table filtered on `binder`.
Run the prep / align steps first.
"""

import os
import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR
from preprocessing.metric_meta import get_all_metric_columns
from preprocessing.scale import fit_scaler, apply_scaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD = os.environ.get("SCALER", "standard")

base = RAW_DATA_DIR / cfg.name
reference_path = base / "merged_aligned.csv"

if not reference_path.exists():
    raise SystemExit(f"run_scale: {reference_path} not found; run run_data_prep.py and run_align.py first.")

table = pd.read_csv(reference_path)
if table.empty:
    raise SystemExit(f"run_scale: {reference_path.name} has no rows; nothing to fit.")

# fit on the labelled rows; a design-only set has none, so fall back to the whole table
fit_rows = table[pd.to_numeric(table["binder"], errors="coerce").isin([0, 1])] if "binder" in table.columns else table
if fit_rows.empty:
    logger.warning("no labelled rows in %s; fitting on all %d rows instead", reference_path.name, len(table))
    fit_rows = table

metrics = get_all_metric_columns(table)   # same columns the aligned table covers
scaler = fit_scaler(fit_rows, metrics, method=METHOD)
logger.info("Fit %s scaler on %d labelled of %d rows in %s (%d metrics)",
            METHOD, len(fit_rows), len(table), reference_path.name, len(metrics))

out = base / "merged_scaled_aligned.csv"
apply_scaler(table, metrics, scaler).to_csv(out, index=False)
logger.info("Wrote %s (%s, fit on the labelled rows)", out.name, METHOD)

print(f"Scaled table ({METHOD}, fit on {len(fit_rows)} labelled rows, applied to {len(table)}) written: {out}")
