"""
Create the sacled tables for the selected dataset.

The scaler is chosen by the scaler env var ('standard' or 'robust', default 'robust'); 
parameters are fit per file, so for a pooled set the fit is on the pool. Run the prep / normalize / align steps first.
"""

import os
import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR
from preprocessing.scale import scale_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD = os.environ.get("SCALER", "standard")

# source-suffix -> scaled-output-suffix
VARIANTS = {
    #"": "_scaled",
    "_aligned": "_scaled_aligned",
}

base = RAW_DATA_DIR / cfg.name
written = []
for name in ("merged", "eval", "design"):
    for src_suffix, out_suffix in VARIANTS.items():
        src = base / f"{name}{src_suffix}.csv"
        if not src.exists():
            continue
        scaled = scale_dataframe(pd.read_csv(src), method=METHOD)
        out = base / f"{name}{out_suffix}.csv"
        scaled.to_csv(out, index=False)
        written.append(out.name)
        logger.info("Wrote %s (%s)", out.name, METHOD)

print(f"Scaled tables ({METHOD}) written to {base}: {', '.join(written) or '(none)'}")
