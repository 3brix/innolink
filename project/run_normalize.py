import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, METRIC_YAML
from preprocessing.normalize import normalize_dataframe


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


base = RAW_DATA_DIR / cfg.name

written = []
for name in ("merged", "eval", "design"):
    src = base / f"{name}_aligned.csv"
    if not src.exists():
        logger.info("Skipping %s (base table not found)", src.name)
        continue
    normalized = normalize_dataframe(pd.read_csv(src), METRIC_YAML)
    out = base / f"{name}_normalized.csv"
    normalized.to_csv(out, index=False)
    written.append(out.name)
    logger.info("Wrote %s", out)

print(f"Normalized tables written to {base}: {', '.join(written) or '(none)'}")