"""
PROFILING stage (lightweight, reproducible tables).

Describes the data before benchmarking: class distribution, dataset/molecule-type
composition and positive-class prevalence, per-metric missingness, and metric ranges.
Detailed distribution/outlier/redundancy FIGURES stay in the optional notebooks
(distributions_plots.ipynb, profiles_plots.ipynb, evaluation_plots.ipynb), which import
the same summary functions so no computation is duplicated.

Outputs (under PROCESSED_DATA_DIR/profiling/<dataset>/):
  composition.csv        per dataset[/mol_type]: counts + prevalence
  class_distribution.csv binder_type counts overall + per dataset[/mol_type]
  missingness.csv        per-metric missing fraction (overall + per dataset)
  metric_ranges.csv      per-metric min/max/mean/std/median + missing
"""

import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_directory
from analysis.distributions.summaries import (
    class_distribution, composition, missingness_summary, metric_ranges,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base = RAW_DATA_DIR / cfg.name
output_dir = ensure_directory(PROCESSED_DATA_DIR / "profiling" / cfg.name)

# Profile the full merged table (all rows incl. designs) so composition/missingness
# cover everything; class distribution naturally reflects labelled vs design.
df = pd.read_csv(base / "merged.csv")
logger.info("Profiling %s (%d rows, %d columns)", cfg.name, len(df), df.shape[1])

composition(df).to_csv(output_dir / "composition.csv", index=False)
class_distribution(df).to_csv(output_dir / "class_distribution.csv", index=False)
missingness_summary(df).to_csv(output_dir / "missingness.csv", index=False)
metric_ranges(df).to_csv(output_dir / "metric_ranges.csv", index=False)

print(f"[{cfg.name}] profiling tables -> {output_dir}")
print(composition(df).to_string(index=False))
