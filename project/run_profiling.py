"""
Describes the data before benchmarking: class distribution, dataset/molecule-type
composition and positive-class prevalence, per-metric missingness, and metric ranges.

Outputs (under PROCESSED_DATA_DIR/profiling/<dataset>/):
  composition.csv        per dataset[/mol_type]: counts + prevalence
  class_distribution.csv binder_class counts overall + per dataset[/mol_type]
  missingness.csv        per-metric missing fraction (overall + per dataset), ALL rows
  missingness_eval.csv   same, labelled benchmark rows only (feeds report_table.py)
  metric_ranges.csv      per-metric min/max/mean/std/median + missing
  lineage_groups.csv     per parent lineage: size + prevalence (the CV grouping unit)
  cv_folds.csv           per grouped-CV fold: size, prevalence, #groups, dataset composition
"""

import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_directory
from analysis.distributions.summaries import (
    class_distribution, composition, missingness_summary, metric_ranges, benchmark_structure,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base = RAW_DATA_DIR / cfg.name
output_dir = ensure_directory(PROCESSED_DATA_DIR / "profiling" / cfg.name)

# Profile the full merged table (all rows incl. designs)
df = pd.read_csv(base / "merged.csv")
logger.info("Profiling %s (%d rows, %d columns)", cfg.name, len(df), df.shape[1])

composition(df).to_csv(output_dir / "composition.csv", index=False)
class_distribution(df).to_csv(output_dir / "class_distribution.csv", index=False)
missingness_summary(df).to_csv(output_dir / "missingness.csv", index=False)

# Benchmark-only missingness for Results 3.1; missingness.csv keeps all rows because design
# coverage is part of deciding what to exclude.
if (base / "eval.csv").exists():
    ev = pd.read_csv(base / "eval.csv")
    if len(ev):  # design-only datasets have an empty eval.csv -- a 0-row table would report 0% for everything
        missingness_summary(ev).to_csv(output_dir / "missingness_eval.csv", index=False)

metric_ranges(df).to_csv(output_dir / "metric_ranges.csv", index=False)

# Lineage structure: the effective sample size downstream is the number of LINEAGES, not rows.
_groups, _folds = benchmark_structure(df)
if len(_groups):
    _groups.to_csv(output_dir / "lineage_groups.csv", index=False)
    _folds.to_csv(output_dir / "cv_folds.csv", index=False)
    logger.info("%d lineage groups for %d labelled samples, %d CV folds",
                len(_groups), int(_groups["n"].sum()), len(_folds))

print(f"[{cfg.name}] profiling tables -> {output_dir}")
print(composition(df).to_string(index=False))
