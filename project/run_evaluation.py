import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from preprocessing.align import load_metric_directions, align_dataframe
from preprocessing.metadata import split_eval_design
from analysis.distributions import get_numeric_metrics
from analysis.evaluation import (
    calculate_all_metrics,
    compute_auroc_pvalue,
    cliffs_delta,
    cohens_d,
    spearman_correlation,
    load_metric_families,
    get_family,
)
from config.analysis import WINDOW_FAMILIES


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



base = RAW_DATA_DIR / cfg.name

output_dir = EVALUATION_DIR / cfg.name
output_dir.mkdir(parents=True, exist_ok=True)

eval_path = base / "eval.csv"
if eval_path.exists():
    df = pd.read_csv(eval_path)
    logger.info("Evaluation on %s (%d labeled rows)", eval_path.name, len(df))
else:
    df, _ = split_eval_design(pd.read_csv(base / "merged.csv"))
    logger.info("eval.csv absent; derived eval split from merged.csv (%d labeled rows)", len(df))

metrics = get_numeric_metrics(df)
directions = load_metric_directions(METRIC_YAML)

# drop unbound (window) metric families from monotonic evaluation --> higher/lower-is-better direction is meaningless for them.
# screened as windows in feasibility instead.
_families = load_metric_families(METRIC_YAML)
_window_cols = [m for m in metrics if get_family(m, _families) in WINDOW_FAMILIES]
if _window_cols:
    metrics = [m for m in metrics if m not in _window_cols]
    logger.info("Excluded %d window-metric column(s) from monotonic evaluation: %s",
                len(_window_cols), _window_cols)

# pooled / single-dataset ranking + effect sizes + redundancy
calculate_all_metrics(df, metrics, directions).to_csv(output_dir / "rankings.csv", index=False)
compute_auroc_pvalue(df, metrics, directions).to_csv(output_dir / "auroc_pvalue.csv", index=False)
cliffs_delta(df, metrics, directions).to_csv(output_dir / "cliffs_delta.csv", index=False)
cohens_d(df, metrics, directions).to_csv(output_dir / "cohens_d.csv", index=False)


aligned_path = base / "eval_aligned.csv"
aligned = pd.read_csv(aligned_path) if aligned_path.exists() else align_dataframe(df, METRIC_YAML)
spearman_correlation(aligned).to_csv(output_dir / "spearman_correlation.csv")          # window metric families included

# per-dataset breakdown (pooled sets)
if cfg.is_set and "dataset" in df.columns:
    rankings_by, cliffs_by, cohens_by = [], [], []
    for ds, group in df.groupby("dataset"):
        if group["binder"].nunique() < 2:
            logger.warning("skipping %s: single binder class (%d rows)", ds, len(group))
            continue
        r = calculate_all_metrics(group, metrics, directions); r.insert(0, "dataset", ds); rankings_by.append(r)
        c = cliffs_delta(group, metrics, directions);          c.insert(0, "dataset", ds); cliffs_by.append(c)
        d = cohens_d(group, metrics, directions);              d.insert(0, "dataset", ds); cohens_by.append(d)
    pd.concat(rankings_by, ignore_index=True).to_csv(output_dir / "rankings_by_dataset.csv", index=False)
    pd.concat(cliffs_by, ignore_index=True).to_csv(output_dir / "cliffs_delta_by_dataset.csv", index=False)
    pd.concat(cohens_by, ignore_index=True).to_csv(output_dir / "cohens_d_by_dataset.csv", index=False)
    print(f"Set '{cfg.name}': wrote pooled + per-dataset breakdown ({df['dataset'].nunique()} datasets)")