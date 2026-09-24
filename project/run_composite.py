import numpy as np
import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from config.analysis import QUALITY_THRESHOLDS, QUALITY_MODE, WINDOW_FAMILIES, PRECISION_AT_PERCENT
from preprocessing.align import load_metric_directions
from analysis.distributions import get_numeric_metrics
from analysis.io import load_eval, rankings_for
from analysis.evaluation import load_metric_families, get_family
from analysis.composite import evaluate_composites, build_composites, quality_filter

base = RAW_DATA_DIR / cfg.name
output_dir = EVALUATION_DIR / cfg.name / "composite"
output_dir.mkdir(parents=True, exist_ok=True)

directions = load_metric_directions(METRIC_YAML)
families = load_metric_families(METRIC_YAML)

# labeled evaluation table
eval_df = load_eval(cfg)

# candidate metrics: numeric, minus window families
candidates = [m for m in get_numeric_metrics(eval_df) if get_family(m, families) not in WINDOW_FAMILIES]

# single-metric separation: the canonical benchmark (run_evaluation's rankings.csv),
# computed once and reused here (no separate single_metric_rankings.csv duplicate).
rankings = rankings_for(cfg, eval_df, candidates, directions)

# honest composite evaluation (out-of-fold CV, stability selection)
y = eval_df["binder"].astype(int).to_numpy()
min_class = int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 0
selected = []
if min_class < 2:
    print(f"[{cfg.name}] too few labeled samples per class (min={min_class}) for CV composite evaluation; skipped.")
else:
    k = min(5, min_class)  # can't have more folds than the smallest class
    cv_table, selected = evaluate_composites(eval_df, y, candidates, directions, families,
                                             k=k, seed=42, selection="stability",
                                             pct=PRECISION_AT_PERCENT)
    cv_table.to_csv(output_dir / "composite_cv_eval.csv", index=False)
    print(f"[{cfg.name}] composite evaluation (lineage-grouped CV, cap {k} folds, {len(selected)} metrics):")
    print(cv_table.to_string(index=False))
    print("selected metrics:", selected)

# scoring: build composite columns to RANK samples
# uses the scaled + aligned table and the stability-selected metrics.
scaled_path = base / "merged_scaled_aligned.csv"
if scaled_path.exists() and selected:
    scaled_aligned = pd.read_csv(scaled_path)
    cols = [c for c in selected if c in scaled_aligned.columns]
    composed = build_composites(scaled_aligned, cols, rankings=rankings)
    keep = [c for c in ("sample", "binder", "binder_type", "dataset") if c in composed.columns]
    composite_cols = [c for c in composed.columns if c.startswith("composite_")]
    composed[keep + composite_cols].to_csv(output_dir / "composite_scores.csv", index=False)
    print("composite score columns:", composite_cols)

# quality filter (screen; raw-unit thresholds, real directions)
flagged = quality_filter(eval_df, QUALITY_THRESHOLDS, directions, mode=QUALITY_MODE)
kept = flagged[flagged["quality_flag"] == "ok"]
flagged[["sample", "quality_flag"]].to_csv(output_dir / "quality_flags.csv", index=False)
print(f"kept {len(kept)} / {len(eval_df)} samples after quality filter")
