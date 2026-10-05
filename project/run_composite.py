import numpy as np
import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from config.analysis import PRECISION_AT_PERCENT
from preprocessing.metric_meta import load_window_families
from preprocessing.align import load_metric_directions
from preprocessing.metric_meta import load_quality
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
candidates = [m for m in get_numeric_metrics(eval_df) if get_family(m, families) not in load_window_families()]

# metrics with missing values are scored on their own observed rows, so their PR-AUC
# is not computed on the same samples as a fully-observed metric's. Report how many.
_incomplete = {m: round(float(eval_df[m].notna().mean()), 3) for m in candidates if eval_df[m].isna().any()}
print(f"[{cfg.name}] {len(candidates)} candidate metrics, {len(_incomplete)} with missing values")
if _incomplete:
    print("  incomplete candidates (coverage):", dict(sorted(_incomplete.items(), key=lambda kv: kv[1])))

# single-metric separation: the canonical benchmark (run_evaluation's rankings.csv),
# computed once and reused here (no separate single_metric_rankings.csv duplicate).
rankings = rankings_for(cfg, eval_df, candidates, directions)

# dataset-aware selection + lineage-grouped CV as a ROBUSTNESS analysis: out-of-fold given the
# metric set, which was chosen on all labelled rows, so not independent validation.
y = eval_df["binder"].astype(int).to_numpy()
min_class = int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 0
selected = []
if min_class < 2:
    print(f"[{cfg.name}] too few labeled samples per class (min={min_class}) for CV composite evaluation; skipped.")
else:
    k = min(5, min_class)  # can't have more folds than the smallest class
    cv_table, selected, by_fold, oof_scores, votes = evaluate_composites(eval_df, y, candidates, directions, families,
                                                             k=k, seed=42, selection="dataset",
                                                             pct=PRECISION_AT_PERCENT, return_votes=True)
    cv_table.to_csv(output_dir / "composite_cv_eval.csv", index=False)
    # WHICH metrics built the composite and how narrowly each was chosen -- nothing else on disk
    # records the metric set.
    votes.to_csv(output_dir / "selected_metrics.csv", index=False)
    # grouped folds are near leave-one-source-out, so the spread across
    # folds is the cross-dataset transfer behaviour, not sampling noise.
    if len(by_fold):
        by_fold.to_csv(output_dir / "cv_by_fold.csv", index=False)
        # so the PR / ROC curves come from the reported predictions
        if len(oof_scores):
            oof_scores.to_csv(output_dir / "composite_oof_scores.csv", index=False)
        pd.set_option("display.width", 200)
        print(f"[{cfg.name}] per-fold detail -> {output_dir / 'cv_by_fold.csv'}")
    print(f"[{cfg.name}] composite evaluation (lineage-grouped CV, cap {k} folds, {len(selected)} metrics):")
    print(cv_table.to_string(index=False))
    print("selected metrics:", selected)
    # flag the fragile picks: a family won by a hair (another model's version would do as well),
    # and any chosen metric that falls below no-skill on at least one source.
    _chosen = votes[votes["chosen"]] if "chosen" in votes.columns else votes.iloc[:0]
    narrow = _chosen[_chosen["margin"] < 0.05]["metric"].tolist()
    below = _chosen[_chosen["ap_norm_ds_min"] < 0]["metric"].tolist()
    print(f"selection table -> {output_dir / 'selected_metrics.csv'}"
          + (f" (narrow within-family wins: {narrow})" if narrow else "")
          + (f" (below no-skill on some dataset: {below})" if below else ""))

# scoring: build composite columns to RANK samples
# uses the scaled + aligned table and the stability-selected metrics.
scaled_path = base / "merged_scaled_aligned.csv"
if scaled_path.exists() and selected:
    scaled_aligned = pd.read_csv(scaled_path)
    cols = [c for c in selected if c in scaled_aligned.columns]
    # standardised against the LABELLED rows only, so a design's score does not depend on which
    # other designs share the file (CLAUDE.md, scaler default)
    _labelled = scaled_aligned[scaled_aligned["binder"].astype(str) != "?"]
    composed = build_composites(scaled_aligned, cols, rankings=rankings,
                                reference=_labelled if len(_labelled) else None)
    print(f"product reference: {len(_labelled)} labelled rows (designs scored against the benchmark)")
    keep = [c for c in ("sample", "binder", "binder_class", "dataset") if c in composed.columns]
    composite_cols = [c for c in composed.columns if c.startswith("composite_")]
    composed[keep + composite_cols].to_csv(output_dir / "composite_scores.csv", index=False)
    print("composite score columns:", composite_cols)

# quality filter (screen; raw-unit thresholds, real directions)
_quality_cutoffs, _quality_mode = load_quality()
flagged = quality_filter(eval_df, _quality_cutoffs, directions, mode=_quality_mode)
kept = flagged[flagged["quality_flag"] == "ok"]
flagged[["sample", "quality_flag"]].to_csv(output_dir / "quality_flags.csv", index=False)
print(f"kept {len(kept)} / {len(eval_df)} samples after quality filter")
