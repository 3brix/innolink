"""
Filter designs by: 
  - quality_filter (analysis.composite)  --> basic confidence QUALITY screen (S1);
  - apply_gates (analysis.feasibility)   --> custom / literature developability, energy, error gates (S2).

  S0  all designs
  S1  + quality screen        (config.analysis.QUALITY_THRESHOLDS)
  S2  + developability gates  (literature values, analysis.feasibility.DEFAULT_GATES)

On the labelled benchmark (eval.csv) each stage reports precision, recall and N-kept.
On the unlabelled designs (design.csv) it reports N-retained plus per-row pass flags.

Outputs (under EVALUATION_DIR/<dataset>/filter/):
  filter_funnel_eval.csv    per-stage precision/recall/N on the benchmark
  filter_funnel_design.csv  per-stage N-retained on the designs
  filtered_designs.csv      designs + per-row pass flags (quality_ok, developability_pass,
                            developability_unknown, passes_filter)
  feasibility_summary.csv   per-model developability gate summary
"""

import logging

import numpy as np
import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from config.analysis import QUALITY_THRESHOLDS, QUALITY_MODE, MODELS
from preprocessing.align import load_metric_directions
from preprocessing.metadata import split_eval_design
from analysis.composite import quality_filter
from analysis.feasibility import DEFAULT_GATES, apply_gates, feasibility_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def developability_pass(df: pd.DataFrame):
    """
    Apply developability gates per model.
    Returns (pass_all, unknown_any, models_used, flagged):
      pass_all    = feasible on every model that has gate columns,
      unknown_any = any of those verdicts rests on a missing value,
      flagged     = the full per-gate flags table.
    """
    flagged = apply_gates(df, DEFAULT_GATES, MODELS)
    feas_cols = [f"{m}_feasible" for m in MODELS if f"{m}_feasible" in flagged.columns]
    unk_cols = [f"{m}_feasible_unknown" for m in MODELS if f"{m}_feasible_unknown" in flagged.columns]
    models_used = [c[:-len("_feasible")] for c in feas_cols]
    if not feas_cols:
        idx = df.index
        return pd.Series(True, index=idx), pd.Series(False, index=idx), [], flagged
    pass_all = flagged[feas_cols].all(axis=1)
    unknown_any = flagged[unk_cols].any(axis=1) if unk_cols else pd.Series(False, index=df.index)
    return pass_all, unknown_any, models_used, flagged


def eval_stage(kept: pd.Series, y_true: pd.Series) -> dict:
    """precision/recall/N for a boolean 'kept' mask against binary labels."""
    kept = kept.fillna(False)
    n_kept = int(kept.sum())
    y = y_true.astype(int)
    n_pos = int((y == 1).sum())
    tp = int(((kept) & (y == 1)).sum())
    return {"n_kept": n_kept,
            "precision": round(tp / n_kept, 4) if n_kept else np.nan,
            "recall": round(tp / n_pos, 4) if n_pos else np.nan}


# load
base = RAW_DATA_DIR / cfg.name
output_dir = EVALUATION_DIR / cfg.name / "filter"
output_dir.mkdir(parents=True, exist_ok=True)

merged = pd.read_csv(base / "merged.csv") if (base / "merged.csv").exists() else None
eval_path, design_path = base / "eval.csv", base / "design.csv"
eval_df = pd.read_csv(eval_path) if eval_path.exists() else (
    split_eval_design(merged)[0] if merged is not None else None)
design_df = pd.read_csv(design_path) if design_path.exists() else (
    split_eval_design(merged)[1] if merged is not None else None)

if eval_df is None:
    raise SystemExit("run_filter: no eval.csv/merged.csv found; run data_prep first.")

directions = load_metric_directions(METRIC_YAML)


def funnel(df: pd.DataFrame, labelled: bool) -> pd.DataFrame:
    """Compute the cumulative funnel S0->S2 (quality, then developability)."""
    q = quality_filter(df, QUALITY_THRESHOLDS, directions, mode=QUALITY_MODE)
    quality_ok = (q["quality_flag"] == "ok")
    dev_ok, _, _, _ = developability_pass(df)

    s0 = pd.Series(True, index=df.index)
    s1 = s0 & quality_ok
    s2 = s1 & dev_ok

    rows = []
    y = df["binder"].astype(int) if labelled else None

    def add(stage, mask):
        rec = {"stage": stage}
        if labelled:
            rec.update(eval_stage(mask, y))
        else:
            rec["n_kept"] = int(mask.fillna(False).sum())
        rows.append(rec)

    add("S0_all", s0)
    add("S1_quality", s1)
    add("S2_developability", s2)
    return pd.DataFrame(rows)


# benchmark funnel (precision/recall/N)
funnel(eval_df, labelled=True).to_csv(output_dir / "filter_funnel_eval.csv", index=False)

# design funnel (N retained) + per-row diagnostics
if design_df is not None and len(design_df):
    funnel(design_df, labelled=False).to_csv(output_dir / "filter_funnel_design.csv", index=False)

    q = quality_filter(design_df, QUALITY_THRESHOLDS, directions, mode=QUALITY_MODE)
    dev_ok, dev_unknown, models_used, dev_flags = developability_pass(design_df)
    diag = design_df.copy()
    diag["quality_ok"] = (q["quality_flag"] == "ok").values
    diag["developability_pass"] = dev_ok.values
    diag["developability_unknown"] = dev_unknown.values
    # passes_filter = the two screens combined.  Consumed by run_consensus for the filter-aware shortlist
    diag["passes_filter"] = diag["quality_ok"] & diag["developability_pass"]
    keep = [c for c in ("sample", "dataset", "binder") if c in diag.columns]
    diag[keep + ["quality_ok", "developability_pass", "developability_unknown", "passes_filter"]] \
        .to_csv(output_dir / "filtered_designs.csv", index=False)

    # per-model developability gate summary
    feasibility_summary(dev_flags, MODELS).to_csv(output_dir / "feasibility_summary.csv", index=False)

print(f"[{cfg.name}] filter report -> {output_dir}")
print("  benchmark funnel (eval):")
print(pd.read_csv(output_dir / "filter_funnel_eval.csv").to_string(index=False))