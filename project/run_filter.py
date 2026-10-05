"""
Feasibility filter -- DESIGN SET ONLY.

  S0  all designs
  S1  + quality screen        (config/thresholds.yaml 'quality:' block)
  S2  + developability gates  (config/thresholds.yaml 'gates:' + 'thresholds:' blocks)

All filter configuration lives in config/thresholds.yaml; nothing is hardcoded here.

Why designs only: the labelled benchmark pools antibody and nanobody complexes, whose
interfaces differ in size, so the PyRosetta interface/developability metrics are not
comparable across it -- a precision/recall funnel computed there would compare unlike
things. The design sets are all nanobodies, so the gates are applied there and nowhere
else. Filtering stays a feasibility screen; it is never used to select or rank binders.

Outputs (under EVALUATION_DIR/<dataset>/filter/):
  filter_funnel_design.csv  per-stage N-retained, plus how many rows S2 could judge
  filtered_designs.csv      per-design flags (quality_ok, developability_status,
                            developability_pass, n_gates_seen, passes_filter)
  feasibility_summary.csv   per-model developability gate summary
"""

import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, EVALUATION_DIR, METRIC_YAML
from config.analysis import MODELS
from preprocessing.align import load_metric_directions
from preprocessing.metadata import split_eval_design
from preprocessing.metric_meta import load_quality, load_gates
from analysis.composite import quality_filter
from analysis.feasibility import apply_gates, feasibility_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def developability_pass(df: pd.DataFrame):
    """
    Apply the S2 feasibility gates (config/thresholds.yaml) per model.
    Returns (not_failed, status, models_used, flagged):
      not_failed  = the row fails no observed gate (True for 'pass' AND 'not_assessable'),
      status      = 'pass' / 'fail' / 'not_assessable' per row,
      flagged     = the full per-gate flags table.

    'not_assessable' means NO gate value was observed for the row, so its feasibility is
    unknown rather than confirmed. Those rows are not excluded here (they are reported
    separately) -- see filter_funnel_design.csv, which counts all three states.
    """
    flagged = apply_gates(df, load_gates(), MODELS)
    feas_cols = [f"{m}_feasible" for m in MODELS if f"{m}_feasible" in flagged.columns]
    models_used = [c[:-len("_feasible")] for c in feas_cols]
    status = flagged["developability_status"]
    return status.ne("fail"), status, models_used, flagged


# load: designs only
base = RAW_DATA_DIR / cfg.name
output_dir = EVALUATION_DIR / cfg.name / "filter"
output_dir.mkdir(parents=True, exist_ok=True)

design_path = base / "design.csv"
if design_path.exists():
    design_df = pd.read_csv(design_path)
elif (base / "merged.csv").exists():
    design_df = split_eval_design(pd.read_csv(base / "merged.csv"))[1]
else:
    raise SystemExit("run_filter: no design.csv/merged.csv found; run data_prep first.")

if not len(design_df):
    raise SystemExit(f"run_filter: no design rows for '{cfg.name}' -- nothing to filter.")

directions = load_metric_directions(METRIC_YAML)
quality_cutoffs, quality_mode = load_quality()
gates = load_gates()
logger.info("Filtering %d designs", len(design_df))
logger.info("S1 quality screen: %s (mode=%s)", quality_cutoffs, quality_mode)
logger.info("S2 gates: %s", [f"{g['metric']} {g['kind']} {g.get('value', g.get('bounds'))}" for g in gates])


def funnel(df: pd.DataFrame, quality_ok: pd.Series, dev_status: pd.Series) -> pd.DataFrame:
    """Cumulative funnel S0->S2. n_assessable records how many rows S2 could actually judge."""
    s0 = pd.Series(True, index=df.index)
    s1 = s0 & quality_ok
    s2 = s1 & dev_status.ne("fail")
    rows = [{"stage": "S0_all", "n_kept": int(s0.sum())},
            {"stage": "S1_quality", "n_kept": int(s1.sum())},
            {"stage": "S2_developability", "n_kept": int(s2.sum()),
             "n_assessable": int((dev_status[s1] != "not_assessable").sum()),
             "n_not_assessable": int((dev_status[s1] == "not_assessable").sum()),
             "n_failed_gate": int((dev_status[s1] == "fail").sum())}]
    return pd.DataFrame(rows)


q = quality_filter(design_df, quality_cutoffs, directions, mode=quality_mode)
quality_ok = (q["quality_flag"] == "ok")
dev_ok, dev_status, models_used, dev_flags = developability_pass(design_df)

funnel(design_df, quality_ok, dev_status).to_csv(output_dir / "filter_funnel_design.csv", index=False)

diag = design_df.copy()
diag["quality_ok"] = quality_ok.values
diag["developability_status"] = dev_status.values        # pass / fail / not_assessable
diag["developability_pass"] = dev_status.eq("pass").values
diag["n_gates_seen"] = dev_flags["n_gates_seen"].values
# not excluded by either screen; 'not_assessable' passes (unknown, not bad)
diag["passes_filter"] = diag["quality_ok"] & dev_ok.values
keep = [c for c in ("sample", "dataset", "binder") if c in diag.columns]
diag[keep + ["quality_ok", "developability_status", "developability_pass",
             "n_gates_seen", "passes_filter"]].to_csv(output_dir / "filtered_designs.csv", index=False)

# per-model developability gate summary
feasibility_summary(dev_flags, MODELS).to_csv(output_dir / "feasibility_summary.csv", index=False)

print(f"[{cfg.name}] filter report -> {output_dir}")
print(f"  models with gate columns: {models_used or '(none)'}")
print("  developability status:", dev_status.value_counts().to_dict())
print("  design funnel:")
print(pd.read_csv(output_dir / "filter_funnel_design.csv").to_string(index=False))
