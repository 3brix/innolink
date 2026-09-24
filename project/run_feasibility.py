import logging

import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, METRIC_YAML, PROCESSED_DATA_DIR
from config.analysis import MODELS, META_COLS
from preprocessing.align import load_metric_directions
from preprocessing.metadata import split_eval_design
from analysis.distributions import get_numeric_metrics
from analysis.feasibility.feasibility import run_feasibility


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base = RAW_DATA_DIR / cfg.name
output_dir = PROCESSED_DATA_DIR / "feasibility" / cfg.name
output_dir.mkdir(parents=True, exist_ok=True)

eval_path = base / "eval.csv"
if eval_path.exists():
    df = pd.read_csv(eval_path)
    logger.info("Feasibility on %s (%d labeled rows)", eval_path.name, len(df))
else:
    df, _ = split_eval_design(pd.read_csv(base / "merged.csv"))
    logger.info("eval.csv absent; derived eval split from merged.csv (%d labeled rows)", len(df))

metrics = get_numeric_metrics(df)
directions = load_metric_directions(METRIC_YAML)

res = run_feasibility(df, metrics, directions, MODELS)

# trim the (wide) flags table to meta + the new gate columns for a readable CSV
meta = [c for c in META_COLS if c in res["flags"].columns]
gate_cols = [c for c in res["flags"].columns if "_feasible" in c or "_fail_" in c]
res["flags"][meta + gate_cols].to_csv(output_dir / "feasibility_flags.csv", index=False)
res["responsiveness"].to_csv(output_dir / "feasibility_responsiveness.csv", index=False)
res["consistency"].to_csv(output_dir / "feasibility_consistency.csv", index=False)
res["summary"].to_csv(output_dir / "feasibility_summary.csv", index=False)

print(f"[{cfg.name}] feasibility written to {output_dir}")
print(res["summary"].to_string(index=False) if not res["summary"].empty else "  (no gate columns present in this table)")