from __future__ import annotations

import pandas as pd

from analysis.feasibility.gates import DEFAULT_GATES
from analysis.feasibility.filters import apply_gates, feasibility_summary
from analysis.feasibility.responsiveness import responsiveness, consistency


def run_feasibility(df: pd.DataFrame, metrics: list[str], directions: dict, models: list[str],
                    gates: list[dict] = DEFAULT_GATES, group_col: str = "dataset") -> dict:
    flags = apply_gates(df, gates, models)
    resp = responsiveness(df, metrics, directions, group_col)
    return {"flags": flags,
            "responsiveness": resp,
            "consistency": consistency(resp, group_col),
            "summary": feasibility_summary(flags, models, group_col)}