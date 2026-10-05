from __future__ import annotations

import pandas as pd

from preprocessing.metric_meta import load_gates


def _fail_mask(series: pd.Series, gate: dict) -> pd.Series:
    """True where the value fails the gate. NaN -> False; missing values are counted in
    'n_gates_seen' rather than read as a pass."""
    if gate["kind"] == "max":
        m = series > gate["value"]
    elif gate["kind"] == "min":
        m = series < gate["value"]
    elif gate["kind"] == "band":
        lo, hi = gate["bounds"]; m = (series < lo) | (series > hi)
    else:
        raise ValueError(f"unknown gate kind: {gate['kind']}")
    return m.fillna(False)


def apply_gates(df: pd.DataFrame, gates: list[dict] | None = None, models: list[str] | None = None) -> pd.DataFrame:
    """Apply the S2 gates per model. Adds, for each model with >=1 present gate column:
    <model>_fail_<metric>, <model>_feasible, <model>_feasible_unknown; plus table-wide
    n_gates_seen and developability_status ('pass' | 'fail' | 'not_assessable').

    `gates` defaults to thresholds.yaml, `models` to config.analysis.MODELS. Absent columns are
    skipped, but a row with no observed gate value is 'not_assessable', never a silent pass.
    """
    if gates is None:
        gates = load_gates()
    if models is None:
        from config.analysis import MODELS
        models = MODELS

    df = df.copy()
    all_fails, seen_counts = [], []
    for model in models:
        masks, na_masks = [], []
        for g in gates:
            if g.get("models") and model not in g["models"]:
                continue
            col = f"{model}_{g['metric']}"
            if col not in df.columns:
                continue
            fail = _fail_mask(df[col], g)
            df[f"{model}_fail_{g['metric']}"] = fail
            masks.append(fail)
            na_masks.append(df[col].isna())
            seen_counts.append(df[col].notna().astype(int))
        if masks:
            df[f"{model}_feasible"] = ~pd.concat(masks, axis=1).any(axis=1)
            df[f"{model}_feasible_unknown"] = pd.concat(na_masks, axis=1).any(axis=1)
            all_fails.extend(masks)

    df["n_gates_seen"] = (pd.concat(seen_counts, axis=1).sum(axis=1) if seen_counts
                          else pd.Series(0, index=df.index))
    any_fail = (pd.concat(all_fails, axis=1).any(axis=1) if all_fails
                else pd.Series(False, index=df.index))
    df["developability_status"] = pd.Series("pass", index=df.index).mask(
        any_fail, "fail").mask(df["n_gates_seen"].eq(0) & ~any_fail, "not_assessable")
    return df


def feasibility_summary(flags: pd.DataFrame, models: list[str], group_col: str | None = "dataset") -> pd.DataFrame:
    """Per (group, model): n_total, n_feasible, frac_feasible and per-gate fail counts."""
    rows = []
    groups = flags.groupby(group_col) if (group_col and group_col in flags.columns) else [("all", flags)]
    for grp, g in groups:
        for model in models:
            feas_col = f"{model}_feasible"
            if feas_col not in g.columns:
                continue
            row = {"group": grp, "model": model, "n_total": len(g),
                   "n_feasible": int(g[feas_col].sum()),
                   "frac_feasible": round(float(g[feas_col].mean()), 4)}
            for c in [c for c in g.columns if c.startswith(f"{model}_fail_")]:
                row[f"nfail_{c[len(model) + 6:]}"] = int(g[c].sum())   # strip "<model>_fail_"
            rows.append(row)
    return pd.DataFrame(rows)