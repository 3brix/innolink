from __future__ import annotations

import pandas as pd


def _fail_mask(series: pd.Series, gate: dict) -> pd.Series:
    """True where the value FAILS the gate. NaN -> False (passes)."""
    if gate["kind"] == "max":
        m = series > gate["value"]
    elif gate["kind"] == "min":
        m = series < gate["value"]
    elif gate["kind"] == "band":
        lo, hi = gate["bounds"]; m = (series < lo) | (series > hi)
    else:
        raise ValueError(f"unknown gate kind: {gate['kind']}")
    return m.fillna(False)


def apply_gates(df: pd.DataFrame, gates: list[dict], models: list[str]) -> pd.DataFrame:
    """
    Literature DEVELOPABILITY/ENERGY gates -- one of three distinct "filter" concepts:
    quality_filter (confidence screen), apply_gates (here), and run_filter.py (the design
    FILTER STAGE that combines both with a data-derived confidence funnel).

    Add, per model with >=1 present gate column:
      `<model>_fail_<metric>` (bool) for each applied gate,
      `<model>_feasible`         (bool) = passes ALL its applied gates, and
      `<model>_feasible_unknown` (bool) = the verdict rests on >=1 MISSING gate value.
    Returns a copy of df with those columns added. Models/metrics whose columns are
    absent are skipped (no flag), so nothing crashes on a partial table.

    NaN handling: a missing gate value still counts as a PASS in
    `_fail_mask` (pass/fail semantics unchanged), but `<model>_feasible_unknown` flags
    those rows so a "feasible" verdict built on incomplete data is not mistaken for a
    fully-screened one.
    """
    df = df.copy()
    for model in models:
        masks, na_masks = [], []
        for g in gates:
            col = f"{model}_{g['metric']}"
            if col not in df.columns:
                continue
            fail = _fail_mask(df[col], g)
            df[f"{model}_fail_{g['metric']}"] = fail
            masks.append(fail)
            na_masks.append(df[col].isna())
        if masks:
            df[f"{model}_feasible"] = ~pd.concat(masks, axis=1).any(axis=1)
            df[f"{model}_feasible_unknown"] = pd.concat(na_masks, axis=1).any(axis=1)
    return df


def feasibility_summary(flags: pd.DataFrame, models: list[str], group_col: str | None = "dataset") -> pd.DataFrame:
    """
    Per model (and per `group_col` if present): n_total, n_feasible, frac_feasible,
    and the per-gate fail counts. Long format, one row per (group, model).
    """
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