"""
Validation.

    PYTHONPATH=. DATASET=rf python validate_pipeline.py

It runs two kinds of checks and prints a pass / fail /skip checklist:

  STATIC  (always run, no data needed): imports, runner compilation, path
          resolution, no machine-specific hardcoded paths, feature-selection
          leakage guard, config sanity, determinism of the scoring function.
  DATA    (skipped unless the outputs exist for the selected DATASET): label
          integrity of eval/design, and presence + key columns of each stage's
          outputs. Run the pipeline first (./run_all.sh) to exercise these.

Exit code is the number of hard FAILURES (0 = all good; SKIP does not count).
Nothing here writes to the pipeline's data; it only reads and checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

results = []  # (level, name, status, detail)


def record(level, name, status, detail=""):
    results.append((level, name, status, detail))


def check(level, name):
    """Decorator-ish helper: run fn, catch exceptions as FAIL."""
    def run(fn):
        try:
            ok, detail = fn()
            record(level, name, "PASS" if ok else "FAIL", detail)
        except _Skip as s:
            record(level, name, "SKIP", str(s))
        except Exception as e:
            record(level, name, "FAIL", f"{type(e).__name__}: {e}")
        return fn
    return run


class _Skip(Exception):
    pass


# ------------------------------------------------------------------ STATIC
@check("STATIC", "core modules import")
def _():
    import config.paths, config.datasets, config.analysis, config.provenance  # noqa
    import preprocessing.align, preprocessing.scale, preprocessing.data_prep  # noqa
    import analysis.evaluation, analysis.composite, analysis.feasibility       # noqa
    import analysis.ranking, analysis.distributions                            # noqa
    return True, "all config/preprocessing/analysis packages import"


@check("STATIC", "runner scripts compile")
def _():
    import py_compile
    runners = sorted(Path(".").glob("run_*.py")) + [Path("validate_pipeline.py")]
    for r in runners:
        py_compile.compile(str(r), doraise=True)
    return True, f"{len(runners)} scripts compile: {', '.join(r.stem for r in runners)}"


@check("STATIC", "path resolution + config files present")
def _():
    from config.paths import PROJECT_ROOT, DATA_DIR, RUNS_DIR, METRIC_YAML, THRESHOLDS_YAML
    missing = [str(p) for p in (METRIC_YAML, THRESHOLDS_YAML) if not Path(p).is_file()]
    if missing:
        return False, f"config file(s) not found: {missing}"
    return True, f"PROJECT_ROOT={PROJECT_ROOT}  DATA_DIR={DATA_DIR}  RUNS_DIR={RUNS_DIR}"


@check("STATIC", "no machine-specific hardcoded paths in pipeline code")
def _():
    offenders = []
    for py in Path(".").rglob("*.py"):
        if "grouped_pyros" in py.parts or "__pycache__" in py.parts:
            continue
        # notebooks/ holds EDA artifacts from the refactoring (incl. a legacy config.py with
        # absolute scicore paths). Only notebooks/ranking_rf_final.ipynb is a pipeline stage,
        # and it is a .ipynb, so nothing here is pipeline code.
        if "notebooks" in py.parts:
            continue
        if py.name == "validate_pipeline.py":
            continue  # this validator names 'scicore' only in docs/messages

        for i, line in enumerate(py.read_text(errors="ignore").splitlines(), 1):
            if "scicore" in line:
                # comments are prose, not paths -- only executable references count
                if line.strip().startswith("#"):
                    continue
                # allowed: the EXTERNAL_DATA_ROOT env fallback in datasets.py
                if py.name == "datasets.py" and "getenv" in line:
                    continue
                offenders.append(f"{py}:{i}")
    if offenders:
        return False, "hardcoded scicore paths: " + ", ".join(offenders)
    return True, "only the EXTERNAL_DATA_ROOT fallback references scicore (intended)"


@check("STATIC", "dataset selection resolves")
def _():
    from config.datasets import cfg, SELECTABLE
    return True, f"DATASET='{cfg.name}' resolves (is_set={cfg.is_set}); {len(SELECTABLE)} selectable"


@check("STATIC", "feature selection excludes label/experimental columns (leakage guard)")
def _():
    from analysis.distributions import get_numeric_metrics
    from config.analysis import EXCLUDE_COLUMNS
    df = pd.DataFrame({
        "sample": ["a", "b"], "binder": [0, 1], "dataset": ["x", "x"], "mol_type": ["antibody", "antibody"],
        "binder_class": ["Binder", "Non-Binder"], "iteration": [1, 2],
        "KD[M]": [1e-9, 1e-8], "EC50[M]": [1e-9, 1e-8],
        "af3_iptm": [0.7, 0.4], "af3_sap_score": [0.3, 0.4], "af3_interface_dG": [-5.0, -3.0],
    })
    feats = get_numeric_metrics(df)
    leaked = [c for c in feats if c in EXCLUDE_COLUMNS]
    if leaked:
        return False, f"excluded columns leaked into features: {leaked}"
    filt = [c for c in feats if c in ("af3_sap_score", "af3_interface_dG")]
    if filt:
        return False, f"filtering-only (developability/energy) columns leaked into features: {filt}"
    if "af3_iptm" not in feats:
        return False, "genuine metric column af3_iptm was dropped"
    return True, f"features={feats} (meta/experimental + developability/energy excluded)"


@check("STATIC", "the two metric sets are consistent (full vs predictor)")
def _():
    """The pipeline uses TWO column sets on purpose (see preprocessing.metric_meta):
    the FULL annotated numeric set for descriptive/preselection work, and the PREDICTOR
    subset for anything fitted. Assert the relationship rather than trusting convention."""
    from preprocessing.metric_meta import get_all_metric_columns, get_metric_columns, load_categories, get_category
    from config.analysis import FILTER_ONLY_CATEGORIES
    df = pd.DataFrame({
        "sample": ["a", "b"], "binder": [0, 1], "dataset": ["x", "x"],
        "af3_iptm": [0.7, 0.4], "af3_plddt": [0.9, 0.8],          # interface / confidence -> predictor
        "af3_sap_score": [0.3, 0.4], "af3_interface_dG": [-5.0, -3.0],  # developability / energy
        "af3_mpnn_nll": [1.1, 1.3],                                # sequence
    })
    full, pred = get_all_metric_columns(df), get_metric_columns(df)
    if not set(pred) <= set(full):
        return False, f"predictor set is not a subset of the full set: {sorted(set(pred) - set(full))}"
    cats = load_categories()
    leaked = [c for c in pred if get_category(c, cats) in FILTER_ONLY_CATEGORIES]
    if leaked:
        return False, f"filter-only categories present in the predictor set: {leaked}"
    dropped = sorted(set(full) - set(pred))
    if dropped != ["af3_interface_dG", "af3_mpnn_nll", "af3_sap_score"]:
        return False, f"unexpected difference between the sets: {dropped}"
    return True, (f"full={len(full)} predictor={len(pred)}; "
                  f"{FILTER_ONLY_CATEGORIES} excluded from the predictor set only")


@check("STATIC", "reference values + gate metrics resolve in metric_data.yaml")
def _():
    import yaml
    from config.paths import METRIC_YAML, THRESHOLDS_YAML
    md = yaml.safe_load(open(METRIC_YAML))
    metrics = md["metrics"]
    if md.keys() - {"metrics"}:
        return False, f"metric_data.yaml has unexpected top-level blocks: {sorted(md.keys() - {'metrics'})}"
    # a reference value belongs to a family, so all members of a family must agree on it
    by_family = {}
    for name, info in metrics.items():
        if info and info.get("reference") is not None:
            by_family.setdefault(info.get("family"), {}).setdefault(str(info["reference"]), []).append(name)
    inconsistent = {f: list(v) for f, v in by_family.items() if len(v) > 1}
    if inconsistent:
        return False, f"families with conflicting reference values: {inconsistent}"
    lit = by_family
    # every gate must resolve to a real family, else it is silently skipped at filter time
    from preprocessing.metric_meta import load_gates
    gates = load_gates()
    unresolved = [g["metric"] for g in gates if g.get("family") is None]
    if unresolved:
        return False, f"gate metrics matching no metric_data family: {unresolved}"
    return True, f"{len(lit)} families carry a consistent reference value; {len(gates)} gates all resolve"


@check("STATIC", "scoring function is deterministic")
def _():
    from preprocessing.align import load_metric_directions
    from analysis.evaluation import calculate_all_metrics
    from config.paths import METRIC_YAML
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    df = pd.DataFrame({"binder": y, "af3_iptm": 0.5 + 0.2 * y + rng.normal(0, 0.1, 100)})
    d = load_metric_directions(METRIC_YAML)
    a = calculate_all_metrics(df, ["af3_iptm"], d)
    b = calculate_all_metrics(df, ["af3_iptm"], d)
    return (a.equals(b)), "calculate_all_metrics identical across two calls" if a.equals(b) else "non-deterministic output"


#  DATA
def _eval_root():
    from config.paths import EVALUATION_DIR
    from config.datasets import cfg
    return EVALUATION_DIR / cfg.name


def _base():
    from config.paths import RAW_DATA_DIR
    from config.datasets import cfg
    return RAW_DATA_DIR / cfg.name


@check("DATA", "eval/design label integrity")
def _():
    base = _base()
    ep, dp = base / "eval.csv", base / "design.csv"
    if not ep.exists():
        raise _Skip(f"{ep} not found (run run_data_prep.py)")
    ev = pd.read_csv(ep)
    bad = set(pd.to_numeric(ev["binder"], errors="coerce").dropna().unique()) - {0, 1}
    if bad:
        return False, f"eval.csv binder has non-0/1 values: {bad}"
    detail = f"eval.csv: {len(ev)} rows, binder in {{0,1}}"
    if dp.exists():
        de = pd.read_csv(dp)
        labelled = pd.to_numeric(de["binder"], errors="coerce").isin([0, 1]).sum()
        if labelled:
            return False, f"design.csv has {labelled} labelled rows (should be unlabelled)"
        detail += f"; design.csv: {len(de)} rows, all unlabelled"
    return True, detail


@check("DATA", "benchmark outputs present (rankings.csv w/ pr_auc)")
def _():
    p = _eval_root() / "rankings.csv"
    if not p.exists():
        raise _Skip(f"{p} not found (run run_evaluation.py)")
    cols = set(pd.read_csv(p, nrows=1).columns)
    need = {"metric", "pr_auc", "aligned_roc", "precision_at_pct"}
    return (need <= cols), "rankings.csv columns ok (pr_auc, aligned_roc, precision@pct point estimates)" if need <= cols else f"missing {need - cols}"


@check("DATA", "filter funnel present (designs only)")
def _():
    d = _eval_root() / "filter"
    p = d / "filter_funnel_design.csv"
    if not p.exists():
        raise _Skip(f"{p} not found (run run_filter.py)")
    cols = set(pd.read_csv(p, nrows=1).columns)
    # designs are unlabelled, so the funnel reports N-kept + how many rows S2 could judge
    need = {"stage", "n_kept", "n_assessable", "n_not_assessable"}
    ok = need <= cols
    # feasibility_summary.csv is only produced when design rows exist (it screens designs);
    # a benchmark-only (all-labelled) dataset legitimately has none.
    if ok and (d / "filtered_designs.csv").exists() and not (d / "feasibility_summary.csv").exists():
        return False, "designs present but feasibility_summary.csv missing"
    return ok, "filter funnel ok" if ok else f"missing cols {need - cols}"


@check("DATA", "profiling outputs present")
def _():
    from config.paths import PROCESSED_DATA_DIR
    from config.datasets import cfg
    d = PROCESSED_DATA_DIR / "profiling" / cfg.name
    p = d / "composition.csv"
    if not p.exists():
        raise _Skip(f"{p} not found (run run_profiling.py)")
    need = {"composition.csv", "class_distribution.csv", "missingness.csv", "metric_ranges.csv"}
    have = {f.name for f in d.glob("*.csv")}
    return (need <= have), "profiling tables present" if need <= have else f"missing {need - have}"


@check("DATA", "mol_type persisted in eval")
def _():
    from config.datasets import cfg
    base = _base()
    p = base / "eval.csv"
    if not p.exists():
        raise _Skip(f"{p} not found")
    cols = set(pd.read_csv(p, nrows=1).columns)
    if cfg.mol_type is None and not cfg.is_set:
        raise _Skip("dataset has no mol_type configured")
    return ("mol_type" in cols), "mol_type column present" if "mol_type" in cols else "mol_type column missing"


@check("DATA", "composite outputs present")
def _():
    p = _eval_root() / "composite" / "composite_scores.csv"
    if not p.exists():
        raise _Skip(f"{p} not found (run run_composite.py)")
    cols = set(pd.read_csv(p, nrows=1).columns)
    has_comp = any(c.startswith("composite_") for c in cols)
    return has_comp, "composite_scores has composite_* columns" if has_comp else "no composite_* column"


@check("DATA", "consensus outputs present")
def _():
    p = _eval_root() / "consensus" / "consensus_scores.csv"
    if not p.exists():
        raise _Skip(f"{p} not found (run run_consensus.py)")
    return True, "consensus_scores.csv present"


@check("DATA", "RF design scores present (contract)")
def _():
    p = _eval_root() / "ranking" / "rf_design_scores.csv"
    if not p.exists():
        raise _Skip(f"{p} not found -- RF notebook not wired to emit it yet (expected)")
    cols = set(pd.read_csv(p, nrows=1).columns)
    need = {"sample", "p_binder"}
    return (need <= cols), "rf_design_scores columns ok" if need <= cols else f"missing {need - cols}"


# report
def main():
    width = max(len(n) for _, n, _, _ in results)
    n_fail = 0
    print("=" * (width + 22))
    print("PHASE 6 PIPELINE VALIDATION")
    print("=" * (width + 22))
    for level in ("STATIC", "DATA"):
        print(f"\n[{level}]")
        for lv, name, status, detail in results:
            if lv != level:
                continue
            mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "skip"}[status]
            print(f"  {mark:4}  {name:<{width}}  {detail}")
            if status == "FAIL":
                n_fail += 1
    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_skip = sum(1 for r in results if r[2] == "SKIP")
    print("\n" + "-" * (width + 22))
    print(f"PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    if n_skip:
        print("SKIP = output not present yet; run ./run_all.sh then re-run this to exercise DATA checks.")
    print("-" * (width + 22))
    return n_fail


if __name__ == "__main__":
    sys.exit(main())