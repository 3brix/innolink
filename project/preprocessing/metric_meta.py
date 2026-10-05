"""Canonical metric metadata: the only reader of metric_data.yaml and thresholds.yaml.

Columns match their metadata by exact name or longest suffix.

TWO METRIC SETS -- every stage must pick one on purpose:
  get_all_metric_columns(df)  all annotated numeric metrics (~140 for 'rf'). DESCRIPTIVE work
                              (QC, profiling, redundancy, benchmark, align, scale), which is the
                              evidence for what to exclude, so filtering it first is circular.
  get_metric_columns(df)      the PREDICTOR subset (~46), minus FILTER_ONLY_CATEGORIES. For
                              anything FITTED: composite candidates, RF features.
The predictor set is a subset of the full set (asserted in validate_pipeline.py). Do not select
numeric columns any other way. See PIPELINE_SPEC.md for the per-stage contracts.
"""

from __future__ import annotations

import logging
import yaml
import pandas as pd

from config.analysis import EXCLUDE_COLUMNS, FILTER_ONLY_CATEGORIES
from config.paths import METRIC_YAML, THRESHOLDS_YAML

logger = logging.getLogger(__name__)


# Feature-column selection
def get_metric_columns(df: pd.DataFrame, exclude_categories=FILTER_ONLY_CATEGORIES) -> list[str]:
    """The predictor set: numeric columns minus meta/label and the filter-only categories.

    A column with no metric_data.yaml entry is warned about and KEPT -- it would otherwise enter
    the model as a predictor at direction +1. Add it to the YAML."""
    numeric = df.select_dtypes(include="number").columns
    cols = [c for c in numeric if c not in EXCLUDE_COLUMNS]
    cats = load_categories()
    unknown = [c for c in cols if get_category(c, cats) is None]
    if unknown:
        logger.warning("%d numeric column(s) have no metric_data.yaml entry (direction defaults to +1, "
                       "category unknown, so they are NOT filtered out): %s", len(unknown), unknown)
    if exclude_categories:
        cols = [c for c in cols if get_category(c, cats) not in exclude_categories]
    return cols


def get_all_metric_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric metric, filter-only categories included (the descriptive set)."""
    return get_metric_columns(df, exclude_categories=None)

def _load_metric_field(yaml_path, field: str) -> dict:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    raw = {name: info.get(field) for name, info in data["metrics"].items()}
    return dict(sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True))


def load_directions(yaml_path=METRIC_YAML) -> dict[str, int]:
    """{metric: +1/-1} directionality."""
    return _load_metric_field(yaml_path, "direction")


def load_families(yaml_path=METRIC_YAML) -> dict[str, str]:
    """{metric: family}."""
    return _load_metric_field(yaml_path, "family")


def load_categories(yaml_path=METRIC_YAML) -> dict[str, str]:
    """{metric: category} (confidence / interface / sequence / developability / energy / ...)."""
    return _load_metric_field(yaml_path, "category")


def load_scales(yaml_path=METRIC_YAML) -> dict[str, str]:
    """{metric: scale} (confidence / error / energy / developability / ...)."""
    return _load_metric_field(yaml_path, "scale")


def load_reference_values(yaml_path=METRIC_YAML) -> dict:
    """{metric: reference value} -- the conventional operating point, metadata not filter config.

    Called 'reference', not 'literature': these are conventional defaults and only the main
    families carry a citation. Read by run_thresholds.py, and a [lo, hi] pair defines window-ness.
    What the filter actually gates on is thresholds.yaml 'gates:' via load_gates()."""
    return {k: v for k, v in _load_metric_field(yaml_path, "reference").items() if v is not None}


def load_window_families(yaml_path=METRIC_YAML) -> set[str]:
    """Families that are TWO-SIDED: their 'reference' is a [lo, hi] pair.

    Derived, not hand-listed -- the old list had three families wrong. Window families are
    excluded from the monotone benchmark and threshold report, where "higher = better" has no
    meaning; the gates screen them as bands."""
    families = load_families(yaml_path)
    return {families[m] for m, v in load_reference_values(yaml_path).items()
            if isinstance(v, (list, tuple)) and families.get(m)}


def load_quality(path=THRESHOLDS_YAML) -> tuple[dict[str, float], str]:
    """S1 confidence screen from thresholds.yaml: ({column: cutoff}, mode)."""
    q = (yaml.safe_load(open(path)) or {}).get("quality") or {}
    cols = {k: v for k, v in (q.get("columns") or {}).items() if v is not None}
    return cols, q.get("mode", "any")


def load_gates(path=THRESHOLDS_YAML) -> list[dict]:
    """S2 feasibility gates from thresholds.yaml.

    Each entry {metric, value|bounds, models?} carries its own number. The fail rule follows the
    column's direction and the value's shape: [lo, hi] -> band, scalar +1 -> min, scalar -1 -> max.
    'models' restricts a gate to named models. Entries with no value are skipped."""
    entries = (yaml.safe_load(open(path)) or {}).get("gates") or []
    families, directions = load_families(), load_directions()
    gates = []
    for entry in entries:
        suffix = entry["metric"]
        value = entry.get("bounds", entry.get("value"))
        models = entry.get("models")
        family = _match_suffix(f"af3_{suffix}", families) or _match_suffix(suffix, families)
        if value is None:
            logger.warning("gate %r skipped: no 'value' or 'bounds' given", suffix)
            continue
        gate = {"metric": suffix, "family": family}
        if models:
            gate["models"] = list(models)
        if isinstance(value, (list, tuple)):
            gate.update(kind="band", bounds=tuple(value))
        else:
            direction = _match_suffix(f"af3_{suffix}", directions) or _match_suffix(suffix, directions) or 1
            gate.update(kind="min" if direction >= 0 else "max", value=value)
        gates.append(gate)
    return gates


def _match_suffix(col: str, mapping: dict, default=None):
    for base, val in mapping.items():
        if col == base or col.endswith(f"_{base}"):
            return val
    return default


def get_direction(col: str, directions: dict[str, int]) -> int:
    """Return +1 / -1 for a column; default +1."""
    return _match_suffix(col, directions, default=1)


def get_family(col: str, families: dict[str, str]) -> str | None:
    """Return a column's family via exact / longest-suffix match; None if unknown."""
    return _match_suffix(col, families, default=None)


def get_reference_value(col: str, reference: dict):
    """A column's reference value by exact / longest-suffix match; None if it has none."""
    return _match_suffix(col, reference, default=None)


def get_category(col: str, categories: dict[str, str]) -> str | None:
    """Return a column's category via exact / longest-suffix match; None if unknown."""
    return _match_suffix(col, categories, default=None)