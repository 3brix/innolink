"""
Regression tests for the metric-metadata invariants that the analysis relies on:

  preprocessing.metric_meta  -> get_metric_columns
  preprocessing.align        -> directions, align_metrics, align_dataframe
  analysis.evaluation        -> load_metric_families, get_family (family lookup)

These are *guards*, not exploration: each asserts a property that must always hold.
Run with:  python -m pytest tests/ -q   (from project_current/)

(The former preprocessing.normalize / preprocessing.diagnostics tests were removed when
the normalize stage was retired and those EDA modules deleted.)
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from preprocessing.metric_meta import get_metric_columns
from preprocessing.align import (
    load_metric_directions,
    get_direction,
    align_metrics,
    align_dataframe,
)
from analysis.evaluation import (
    top_metrics_distinct_family,
    load_metric_families,
    get_family,
)


METRICS = ["af3_iptm", "af3_plddt", "af3_ranking_score", "af3_pae", "af3_energy"]


@pytest.fixture
def metric_yaml(tmp_path):
    """A minimal metric registry, including a shorter/longer suffix clash
    (`score` vs `ranking_score`) to exercise longest-suffix matching."""
    data = {
        "metrics": {
            "iptm": {"direction": 1, "scale": "confidence", "family": "iptm"},
            "plddt": {"direction": 1, "scale": "confidence", "family": "plddt"},
            "score": {"direction": 1, "scale": "ratio", "family": "score"},
            "ranking_score": {"direction": -1, "scale": "ratio", "family": "ranking_score"},
            "pae": {"direction": -1, "scale": "error", "family": "pae"},
            "energy": {"direction": -1, "scale": "energy", "family": "energy"},
        }
    }
    path = tmp_path / "metric_data.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "sample": [f"s{i}" for i in range(n)],
        "source": "x",
        "type": "original",
        "binder": rng.integers(0, 2, n),
        "dataset": "ds1",
        "af3_iptm": rng.uniform(0, 1, n),
        "af3_plddt": rng.uniform(0, 100, n),
        "af3_ranking_score": rng.uniform(0, 5, n),
        "af3_pae": np.abs(rng.normal(0, 3, n)),
        "af3_energy": rng.normal(0, 50, n),
    })


# ---------------------------------------------------------------------------
# metric_meta: metric-column selection
# ---------------------------------------------------------------------------

def test_get_metric_columns_excludes_meta(df):
    cols = set(get_metric_columns(df))
    assert cols == set(METRICS)
    for meta in ["sample", "source", "type", "binder", "dataset"]:
        assert meta not in cols


# ---------------------------------------------------------------------------
# align: direction resolution + alignment
# ---------------------------------------------------------------------------

def test_direction_longest_suffix_wins(metric_yaml):
    directions = load_metric_directions(metric_yaml)
    # 'af3_ranking_score' must match 'ranking_score' (-1), not the shorter 'score' (+1)
    assert get_direction("af3_ranking_score", directions) == -1
    assert get_direction("af3_iptm", directions) == 1
    # unknown column defaults to +1
    assert get_direction("some_unknown_metric", directions) == 1


def test_align_is_involutive(df, metric_yaml):
    directions = load_metric_directions(metric_yaml)
    once = align_metrics(df, METRICS, directions)
    twice = align_metrics(once, METRICS, directions)
    pd.testing.assert_frame_equal(twice[METRICS], df[METRICS])


def test_align_dataframe_flips_by_direction(df, metric_yaml):
    aligned = align_dataframe(df, metric_yaml)
    directions = load_metric_directions(metric_yaml)
    for m in METRICS:
        expected = df[m] * get_direction(m, directions)
        np.testing.assert_allclose(aligned[m].to_numpy(), expected.to_numpy())
    # meta columns untouched
    pd.testing.assert_series_equal(aligned["binder"], df["binder"])


# ---------------------------------------------------------------------------
# family resolution + distinct-family selection
# ---------------------------------------------------------------------------

def test_family_longest_suffix_wins(metric_yaml):
    families = load_metric_families(metric_yaml)
    # 'af3_ranking_score' must match 'ranking_score', not the shorter 'score'
    assert get_family("af3_ranking_score", families) == "ranking_score"
    assert get_family("af3_iptm", families) == "iptm"
    assert get_family("cf_iptm", families) == "iptm"      # sibling -> same family
    assert get_family("some_unknown_metric", families) is None


def test_top_metrics_distinct_family_dedupes_and_sorts(metric_yaml):
    families = load_metric_families(metric_yaml)
    # cf_iptm shares 'iptm' family with the top af3_iptm; deliberately unsorted
    rankings = pd.DataFrame({
        "metric": ["af3_plddt", "af3_iptm", "cf_iptm", "af3_pae"],
        "pr_auc": [0.80, 0.90, 0.85, 0.70],
    })
    picked = top_metrics_distinct_family(rankings, families, top_n=2)
    # best-first: af3_iptm (iptm) then af3_plddt (plddt); cf_iptm skipped (dup family)
    assert picked["metric"].tolist() == ["af3_iptm", "af3_plddt"]


def test_top_metrics_distinct_family_unknown_family_not_merged(metric_yaml):
    families = load_metric_families(metric_yaml)
    rankings = pd.DataFrame({
        "metric": ["mystery_a", "mystery_b"],   # both family None
        "pr_auc": [0.9, 0.8],
    })
    picked = top_metrics_distinct_family(rankings, families, top_n=5)
    # unknown-family metrics fall back to their own name -> both kept
    assert picked["metric"].tolist() == ["mystery_a", "mystery_b"]
