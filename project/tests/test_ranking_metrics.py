"""Tests for analysis.evaluation.ranking_metrics (precision@k primitive + precision@10%)."""
import numpy as np
from analysis.evaluation.ranking_metrics import precision_at_k, precision_at_percent, top_k_at_percent


def test_precision_at_k_perfect_and_counts():
    y = [1, 1, 1, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5]          # positives on top
    p, npos, k = precision_at_k(y, s, 3)
    assert (p, npos, k) == (1.0, 3, 3)
    p, npos, k = precision_at_k(y, s, 4)   # top4 = 3 pos + 1 neg
    assert (round(p, 2), npos, k) == (0.75, 3, 4)


def test_precision_at_k_caps_at_n_and_drops_nan():
    y = [1, 0, 1]
    s = [0.9, np.nan, 0.5]
    p, npos, k = precision_at_k(y, s, 10)   # only 2 non-nan -> k_used=2
    assert k == 2 and npos == 2 and p == 1.0


def test_calculate_all_metrics_has_ranking_columns():
    import pandas as pd
    from analysis.evaluation import calculate_all_metrics
    from preprocessing.metric_meta import load_directions
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    df = pd.DataFrame({"binder": y, "af3_iptm": 0.5 + 0.25 * y + rng.normal(0, 0.1, 60)})
    out = calculate_all_metrics(df, ["af3_iptm"], load_directions(), pct=0.10)
    for c in ["prevalence", "pct", "precision_at_pct", "n_pos_at_pct", "k"]:
        assert c in out.columns


def test_precision_at_percent_matches_notebook_definition():
    # top 10% of N = max(1, int(N*0.1)); matches the RF notebook's precision@10%
    assert top_k_at_percent(150, 0.10) == 15
    assert top_k_at_percent(15, 0.10) == 1
    assert top_k_at_percent(3, 0.10) == 1          # floor(0.3)=0 -> min 1
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]             # top 5 (50%) all positive
    s = list(range(10, 0, -1))
    p, npos, k = precision_at_percent(y, s, 0.50)
    assert (p, npos, k) == (1.0, 5, 5)
