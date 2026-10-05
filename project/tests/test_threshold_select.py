"""Test that the single-metric benchmark emits the reference cutoff but not F1/MCC.

(The former select_threshold tests were removed with that function: the precision-target +
N-floor cutoff existed to pick an operating point for the data-derived S3 confidence gate,
which was retired -- see analysis/feasibility and run_thresholds.py.)
"""

def test_calculate_all_metrics_emits_reference_threshold_not_f1_mcc():
    # F1 and MCC are intentionally dropped from the single-metric ranking benchmark, but the
    # F1-optimal operating point is still emitted as a downstream reference cutoff.
    import pandas as pd, numpy as np
    from analysis.evaluation import calculate_all_metrics
    from preprocessing.metric_meta import load_directions
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    df = pd.DataFrame({"binder": y, "af3_iptm": 0.5 + 0.25 * y + rng.normal(0, 0.1, 60)})
    out = calculate_all_metrics(df, ["af3_iptm"], load_directions())
    assert "f1" not in out.columns and "mcc" not in out.columns
    assert {"opt_threshold_raw", "precision", "recall"} <= set(out.columns)
    assert np.isfinite(float(out.iloc[0]["opt_threshold_raw"]))
