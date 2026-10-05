"""Tests for analysis.distributions.summaries (profiling tables)."""
import pandas as pd
from analysis.distributions.summaries import composition, class_distribution, missingness_summary


def _df():
    return pd.DataFrame({
        "sample": [f"s{i}" for i in range(6)],
        "binder": [1, 1, 0, 0, "?", "?"],
        "type": ["original"] * 6,
        "dataset": ["a", "a", "a", "b", "b", "b"],
        "af3_iptm": [0.9, None, 0.4, 0.5, 0.6, 0.7],
    })


def test_composition_prevalence_and_design_counts():
    c = composition(_df()).set_index("dataset")
    assert c.loc["a", "n_binder"] == 2 and c.loc["a", "n_nonbinder"] == 1
    assert c.loc["a", "prevalence"] == round(2 / 3, 4)
    assert c.loc["b", "n_design"] == 2


def test_class_distribution_derives_binder_class():
    cd = class_distribution(_df())
    assert "binder_class" in cd.columns
    # 2 binders, 1 non-binder, 2 designs, 1 non-binder... labelled types present
    assert cd["n"].sum() == 6


def test_missingness_overall_and_per_dataset():
    m = missingness_summary(_df()).set_index("metric")
    assert abs(m.loc["af3_iptm", "missing_frac_overall"] - round(1 / 6, 4)) < 1e-9
    assert any(col.startswith("missing_dataset_") for col in m.columns)
