"""Tests for analysis.ranking.consensus (RF-primary consensus + disagreement)."""

import numpy as np
import pandas as pd

from analysis.ranking import (
    add_rank, build_consensus, shortlist, disagreements, rank_agreement,
)


def _rf():
    # higher p_binder = better; d1 best ... d5 worst
    return pd.DataFrame({"sample": ["d1", "d2", "d3", "d4", "d5"],
                         "p_binder": [0.9, 0.8, 0.6, 0.4, 0.1]})


def _comp():
    # composite mostly agrees but ranks d4 highly and d2 low (a disagreement)
    return pd.DataFrame({"sample": ["d1", "d2", "d3", "d4", "d5"],
                         "composite_product": [0.95, 0.20, 0.60, 0.85, 0.05]})


def test_add_rank_best_is_rank1_pct1():
    r = add_rank(_rf(), "p_binder", "rf")
    row = r.set_index("sample")
    assert row.loc["d1", "rf_rank"] == 1
    assert row.loc["d5", "rf_rank"] == 5
    assert row.loc["d1", "rf_pct"] == 1.0          # best -> percentile 1.0
    assert row.loc["d1", "rf_score"] == 0.9


def test_lower_is_better_direction():
    # e.g. an energy/pae-like score where lower is better
    df = pd.DataFrame({"sample": ["a", "b", "c"], "s": [1.0, 2.0, 3.0]})
    r = add_rank(df, "s", "m", ascending=True).set_index("sample")
    assert r.loc["a", "m_rank"] == 1               # smallest is best
    assert r.loc["a", "m_pct"] == 1.0


def test_build_consensus_columns_and_order():
    c = build_consensus(_rf(), _comp())
    for col in ["rf_rank", "comp_rank", "rank_gap", "pct_gap"]:
        assert col in c.columns
    # ordered by RF (primary): first row is d1
    assert c.iloc[0]["sample"] == "d1"
    # rank_gap is |rf_rank - comp_rank|, non-negative
    assert (c["rank_gap"].dropna() >= 0).all()


def test_shortlist_marks_consensus_vs_primary_only():
    c = build_consensus(_rf(), _comp())
    sl = shortlist(c, k=2).set_index("sample")
    # RF top-2 = d1, d2. d1 also composite top-2 -> consensus; d2 not -> primary_only
    assert sl.loc["d1", "consensus"] == "consensus"
    assert sl.loc["d2", "consensus"] == "primary_only"


def test_disagreements_are_exclusive_top_k():
    c = build_consensus(_rf(), _comp())
    d = disagreements(c, k=2)
    only = set(d["sample"])
    # RF top2={d1,d2}, comp top2={d1,d4}; XOR = {d2 (rf only), d4 (composite only)}
    assert only == {"d2", "d4"}
    label = d.set_index("sample")["only_in"].to_dict()
    assert label["d2"] == "rf" and label["d4"] == "composite"


def test_rank_agreement_perfect_and_small_n():
    same = pd.DataFrame({"sample": list("abcde"), "p_binder": [5, 4, 3, 2, 1]})
    c = build_consensus(same, same.rename(columns={"p_binder": "composite_product"}))
    ag = rank_agreement(c)
    assert ag["spearman"] == 1.0 and ag["n"] == 5
    # < 3 overlap -> None
    tiny = build_consensus(same.head(2), same.head(2).rename(columns={"p_binder": "composite_product"}))
    assert rank_agreement(tiny)["spearman"] is None


def test_outer_join_keeps_single_method_designs():
    rf = _rf()
    comp = _comp().head(3)                          # composite missing d4, d5
    c = build_consensus(rf, comp).set_index("sample")
    assert np.isnan(c.loc["d4", "comp_rank"])       # kept, composite side NaN
    assert not np.isnan(c.loc["d4", "rf_rank"])
