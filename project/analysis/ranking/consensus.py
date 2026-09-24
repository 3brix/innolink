"""
Consensus between the two design rankers.

The Random Forest (RF) is the primary ranker; the product-composite is complementary.

  - rank each method independently (rank 1 = best; percentile 1.0 = best);
  - merge them into one table ('build_consensus');
  - pick a shortlist ordered by the primary method, flagging which entries the
    complementary method also ranks highly ("consensus" / "primary_only")
  - list the strongest disagreements
  - report an overall rank-agreement (Spearman + Kendall) between the methods.

No combined/blended score is created. Nothing here decides adoption; it reports
where the two methods agree and where they do not, so the choice stays explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_rank(df: pd.DataFrame, score_col: str, prefix: str,
             sample_col: str = "sample", ascending: bool = False) -> pd.DataFrame:
    """
    Return a copy with '<prefix>_score', '<prefix>_rank' (1 = best) and
    '<prefix>_pct' (0..1 percentile, 1 = best) for one method.
    ascending=False means a higher score is better --> the default for p_binder + the product-composite). Rows with a missing score get NaN rank/percentile.
    """
    out = df[[sample_col, score_col]].copy()
    out = out.rename(columns={score_col: f"{prefix}_score"})
    s = out[f"{prefix}_score"]
    # rank: 1 = best. method="min" so ties share the better rank.
    out[f"{prefix}_rank"] = s.rank(ascending=ascending, method="min")
    # percentile: 1.0 = best, 0.0 = worst; NaN scores stay NaN.
    # `not ascending` so the BEST score maps to 1.0 regardless of direction.
    out[f"{prefix}_pct"] = s.rank(ascending=(not ascending), pct=True, method="average")
    return out


def build_consensus(rf_scores: pd.DataFrame, composite_scores: pd.DataFrame,
                    rf_score_col: str = "p_binder",
                    composite_score_col: str = "composite_product",
                    sample_col: str = "sample") -> pd.DataFrame:
    """
    Merge RF and composite scores into one ranked table (one row per design).
    Returns columns: sample, rf_score, rf_rank, rf_pct, comp_score, comp_rank,
    comp_pct, rank_gap (|rf_rank - comp_rank|), pct_gap (rf_pct - comp_pct).
    An outer join keeps designs scored by only one method (the other side NaN).
    """
    rf = add_rank(rf_scores, rf_score_col, "rf", sample_col=sample_col, ascending=False)
    comp = add_rank(composite_scores, composite_score_col, "comp", sample_col=sample_col, ascending=False)

    merged = rf.merge(comp, on=sample_col, how="outer")
    merged["rank_gap"] = (merged["rf_rank"] - merged["comp_rank"]).abs()
    merged["pct_gap"] = merged["rf_pct"] - merged["comp_pct"]
    # order by the PRIMARY method (RF), best first
    return merged.sort_values("rf_rank", na_position="last").reset_index(drop=True)


def shortlist(consensus: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-k designs by the Prf ranking, annotated with agreement."""
    top = consensus.head(k).copy()
    top["consensus"] = np.where(top["comp_rank"] <= k, "consensus", "primary_only")
    return top


def disagreements(consensus: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Designs the two methods most disagree on.
    Returns designs that are in EXACTLY ONE method's top-k, sorted by the size of the rank gap (largest first).
    The 'only_in' column says which method ranked it highly.
    """
    in_rf = consensus["rf_rank"] <= k
    in_comp = consensus["comp_rank"] <= k
    disagree = consensus[in_rf ^ in_comp].copy()          # exclusive-or: exactly one
    disagree["only_in"] = np.where(disagree["rf_rank"] <= k, "rf", "composite")
    return disagree.sort_values("rank_gap", ascending=False, na_position="last").reset_index(drop=True)


def rank_agreement(consensus: pd.DataFrame) -> dict:
    """
    Overall agreement between the two rankings on designs both methods scored.
    Returns {"spearman", "kendall", "n"} over the complete-case overlap.
    Values are None if fewer than 3 designs are ranked by both methods.
    """
    from scipy.stats import spearmanr, kendalltau

    both = consensus.dropna(subset=["rf_rank", "comp_rank"])
    n = len(both)
    if n < 3:
        return {"spearman": None, "kendall": None, "n": n}
    rho, _ = spearmanr(both["rf_rank"], both["comp_rank"])
    tau, _ = kendalltau(both["rf_rank"], both["comp_rank"])
    return {"spearman": round(float(rho), 4), "kendall": round(float(tau), 4), "n": n}
