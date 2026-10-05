"""Consensus ranking of designs: Random Forest (primary) + product composite.

Filtering never changes the ranking; it only selects from it at the last step. Three artefacts,
in increasing restriction:

  ranking/rf_design_scores.csv    the RF ranking of every design, unfiltered
  consensus/consensus_scores.csv  the complete ranking + feasibility flags ATTACHED, not applied
  consensus/shortlist.csv         top-k RF among passes_filter  <- the final prioritisation

'passes_filter' can rest on different gate sets per design set (developability metrics exist only
where PyRosetta ran); n_gates_seen records how many gates each design was judged on.

Config: SHORTLIST_K (default 25), COMPOSITE_SCORE (default composite_product).

Outputs (EVALUATION_DIR/<dataset>/consensus/):
  consensus_scores.csv   every design with rf/comp rank, percentile, rank_gap + filter flags
  shortlist.csv          top-k RF among filter-passing designs, flagged consensus / primary_only
  disagreements.csv      designs in exactly one method's top-k
  agreement.txt          Spearman / Kendall rank agreement
"""

import os
import logging

import pandas as pd

from config.datasets import cfg
from config.paths import EVALUATION_DIR
from analysis.ranking import build_consensus, shortlist, disagreements, rank_agreement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHORTLIST_K = int(os.getenv("SHORTLIST_K", "25"))
COMPOSITE_SCORE = os.getenv("COMPOSITE_SCORE", "composite_product")

eval_root = EVALUATION_DIR / cfg.name
rf_path = eval_root / "ranking" / "rf_design_scores.csv"
comp_path = eval_root / "composite" / "composite_scores.csv"
output_dir = eval_root / "consensus"
output_dir.mkdir(parents=True, exist_ok=True)


def design_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose binder label is not 0/1; all rows if there is no binder column."""
    if "binder" not in df.columns:
        return df
    is_labelled = pd.to_numeric(df["binder"], errors="coerce").isin([0, 1])
    return df[~is_labelled].copy()


def pick_composite_col(df: pd.DataFrame) -> str | None:
    if COMPOSITE_SCORE in df.columns:
        return COMPOSITE_SCORE
    alts = [c for c in df.columns if c.startswith("composite_")]
    return alts[0] if alts else None


def load_filter_flags() -> pd.DataFrame | None:
    """Per-design feasibility flags from run_filter, or None if that stage has not run.

    Attached to consensus_scores.csv for information; write_shortlist is what applies them."""
    fp = eval_root / "filter" / "filtered_designs.csv"
    if not fp.exists():
        return None
    f = pd.read_csv(fp)
    if "sample" not in f.columns or "passes_filter" not in f.columns:
        return None
    # strings -> boolean
    f["passes_filter"] = f["passes_filter"].astype(str).str.strip().str.lower().eq("true")
    cols = [c for c in ("sample", "passes_filter", "developability_status", "n_gates_seen") if c in f.columns]
    return f[cols]


def attach_filter(ranked: pd.DataFrame, flags: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join filter flags onto a ranked design table."""
    if flags is None:
        out = ranked.copy()
        out["passes_filter"] = pd.NA
        return out
    return ranked.merge(flags, on="sample", how="left")


def write_shortlist(ranked: pd.DataFrame, flags: pd.DataFrame | None, use_consensus_fn: bool) -> None:
    """Write the final shortlist: top-k by the primary ranker among filter-passing designs."""
    if flags is not None:
        feasible = ranked[ranked["passes_filter"].eq(True)]
        notes.append(f"shortlist restricted to filter-passing designs "
                     f"({len(feasible)} of {len(ranked)} pass).")
        if "developability_status" in ranked.columns:
            notes.append(f"feasibility verdicts over all designs: "
                         f"{ranked['developability_status'].value_counts().to_dict()}")
    else:
        feasible = ranked
        notes.append("no feasibility report found; shortlist is the unfiltered top-k.")
    sl = shortlist(feasible, SHORTLIST_K) if use_consensus_fn else feasible.head(SHORTLIST_K)
    sl.to_csv(output_dir / "shortlist.csv", index=False)
    notes.append(f"shortlist = top {len(sl)} of those, ranked by the primary method "
                 f"(complete unfiltered ranking kept in consensus_scores.csv).")


notes = []

# composite design scores required
if not comp_path.exists():
    raise SystemExit(f"run_consensus: composite scores not found at {comp_path}; run run_composite.py first.")
comp_all = pd.read_csv(comp_path)
comp_designs = design_rows(comp_all)
comp_col = pick_composite_col(comp_designs)
if comp_col is None:
    raise SystemExit(f"run_consensus: no composite_* column in {comp_path}.")
logger.info("composite design scores: %d rows, using column '%s'", len(comp_designs), comp_col)

flags = load_filter_flags()
if flags is None:
    notes.append("feasibility report not found (run run_filter.py first); shortlist is NOT filter-aware.")

# RF design scores 
if rf_path.exists():
    rf_designs = pd.read_csv(rf_path)
    if "p_binder" not in rf_designs.columns:
        raise SystemExit(f"run_consensus: {rf_path} lacks required 'p_binder' column.")
    logger.info("RF design scores: %d rows", len(rf_designs))

    consensus = build_consensus(rf_designs, comp_designs,
                                rf_score_col="p_binder", composite_score_col=comp_col)
    consensus = attach_filter(consensus, flags)          # complete ranking + filter flags retained
    consensus.to_csv(output_dir / "consensus_scores.csv", index=False)
    write_shortlist(consensus, flags, use_consensus_fn=True)
    disagreements(consensus, SHORTLIST_K).to_csv(output_dir / "disagreements.csv", index=False)

    ag = rank_agreement(consensus)
    lines = [
        f"dataset: {cfg.name}",
        f"designs (RF & composite overlap n): {ag['n']}",
        f"Spearman rank agreement: {ag['spearman']}",
        f"Kendall  rank agreement: {ag['kendall']}",
        f"shortlist size (k): {SHORTLIST_K}",
        f"composite column: {comp_col}",
    ] + notes
    (output_dir / "agreement.txt").write_text("\n".join(lines) + "\n")
    print(f"[{cfg.name}] consensus -> {output_dir}")
    print("\n".join(lines))
else:
    # RF not available yet: still deliver the composite-only design ranking.
    notes.append(f"RF design scores absent ({rf_path}); RF notebook not wired to emit them yet.")
    ranked = comp_designs.sort_values(comp_col, ascending=False).reset_index(drop=True)
    ranked["comp_rank"] = range(1, len(ranked) + 1)
    ranked = attach_filter(ranked, flags)
    ranked.to_csv(output_dir / "consensus_scores.csv", index=False)
    write_shortlist(ranked, flags, use_consensus_fn=False)
    lines = ["dataset: " + cfg.name,
             "consensus SKIPPED (composite-only ranking):",
             *["  - " + n for n in notes],
             f"composite column: {comp_col}"]
    (output_dir / "agreement.txt").write_text("\n".join(lines) + "\n")
    print(f"[{cfg.name}] consensus (composite-only) -> {output_dir}")
    for n in notes:
        print("  NOTE:", n)
