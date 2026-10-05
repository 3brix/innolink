ANALYSIS_SETS = {

    "all": [ "snir", "mcmahon", "germinal", "harvey"],

    "current": ["peptide", "sabdab_nb", "germinal0", "top5"],

    "antibody": ["alphaseq", "snir",],

    "nanobody": ["mcmahon", "germinal", "harvey",],

    "old": ["mcmahon", "germinal", "harvey", "snir",],

    # benchmark (labelled: peptide/alphaseq/benoit/snir) + the design sets that get ranked
    "rf": ["peptide", "alphaseq", "benoit", "snir", "germinal0","esm0"],
    "rf_sabdab": ["sabdab_nb", "alphaseq", "benoit", "snir", "germinal0","esm0"],


    # the design sets that get ranked (germinal0 + the full esm0 design set).
    # NOTE: a design-only set has no labelled rows, so only the label-free stages apply
    # (run_scale / run_evaluation / run_composite all need a non-empty eval.csv).
    # Feasibility screening needs NO pooled set: germinal0's designs are already in 'rf'
    # (separable by the 'dataset' column), and the RF-selected top-10 is screened directly with
    #     DATASET=esm0_top10 PYTHONPATH=. python run_filter.py
    "designs": ["germinal0","esm0",],
}

META_COLS = ["sample", "binder", "source", "type", "iteration", "binder_class", "dataset", "mol_type"]

# Categories kept in the data for FILTERING only, and excluded from the model/benchmark
# feature set (eval / RF / composite).
#   developability, energy -- manufacturability / structural feasibility, not binding evidence.
#   sequence (MPNN)        -- present for only 60 of 133 labelled samples, on a 62%-positive
#                             subset, so PR-AUC ranked 1st-2nd at ROC-AUC 0.56 (chance).
#                             Treated as a filtering signal instead.
FILTER_ONLY_CATEGORIES = {"developability", "energy", "sequence"}

# Ranking benchmark: precision@pct (10%)  + PR-AUC, reported
PRECISION_AT_PERCENT = 0.10


EXCLUDE_COLS = ["KD[M]", "EC50[M]",]

EXCLUDE_COLUMNS = META_COLS + EXCLUDE_COLS

# The validated palette lives in config/palette.py; COLORS_MAP is re-exported here for the figure
# notebooks that already import it from this module. The old hand-written map failed a
# colour-vision check and keyed three datasets by names the data does not use.
from config.palette import COLORS_MAP  # noqa: E402,F401

# Window (two-sided) families are DERIVED from metric_data.yaml -- a family is a window exactly
# when its 'reference:' is a [lo, hi] pair. Use metric_meta.load_window_families; a hand-kept list
# here drifted out of step.

# model prefixes. 'pred' covers pred_lddt / pred_ilddt; without it assign_model() returns None
# for those two and they vanish from the model-grouped figures.
MODELS = ["af3", "cf", "esmfold", "esmfold2", "chai_constrained", "chai_unconstrained", "boltz_free", "boltz_template", "pred"]

# Filter config (S1 + S2, each gate with its own number) lives in config/thresholds.yaml.
# Conventional reference values are metadata: metric_data.yaml's per-metric 'reference:' field,
# which also defines which families are two-sided windows.