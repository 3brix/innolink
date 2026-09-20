ANALYSIS_SETS = {

    "current": ["peptide", "sabdab_nb", "germinal0", "top5"],

    "antibody": ["alphaseq", "snir",],

    "nanobody": ["mcmahon", "germinal", "harvey",],

    "old": ["mcmahon", "germinal", "harvey", "snir",],

    "rf": ["peptide", "alphaseq", "benoit", "snir", "germinal0","esm0"],

    "designs": ["germinal0","esm0",], 
}

META_COLS = ["sample", "binder", "source", "type", "iteration", "binder_type", "dataset", "mol_type"]

# Categories kept in the data for FILTERING only, and excluded from the model/benchmark
# feature set (eval / RF / composite). Developability & energy metrics are used as
# literature filtering gates (energy mainly for nanobodies) but are NOT features.
FILTER_ONLY_CATEGORIES = {"developability", "energy"}

EXCLUDE_COLS = ["KD[M]", "EC50[M]",]

EXCLUDE_COLUMNS = META_COLS + EXCLUDE_COLS

# --- Plot styling ---
COLORS_MAP = {
    "Binder": "#2ecc71",
    "Design": "#d68039",    
    "Non-Binder": "#eb405d",
    "Mutant+": "#40ebdd",
    "Mutant-": "#ac40eb",
    #"Target Shuffle": "#273397",
    "germinal": "#2ecc71",
    #"mcmahon": "#d6a439",
    "snir_ab": "#1f3397",
    #"harvey": "#971f3d",
    "peptide": "#7f00ff",
    "alphaseq": "#ff2600",
    "germinal": "#2ecc71",
    "esm": "#d6399a", 
    "benoit": "#20B2AA",
    "top5": "#FF8C00",
    "sabdab_nb": "#FF1493",
    "sabdab": "#FF1493",
}

# Window metric families: an optimal RANGE, not higher/lower-is-better..)
WINDOW_FAMILIES = {"net_charge", "surface_hydrophobicity"}

# composite test (unused)
MODELS = ["af3", "cf", "esmfold", "esmfold2", "chai_constrained", "chai_unconstrained", "boltz_free", "boltz_template"]
QUALITY_THRESHOLDS = {"af3_iptm": 0.5, "af3_plddt": 0.8}  # plddt is 0-1 post data_prep.normalize_plddt (decision: 0.8, was stale 80)
QUALITY_MODE = "any"