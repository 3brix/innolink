"""Paths and shared constants for the rankings EDA pipeline.

Everything here was previously inline in 03_rankings.ipynb. Edit values here
rather than in the pipeline code.
"""

# ============================================================================
# USAGE
# ============================================================================
# To run a notebook for a specific dataset, set the DATASET environment variable:
#
#   export DATASET=mcmahon && jupyter notebook prep.ipynb
#   export DATASET=snir && jupyter notebook prep.ipynb
#   export DATASET=germinal && jupyter notebook prep.ipynb
#   export DATASET=harvey && jupyter notebook prep.ipynb
#   export DATASET=peptide && jupyter notebook prep.ipynb
#
# All dataset-specific paths will be loaded automatically from DATASET_CONFIGS



import os
from pathlib import Path

# ============================================================================
# DATASET SELECTOR
# ============================================================================
# Set DATASET via environment variable: export DATASET=mcmahon
# Defaults to 'mcmahon' if not set
DATASET = os.getenv('DATASET', 'alphaseq').lower()

DATASET_CONFIGS = {
    'mcmahon': {
        'KEEP_INTERFACES': None,
        'PREDS_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/preds_mcmahon.csv"),
        'MASTER_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/mt_mcmahon.csv"),
        'PROCESSED_DATA_DIR': Path("/scicore/home/schwede/barta0000/project/data/processed/mcmahon_ds"),
        'RAW_DATA_DIR':Path("/scicore/home/schwede/barta0000/project/data/raw/mcmahon"),
    },
    'snir': {
        'KEEP_INTERFACES': {"A,C", "B,C"},
        'PREDS_PATH': Path("/scicore/home/schwede/barta0000/scoring_pipeline/run1/predictions.csv"),
        'MASTER_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/mt_snir.csv"),
        'PROCESSED_DATA_DIR':  Path("/scicore/home/schwede/barta0000/project/data/processed/snir_ds"),
        'RAW_DATA_DIR':Path("/scicore/home/schwede/barta0000/project/data/raw/snir"),
    },
    'germinal': {
        'KEEP_INTERFACES': None,
        'PREDS_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/preds_germinal.csv"),
        'MASTER_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/mt_germinal.csv"),
        'PROCESSED_DATA_DIR':  Path("/scicore/home/schwede/barta0000/project/data/processed/germinal_ds"),
        'RAW_DATA_DIR':Path("/scicore/home/schwede/barta0000/project/data/raw/germinal"),
    },
    'harvey': {
        'KEEP_INTERFACES': None,
        'PREDS_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/preds_harvey.csv"),
        'MASTER_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/mt_harvey.csv"),
        'PROCESSED_DATA_DIR':  Path("/scicore/home/schwede/barta0000/project/data/processed/harvey_ds"),
        'RAW_DATA_DIR':Path("/scicore/home/schwede/barta0000/project/data/raw/harvey"),
    },
    'peptide': {
        'KEEP_INTERFACES': None,
        'PREDS_PATH': Path("/scicore/home/schwede/barta0000/peptide_ds/originals/predictions.csv"),
        'MASTER_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/mt_peptide.csv"),
        'PROCESSED_DATA_DIR':  Path("/scicore/home/schwede/barta0000/project/data/processed/peptide_ds"),
        'RAW_DATA_DIR':Path("/scicore/home/schwede/barta0000/project/data/raw/peptide"),
    },
    'alphaseq': {
    'KEEP_INTERFACES': {"A,C", "B,C"},
    'PREDS_PATH': Path("/scicore/home/schwede/barta0000/alphaseq_ds/test/predictions.csv"),
    'MASTER_PATH': Path("/scicore/home/schwede/barta0000/project/data/raw/mt_alphaseq.csv"),
    'PROCESSED_DATA_DIR':  Path("/scicore/home/schwede/barta0000/project/data/processed/alphaseq_ds"),
    'RAW_DATA_DIR':Path("/scicore/home/schwede/barta0000/project/data/raw/alphaseq"),
},
}

# Load current dataset config
if DATASET not in DATASET_CONFIGS:
    raise ValueError(f"Unknown dataset: {DATASET}. Must be one of {list(DATASET_CONFIGS.keys())}")

_current = DATASET_CONFIGS[DATASET]
KEEP_INTERFACES = _current['KEEP_INTERFACES']
PREDS_PATH = _current['PREDS_PATH']
MASTER_PATH = _current['MASTER_PATH']
PROCESSED_DATA_DIR = _current['PROCESSED_DATA_DIR']
RAW_DATA_DIR = _current['RAW_DATA_DIR']
STAT_DIR= "/scicore/home/schwede/barta0000/EDA/static/snir_outputs/snir_stats"
# --- Columns dropped before analysis ---
# COLS_TO_DROP (in data_prep)
COLS_TO_DROP = ["af3_rank", "af3_ranking_score", "af3_rank0_masif", "cf_rank", "cf_rank0_masif", "binding", "interface","chai_unconstrained_model_path", "chai_constrained_model_path", "boltz_free_model_path", "boltz_template_model_path"]

# ============================================================================
# 01_data_prep.ipynb: Paths to raw prediction CSVs and metric directions YAML
# ============================================================================
AMBIGUOUS_TARGET= "HSA"   
#DATA_DIR = Path("/scicore/home/schwede/barta0000/project/data/processed")

GERMINAL_CSV = DATA_DIR / "germinal_ds/scores_df_extended_with_chai.csv"
MCMAHON_CSV = DATA_DIR / "mcmahon_ds/scores_df_extended_with_chai.csv"
SNIR_CSV = DATA_DIR / "snir_ds/scores_df_extended_with_chai.csv"
HARVEY_CSV = DATA_DIR / "harvey_ds/scores_df_extended_with_chai.csv"
PEPTIDE_CSV = DATA_DIR / "peptide_ds/scores_df_extended_with_chai.csv"
ALPHASEQ_CSV = DATA_DIR / "alphaseq_ds/scores_df_extended_with_chai.csv"
YAML_PATH = Path("/home/bri/pymol/project_offline/project_current/config/metric_data.yaml")
# --- Output ---
OUT_DIR = Path("/home/bri/pymol/project_offline/project_current/notebooks")

# --- Stratified subsampling: (binder, type) -> n samples drawn from germinal ---
GERMINAL_SAMPLE_SPEC = {
    (1, "original"): 24,  # Positive
    (0, "original"): 13,  # Hard Negative
    (0, "target_shuffle"): 13,  # Easy Negative
}
# NOTE: the original notebook comment said "6 iterations" but the loop only
# ever ran `range(1)`. Bump this once you've confirmed which was intended —
# everything downstream already supports n_iterations > 1.
N_ITERATIONS = 1


# --- Columns excluded from "numeric metric" treatment ---
META_COLS = ["sample", "binder", "source", "type", "iteration", "binder_class"]
EXCLUDE_COLS = ["KD[M]", "EC50[M]"]

# --- Plot styling ---
COLORS_MAP = {
    "Binder": "#2ecc71",
    "Non-Binder": "#67e7dd",
    "Target Shuffle": "#273397",
    "germinal": "#2ecc71",
    "mcmahon": "#d6a439",
    "snir_ab": "#1f3397",
    "harvey": "#971f6f",
    "peptide": "#7f00ff",
    "alphaseq": "#ff7f00",
}

# --- Model groupings used for per-model separability plots ---
MODELS = ["af3", "cf", "esmfold", "chai_constrained", "chai_unconstrained", "boltz_free", "boltz_template"]
MODEL_ORDER = MODELS  # consistent x-axis ordering for the agreement plot

# --- Metric families used for the violin "model agreement" plot ---
# Only "ipsae" was actually plotted in the original notebook.
AGREEMENT_FAMILIES = ["ipsae"]

# Defined in the original notebook but not consumed anywhere downstream yet
# (kept here in case you wire it into per-family separability plots later).
METRIC_FAMILIES = {
    "plddt": ["af3_plddt", "cf_plddt", "esmfold_plddt", "boltz_free_plddt", "boltz_template_plddt"],
    "ptm": [
        "af3_ptm", "cf_ptm", "esmfold_ptm", "chai_unconstrained_ptm", "chai_constrained_ptm",
        "boltz_free_ptm", "boltz_template_ptm",
    ],
    "iptm": [
        "af3_iptm", "cf_iptm", "chai_unconstrained_iptm", "chai_constrained_iptm",
        "boltz_free_iptm", "boltz_template_iptm",
    ],
    "pdockq2": [
        "af3_pdockq2", "cf_pdockq2", "chai_unconstrained_pdockq2", "chai_constrained_pdockq2",
        "boltz_free_pdockq2", "boltz_template_pdockq2",
    ],
}

# Focused subset of physics-based metrics used for the secondary histogram pass
PHYSICS_COLS = [
    "cf_af3_rmsd",
    "af3_interface_dG",
    "af3_interface_dG_SASA_ratio",
    "cf_interface_dG",
    "cf_interface_dG_SASA_ratio",
    "boltz_free_pdockq",
    "cf_pyros_binder_score",
]


