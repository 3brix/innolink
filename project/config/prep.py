"""
Columns removed during preprocessing (config.prep.COLS_TO_DROP).
Curation list for the prediction tables. Grouped by reason below.
"""

COLS_TO_DROP = [
    # ALL cf PyRosetta metrics --> dropped: the cf structures were not relaxed first, so the
    # energy terms blow up (cf_interface_dG median 401 vs af3 25; corr(af3, cf) ~ 0) and the
    # side-chain-sensitive terms are unreliable (interface_sc 0.21), while the
    # surface/sequence terms merely duplicate af3 (net_charge rho = 1.000, hydrophobic_sasa
    # 0.975, sap_score 0.954). Developability is therefore assessed on ONE reference
    # structure (af3) instead of an arbitrary AND across two structures of unequal quality.
    "cf_interface_dG",
    "cf_interface_dSASA",
    "cf_interface_dG_SASA_ratio",
    "cf_interface_sc",
    "cf_interface_packstat",
    "cf_interface_nres",
    "cf_interface_hydrophobicity",
    "cf_interface_interface_hbonds",
    "cf_interface_delta_unsat_hbonds",
    "cf_sap_score",
    "cf_sap_score_complex",
    "cf_dsap",
    "cf_hydrophobic_sasa",
    "cf_net_charge",
    "cf_surface_hydrophobicity",
    "cf_pyros_binder_score",

    # sap_score_unbound_cframe is byte-identical to sap_score (max|diff| = 0, rho = 1.000000 on
    # 154 samples, both af3 and cf) --> a duplicate column, not a second measurement.
    # (sap_score_complex IS distinct, rho = 0.919, and is kept.)
    "af3_sap_score_unbound_cframe",
    "cf_sap_score_unbound_cframe",

    # cf-vs-af3 RMSD --> measures disagreement between two PREDICTED structures, not
    # binding or structural quality, so it is not a useful predictor here.
    "cf_af3_rmsd",

    # Non-metric columns (labels, file paths) --> never features
    "binding",
    "interface",
    "chai_unconstrained_model_path",
    "chai_constrained_model_path",
    "boltz_free_model_path",
    "boltz_template_model_path",

    # rank / ranking-score columns
    "af3_rank",
    "af3_ranking_score",
    "af3_rank0_masif",
    "cf_rank",
    "cf_rank0_masif",
    "chai_unconstrained_score",
    "chai_unconstrained_aggregate_score",
    "boltz_free_confidence_score",
    "boltz_template_confidence_score",

     # Chai-1 --> currently dropped + duplicate confidence/interface columns (stay dropped)

    "chai_unconstrained_ptm",
    "chai_unconstrained_per_chain_ptm",
    "chai_unconstrained_iptm",
    "chai_unconstrained_per_chain_pair_iptm",
    "chai_unconstrained_interface_iptm",
    "chai_unconstrained_ipsae",
    "chai_unconstrained_pdockq2",
    "chai_unconstrained_lis",
    "chai_constrained_score",
    "chai_constrained_aggregate_score",
    "chai_constrained_ptm",
    "chai_constrained_per_chain_ptm",
    "chai_constrained_iptm",
    "chai_constrained_interface_iptm",
    "chai_constrained_per_chain_pair_iptm",
    "chai_constrained_ipsae",
    "chai_constrained_pdockq2",
    "chai_constrained_lis",
    "chai_unconstrained_pdockq",
    "chai_constrained_pdockq",

    # Chai (both variants) --> excluded: metrics absent for benoit (100%) and near-absent for the
    # esm0 designs (97%). An excluded model must not supply gate verdicts either.
    "chai_unconstrained_esm3dg_dg",
    "chai_unconstrained_esm3dg_dg_a",
    "chai_unconstrained_esm3dg_dg_b",
    "chai_unconstrained_esm3dg_dg_std",
    "chai_unconstrained_mpnn_confidence",
    "chai_unconstrained_mpnn_nll",
    "chai_unconstrained_mpnn_nll_a",
    "chai_unconstrained_mpnn_nll_b",
    "chai_constrained_esm3dg_dg",
    "chai_constrained_esm3dg_dg_a",
    "chai_constrained_esm3dg_dg_b",
    "chai_constrained_esm3dg_dg_std",
    "chai_constrained_mpnn_confidence",
    "chai_constrained_mpnn_nll",
    "chai_constrained_mpnn_nll_a",
    "chai_constrained_mpnn_nll_b",


    # ESMFold (v1) --> excluded (ESMFold2 replaces it). Its esm3dg columns go too: keeping an
    # excluded model's energy output is not a defensible split, and coverage was 56.5% on benoit.
    "esmfold_pae",
    "esmfold_plddt",
    "esmfold_ptm",
    "esmfold_ipae",
    "esmfold_iplddt",
    "esmfold_mpnn_confidence",
    "esmfold_mpnn_nll",
    "esmfold_mpnn_nll_a",
    "esmfold_mpnn_nll_b",
    "esmfold_esm3dg_dg",
    "esmfold_esm3dg_dg_a",
    "esmfold_esm3dg_dg_b",
    "esmfold_esm3dg_dg_std",



    # Single-dataset metrics --> excluded. af3/cf ESM3 dG and MPNN exist for the peptide dataset
    # only, so their pooled PR-AUC reflects that dataset's prevalence, not quality (they took 4 of
    # the top 5 pooled slots). The ESMFold2 / Boltz variants are complete and remain.
    "af3_esm3dg_dg",
    "af3_esm3dg_dg_a",
    "af3_esm3dg_dg_b",
    "af3_esm3dg_dg_std",
    "af3_mpnn_confidence",
    "af3_mpnn_nll",
    "af3_mpnn_nll_a",
    "af3_mpnn_nll_b",
    "cf_esm3dg_dg",
    "cf_esm3dg_dg_a",
    "cf_esm3dg_dg_b",
    "cf_esm3dg_dg_std",
    "cf_mpnn_confidence",
    "cf_mpnn_nll",
    "cf_mpnn_nll_a",
    "cf_mpnn_nll_b",


    # pDockQ (v1) columns --> excluded, pDockQ2 is used instead
    "af3_pdockq",
    "cf_pdockq",
    "boltz_free_pdockq",
    "boltz_template_pdockq",

]


#AMBIGUOUS_TARGET = "HSA"

