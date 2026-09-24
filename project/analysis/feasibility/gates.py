# base metric name (without the <model>_ prefix), kind, threshold/bounds
DEFAULT_GATES = [
    {"metric": "interface_delta_unsat_hbonds", "kind": "max",  "value": 4},
    {"metric": "sap_score",                    "kind": "max",  "value": 0.4},
    #{"metric": "hydrophobic_sasa_ratio",       "kind": "max",  "value": 1.7},   # needs the ratio column (see note)
    {"metric": "net_charge",                   "kind": "band", "bounds": (-4, 4)},
    {"metric": "surface_hydrophobicity",       "kind": "band", "bounds": (0.25, 0.40)},
]

# NOTE: hydrophobic_sasa_ratio = hydrophobic_sasa / ideal_surface is NOT emitted by
# calc_sap.py yet (only hydrophobic_sasa is). Until that column exists this gate is
# auto-skipped. Add it at SAP-compute time, or drop this gate.