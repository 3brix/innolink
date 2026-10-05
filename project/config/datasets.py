"""Dataset configuration. DATASET selects one dataset or a pooled set from ANALYSIS_SETS:

    export DATASET=alphaseq     # one dataset
    export DATASET=antibody     # a pooled set

The selection is 'cfg'. A set has 'members' and 'is_set == True'; runners use only `cfg.name`
and work the same either way, since a set is materialised as a virtual dataset under RAW_DATA_DIR.
"""

from dataclasses import dataclass
from pathlib import Path
import os

from config.paths import RAW_DATA_DIR
from config.analysis import ANALYSIS_SETS


# Root for prediction input files.
EXTERNAL_DATA_ROOT = Path(
    os.getenv("EXTERNAL_DATA_ROOT", ".../project/")
).expanduser()



# Dataset definition


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    predictions_path: Path | None = None
    mastertable_path: Path | None = None
    mastertable_strategy: str | None = None
    #ambiguous_target: str | None = None
    keep_interfaces: set[str] | None = None
    members: tuple[str, ...] | None = None   # populated only for pooled sets
    mol_type: str | None = None              # 'nanobody' / 'antibody' (persisted downstream)

    @property
    def is_set(self) -> bool:
        """True for a pooled ANALYSIS_SET (has members); False for a real dataset."""
        return self.members is not None


# ---------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------

def make_dataset(
    *,
    name: str,
    predictions: str | Path,
    mastertable_strategy: str | None = None,
    #ambiguous_target: str | None = None,
    keep_interfaces: set[str] | None = None,
    mol_type: str | None = None,
    mastertable: str | Path | None = None,
) -> DatasetConfig:
    """`mastertable` defaults to RAW_DATA_DIR/mt_<name>.csv; pass it explicitly when two
    datasets are different prediction subsets of the SAME samples (e.g. esm0 / esm0_top10)."""

    # molecule type defaults to the mastertable strategy (nanobody / antibody)
    if mol_type is None and mastertable_strategy in {"nanobody", "antibody"}:
        mol_type = mastertable_strategy

    return DatasetConfig(
        name=name,
        predictions_path=Path(predictions),
        mastertable_path=Path(mastertable) if mastertable else RAW_DATA_DIR / f"mt_{name}.csv",
        mastertable_strategy=mastertable_strategy,
        #ambiguous_target=ambiguous_target,
        keep_interfaces=keep_interfaces,
        mol_type=mol_type,
    )


# Available datasets
DATASETS = {

    "mcmahon": make_dataset(
        name="mcmahon",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/preds_mcmahon.csv",
        mastertable_strategy="nanobody",
        #ambiguous_target="HSA",
    ),

    "snir": make_dataset(
        name="snir",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/snir_predictions_with_sap.csv",
        keep_interfaces={"A,C", "B,C"},
        mastertable_strategy="antibody",
    ),

    "germinal": make_dataset(
        name="germinal",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/preds_germinal.csv",
        mastertable_strategy="nanobody",
    ),

    "harvey": make_dataset(
        name="harvey",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/preds_harvey.csv",
        mastertable_strategy="nanobody",
    ),

    "peptide": make_dataset(
        name="peptide",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/peptide_predictions_with_sap.csv",
        mastertable_strategy="nanobody",
    ),

    "alphaseq": make_dataset(
        name="alphaseq",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/alphaseq_predictions_with_sap.csv",
        mastertable_strategy="antibody",
        keep_interfaces={"A,C", "B,C"},
    ),
    
    "germinal0": make_dataset(
        name="germinal0",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/germinal0_predictions_with_sap.csv",
        mastertable_strategy="nanobody",
    ),
    
    # esm0 = the full design set that gets RANKED; it needs no developability metrics
    "esm0": make_dataset(
        name="esm0",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/esm0_predictions.csv",
        mastertable_strategy="nanobody",
    ),

    # esm0_top10 = the 10 RF-selected esm0 designs re-run with developability metrics, to TEST
    # the filter. Shares mt_esm0.csv. Kept OUT of 'rf' / 'designs': chosen by the RF, so ranking
    # them with the RF would be circular.
    "esm0_top10": make_dataset(
        name="esm0_top10",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/esm0_top10_predictions_with_sap.csv",
        mastertable=EXTERNAL_DATA_ROOT / "data/raw/mt_esm0.csv",
        mastertable_strategy="nanobody",
    ),

    "top5": make_dataset(
        name="top5",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/top5_predictions_with_sap.csv",
        mastertable_strategy="nanobody",
    ),

    "benoit": make_dataset(
        name="benoit",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/benoit_predictions_with_sap.csv",
        mastertable_strategy="antibody",
    ),

    "sabdab_nb": make_dataset(
        name="sabdab_nb",
        predictions=EXTERNAL_DATA_ROOT / "data/raw/sabdab_predictions_with_sap.csv",
        mastertable_strategy="nanobody",
    ),

}



# Pooled analysis sets (from config.analysis.ANALYSIS_SETS)
SETS = {
    name: DatasetConfig(name=name, members=tuple(members))
    for name, members in ANALYSIS_SETS.items()
}

# dataset names and set names share one selection namespace -> must not collide
_overlap = set(DATASETS) & set(SETS)
if _overlap:
    raise ValueError(f"Name collision between datasets and ANALYSIS_SETS: {sorted(_overlap)}")

# every set member must be a known dataset
for _set, _c in SETS.items():
    _unknown = [m for m in _c.members if m not in DATASETS]
    if _unknown:
        raise ValueError(f"ANALYSIS_SET '{_set}' references unknown datasets: {_unknown}")

SELECTABLE = {**DATASETS, **SETS}



# Select current dataset or set

DATASET = os.getenv("DATASET", "rf").lower()

if DATASET not in SELECTABLE:
    raise ValueError(
        f"Unknown selection '{DATASET}'. "
        f"Datasets: {', '.join(DATASETS)}. Sets: {', '.join(SETS)}."
    )

cfg = SELECTABLE[DATASET]