from pathlib import Path


from pathlib import Path


def ensure_directory(path: Path):
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
# ---------------------------------------------------------------------
# Base project directories
# ---------------------------------------------------------------------


PROJECT_ROOT = Path("/scicore/home/schwede/barta0000/project")

RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
QC_DIR = PROJECT_ROOT / "data/qc"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
DISTRIBUTIONS_DIR = PROJECT_ROOT / "data/processed/distributions"
EDA_ROOT = Path("/scicore/home/schwede/barta0000/EDA")
METRIC_YAML = PROJECT_ROOT / "config/metric_data.yaml"

EVALUATION_DIR = PROJECT_ROOT / "data/processed/evaluation"

STATIC_DIR = EDA_ROOT / "static"

RANKINGS_OUT_DIR = STATIC_DIR / "rankings_base"

STAT_DIR = STATIC_DIR / "snir_outputs/snir_stats"