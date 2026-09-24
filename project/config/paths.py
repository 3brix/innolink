"""
Central path configuration.

Every location derives from PROJECT_ROOT, resolved in this order:
  1. the PROJECT_ROOT environment variable, if set;
  2. otherwise the repository root (the parent of this config/ directory),
     so the pipeline runs from a fresh checkout with no configuration.

DATA_DIR and RUNS_DIR can each be overridden the same way; by default they sit
under PROJECT_ROOT. Overrides may also be placed in a .env file at the
repository root (KEY=VALUE lines). Real environment variables always win over
the .env file, which only fills in values that are not already set.
"""

from pathlib import Path
import os


# Minimal .env support
def _load_dotenv(env_file: Path) -> None:
    """Read KEY=VALUE lines from `env_file` into os.environ, without overriding
    variables that are already set in the real environment."""
    if not env_file.is_file():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# repository root is the parent of this config / directory
_REPO_ROOT = Path(__file__).resolve().parent.parent

# load optional .env
_load_dotenv(_REPO_ROOT / ".env")


def _env_path(var: str, default: Path) -> Path:
    """Return Path from environment variable 'var', else 'default'."""
    value = os.getenv(var)
    return Path(value).expanduser() if value else default


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Base project directories
PROJECT_ROOT = _env_path("PROJECT_ROOT", _REPO_ROOT)
DATA_DIR = _env_path("DATA_DIR", PROJECT_ROOT / "data")
RUNS_DIR = _env_path("RUNS_DIR", PROJECT_ROOT / "runs")

RAW_DATA_DIR = DATA_DIR / "raw"
QC_DIR = DATA_DIR / "qc"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DISTRIBUTIONS_DIR = PROCESSED_DATA_DIR / "distributions"
EVALUATION_DIR = PROCESSED_DATA_DIR / "evaluation"

METRIC_YAML = PROJECT_ROOT / "config" / "metric_data.yaml"
THRESHOLDS_YAML = PROJECT_ROOT / "config" / "thresholds.yaml"
