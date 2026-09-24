"""
Run provenance: snapshot the configuration used for a run and record what ran.

Reproducibility helper with no third-party dependencies. A "run" is a directory
under RUNS_DIR named ``<dataset>_<timestamp>``. It holds:

  - ``config_snapshot/``  a copy of every config/*.py and config/*.yaml as used;
  - ``manifest.json``     dataset, resolved paths, environment overrides, Python
                          and package versions, platform, git commit (if any),
                          and one entry per pipeline stage that ran.

Typical use from a runner::

    from config.provenance import new_run_dir, record_stage
    run = new_run_dir()                       # once per pipeline invocation
    ...                                       # do the stage's work
    record_stage(run, "evaluation", outputs=["rankings.csv"])

Or reuse an existing run directory across stages by passing its path
(e.g. from the RUN_DIR environment variable that run_all.sh will set).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess

from config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RUNS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EVALUATION_DIR,
    ensure_directory,
)

# Packages worth recording for reproducibility (best-effort; missing ones skipped).
_TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn",
    "matplotlib", "seaborn", "pyyaml", "biopython",
    "nbconvert", "nbformat",
]

# Environment overrides that affect where inputs/outputs live.
_TRACKED_ENV = [
    "PROJECT_ROOT", "DATA_DIR", "RUNS_DIR", "EXTERNAL_DATA_ROOT", "DATASET",
]

_CONFIG_DIR = Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _package_versions() -> dict:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _git_info() -> dict:
    """Return {commit, dirty} if this is a git checkout, else {available: False}."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True,
        )
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"available": False}


def _current_dataset_name() -> str:
    """Resolve the selected dataset/set name."""
    try:
        from config.datasets import cfg
        return cfg.name
    except Exception:
        return os.getenv("DATASET", "unknown").lower()


def _members_for(name: str) -> list[str] | None:
    """Return the member datasets if `name` is a pooled analysis set, else None."""
    try:
        from config.analysis import ANALYSIS_SETS
        return list(ANALYSIS_SETS[name]) if name in ANALYSIS_SETS else None
    except Exception:
        return None


def _snapshot_config(dest: Path) -> None:
    ensure_directory(dest)
    for pattern in ("*.py", "*.yaml"):
        for src in _CONFIG_DIR.glob(pattern):
            if src.name.endswith(".bak"):
                continue
            shutil.copy2(src, dest / src.name)


def build_manifest(dataset: str | None = None) -> dict:
    """Assemble the manifest dict describing the current environment."""
    name = dataset or _current_dataset_name()
    members = _members_for(name)
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": name,
        "dataset_members": members,
        "paths": {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "DATA_DIR": str(DATA_DIR),
            "RAW_DATA_DIR": str(RAW_DATA_DIR),
            "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
            "EVALUATION_DIR": str(EVALUATION_DIR),
            "RUNS_DIR": str(RUNS_DIR),
        },
        "env_overrides": {k: os.getenv(k) for k in _TRACKED_ENV},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git": _git_info(),
        "stages": [],
    }


def new_run_dir(dataset: str | None = None, run_id: str | None = None) -> Path:
    """Create runs/<dataset>_<timestamp>/, snapshot config, write manifest, return it."""
    name = dataset or _current_dataset_name()
    run_id = run_id or f"{name}_{_timestamp()}"
    run_dir = ensure_directory(RUNS_DIR / run_id)
    _snapshot_config(run_dir / "config_snapshot")
    manifest = build_manifest(dataset=name)
    manifest["run_id"] = run_id
    _write_manifest(run_dir, manifest)
    return run_dir


def _manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / "manifest.json"


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    _manifest_path(run_dir).write_text(json.dumps(manifest, indent=2))


def read_manifest(run_dir: Path) -> dict:
    return json.loads(_manifest_path(run_dir).read_text())


def record_stage(run_dir: Path, stage: str, status: str = "ok",
                 outputs: list[str] | None = None, extra: dict | None = None) -> None:
    """Append one stage record to the run's manifest.json."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir) if _manifest_path(run_dir).is_file() else build_manifest()
    entry = {
        "stage": stage,
        "status": status,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if outputs:
        entry["outputs"] = outputs
    if extra:
        entry["extra"] = extra
    manifest.setdefault("stages", []).append(entry)
    _write_manifest(run_dir, manifest)


if __name__ == "__main__":
    # Convenience CLI: `python -m config.provenance` creates a run dir and prints it.
    # run_all.sh (Phase 4) will use this to make one run dir shared across stages.
    print(new_run_dir())
