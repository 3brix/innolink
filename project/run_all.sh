#!/usr/bin/env bash
# ===========================================================================
# run_all.sh — run the binder benchmarking & ranking pipeline end to end
# ===========================================================================
# Runs every stage in order for ONE dataset (or pooled set), stopping at the
# first failure. All configuration comes from environment variables (see
# README.md / .env.example); nothing is hardcoded here.
#
#   DATASET   which dataset or pooled set to run        (default: rf)
#   SCALER    'standard' or 'robust' for run_scale.py   (default: standard)
#
# Every stage is a plain `"$PYTHON" run_<stage>.py`, so a failure anywhere stops the
# run. (The RF used to run as a notebook via nbconvert --allow-errors, which hid
# failures and needed a kernelspec workaround; it is now run_ranking_rf.py.)
#
# One run directory is created up front (runs/<dataset>_<timestamp>/). It holds
# a snapshot of config/, a manifest.json and per-stage logs. Stage OUTPUT tables
# still land in data/ exactly as before — the run directory is provenance only.
#
# Usage:
#   DATASET=rf ./run_all.sh
# ===========================================================================

# Fail fast: -e stop on first error, -u error on unset vars,
# pipefail so a failure inside a `cmd | tee` pipe is not masked by tee.
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the repo (this script's own folder) and make packages importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${PYTHONPATH:-.}"

# ---------------------------------------------------------------------------
# Configuration (all overridable from the environment / .env).
# ---------------------------------------------------------------------------
export DATASET="${DATASET:-rf}"
export SCALER="${SCALER:-standard}"

# Interpreter: prefer $PYTHON, then `python`, then `python3`. Conda/micromamba envs provide
# `python`, but a bare system PATH often only has `python3` -- assuming `python` made this script
# die on line 1 with "python: command not found".
PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [ -z "$PYTHON" ]; then
    echo "!!! no Python interpreter found. Activate your env, or set PYTHON=/path/to/python."
    exit 1
fi
# Fail early and clearly if the env is missing a dependency, rather than mid-pipeline.
if ! "$PYTHON" -c "import pandas, numpy, sklearn, yaml, scipy" 2>/dev/null; then
    echo "!!! $PYTHON cannot import the core dependencies (pandas/numpy/scikit-learn/pyyaml/scipy)."
    echo "    Activate the project env or: pip install -r requirements.txt"
    exit 1
fi

echo "=================================================================="
echo " Binder pipeline"
echo "   dataset : $DATASET"
echo "   scaler  : $SCALER"
echo "   python  : $PYTHON"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Create ONE run directory shared by every stage (config snapshot + manifest).
# config/provenance.py prints the directory path on stdout.
# ---------------------------------------------------------------------------
RUN_DIR="$("$PYTHON" -m config.provenance)"
export RUN_DIR
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Run directory: $RUN_DIR"

# ---------------------------------------------------------------------------
# run_stage <name> <command...>
#   Runs a stage, tees its combined output to logs/<name>.log, and appends a
#   record to the run manifest (status ok/failed). On failure: record, then
#   exit 1 so the whole pipeline stops (fail-fast).
# ---------------------------------------------------------------------------
run_stage () {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    echo ""
    echo "------------------------------------------------------------------"
    echo ">>> STAGE: $name"
    echo "    cmd  : $*"
    echo "    log  : $log"
    echo "------------------------------------------------------------------"

    # Run the stage; capture success/failure without tripping `set -e`.
    local ok=1
    if "$@" 2>&1 | tee "$log"; then ok=1; else ok=0; fi

    # Record the outcome in the manifest (RUN_DIR/name/status passed via env,
    # so paths with spaces or odd characters are handled safely).
    STAGE_NAME="$name" STAGE_STATUS="$([ "$ok" -eq 1 ] && echo ok || echo failed)" \
    "$PYTHON" -c "import os; from config.provenance import record_stage; \
record_stage(os.environ['RUN_DIR'], os.environ['STAGE_NAME'], \
status=os.environ['STAGE_STATUS'], extra={'log': 'logs/'+os.environ['STAGE_NAME']+'.log'})"

    if [ "$ok" -ne 1 ]; then
        echo "!!! STAGE '$name' FAILED — see $log"
        exit 1
    fi
    echo "<<< STAGE '$name' OK"
}

# ---------------------------------------------------------------------------
# Pipeline stages, in dependency order.
# ---------------------------------------------------------------------------
run_stage data_prep   "$PYTHON" run_data_prep.py     # merge, label, split -> merged/eval/design.csv
run_stage align       "$PYTHON" run_align.py         # direction-align metrics
run_stage scale       "$PYTHON" run_scale.py         # scale (SCALER)
run_stage qc          "$PYTHON" run_qc.py            # integrity / missingness / non-finite reports
run_stage profiling   "$PYTHON" run_profiling.py     # PROFILING: composition, class dist, missingness, ranges
run_stage evaluation  "$PYTHON" run_evaluation.py    # BENCHMARK: which metrics separate binders
# run_thresholds.py is NOT a pipeline stage: it is a one-off, report-only comparison of
# data-derived cutoffs against the literature values, and nothing downstream reads its output.
# Run it by hand when the report is wanted:  DATASET=rf PYTHONPATH=. python run_thresholds.py
run_stage composite   "$PYTHON" run_composite.py     # leakage-safe composite metric development

# ---------------------------------------------------------------------------
# Random Forest ranker (primary ranking method). Extracted from
# notebooks/ranking_rf_final.ipynb into a plain stage script, so it needs no
# jupyter and a failure stops the pipeline instead of being hidden.
# ---------------------------------------------------------------------------
run_stage ranking_rf   "$PYTHON" run_ranking_rf.py    # RF: grouped-CV metrics + design p_binder

# ---------------------------------------------------------------------------
# Filter designs (reported funnel) and build the RF+composite consensus.
# run_consensus tolerates a missing RF file (until the RF notebook is wired):
# it then reports the composite-only ranking instead of failing.
# ---------------------------------------------------------------------------
run_stage filter      "$PYTHON" run_filter.py        # feasibility screen: quality + literature developability gates
run_stage consensus   "$PYTHON" run_consensus.py     # RF (primary) + composite -> shortlist + disagreement

echo ""
echo "=================================================================="
echo " Pipeline complete."
echo "   manifest : $RUN_DIR/manifest.json"
echo "   logs     : $LOG_DIR"
echo "=================================================================="
