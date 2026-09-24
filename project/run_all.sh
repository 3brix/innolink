#!/usr/bin/env bash
# ===========================================================================
# run_all.sh — run the binder benchmarking & ranking pipeline end to end
# ===========================================================================
# Runs every stage in order for ONE dataset (or pooled set), stopping at the
# first failure. All configuration comes from environment variables (see
# README.md / .env.example); nothing is hardcoded here.
#
#   DATASET   which dataset or pooled set to run        (default: rf)
#   SCALER    'robust' or 'standard' for run_scale.py   (default: robust)
#   RUN_RF    1 = also run the Random Forest notebook   (default: 1)
#             set RUN_RF=0 to skip it (e.g. no Jupyter installed)
#
# One run directory is created up front (runs/<dataset>_<timestamp>/). It holds
# a snapshot of config/, a manifest.json, per-stage logs, and the executed RF
# notebook. Stage OUTPUT tables still land in data/ exactly as before — the run
# directory is added for provenance, it does not move the pipeline's outputs.
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
export SCALER="${SCALER:-robust}"
RUN_RF="${RUN_RF:-1}"
RF_NOTEBOOK="notebooks/ranking_rf_final.ipynb"

echo "=================================================================="
echo " Binder pipeline"
echo "   dataset : $DATASET"
echo "   scaler  : $SCALER"
echo "   run RF  : $RUN_RF"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Create ONE run directory shared by every stage (config snapshot + manifest).
# config/provenance.py prints the directory path on stdout.
# ---------------------------------------------------------------------------
RUN_DIR="$(python -m config.provenance)"
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
    python -c "import os; from config.provenance import record_stage; \
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
run_stage data_prep   python run_data_prep.py     # merge, label, split -> merged/eval/design.csv
run_stage align       python run_align.py         # direction-align metrics
run_stage scale       python run_scale.py         # scale (SCALER)
run_stage qc          python run_qc.py            # integrity / missingness / non-finite reports
run_stage profiling   python run_profiling.py     # PROFILING: composition, class dist, missingness, ranges
run_stage evaluation  python run_evaluation.py    # BENCHMARK: which metrics separate binders
run_stage thresholds  python run_thresholds.py    # REPORT-ONLY: custom /literature cutoffs
run_stage composite   python run_composite.py     # leakage-safe composite metric development

# ---------------------------------------------------------------------------
# Random Forest ranker (primary ranking method), kept as a notebook and run
# headlessly. The executed copy is archived in the run directory.
# ---------------------------------------------------------------------------
if [ "$RUN_RF" = "1" ]; then
    if ! command -v jupyter >/dev/null 2>&1; then
        echo "!!! RUN_RF=1 but 'jupyter' is not installed (pip install -r requirements.txt),"
        echo "    or set RUN_RF=0 to skip the RF stage."
        exit 1
    fi
    # --allow-errors: the notebook keeps exploratory cells (with stale hardcoded paths) that would
    # otherwise abort a headless run. The canonical PIPELINE STAGE cell runs FIRST and writes the
    # outputs; --allow-errors lets the later exploratory cells fail without killing the pipeline.
    run_stage ranking_rf jupyter nbconvert --to notebook --execute --allow-errors "$RF_NOTEBOOK" \
        --output ranking_rf_final.executed.ipynb --output-dir "$RUN_DIR"
    # Guard: because --allow-errors hides failures, confirm the canonical RF output actually exists.
    run_stage ranking_rf_check python -c 'from config.datasets import cfg; from config.paths import EVALUATION_DIR; import sys; sys.exit(0 if (EVALUATION_DIR/cfg.name/"ranking"/"rf_design_scores.csv").exists() else 1)' 
else
    echo ""
    echo "(skipping RF notebook stage; RUN_RF=$RUN_RF)"
fi

# ---------------------------------------------------------------------------
# Filter designs (reported funnel) and build the RF+composite consensus.
# run_consensus tolerates a missing RF file (until the RF notebook is wired):
# it then reports the composite-only ranking instead of failing.
# ---------------------------------------------------------------------------
run_stage filter      python run_filter.py        # literature /custom dev gates
run_stage consensus   python run_consensus.py     # RF (primary) + composite -> shortlist + disagreement

echo ""
echo "=================================================================="
echo " Pipeline complete."
echo "   manifest : $RUN_DIR/manifest.json"
echo "   logs     : $LOG_DIR"
echo "=================================================================="
