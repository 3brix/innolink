 Binder benchmarking & design-ranking pipeline

This pipeline takes structure-prediction metrics for antibody / nanobody designs,
**benchmarks which metrics separate binders from non-binders**, and then uses those
findings to **filter and rank new binder designs**. It grew out of a set of Jupyter
notebooks and is being organised into small, readable Python stage-scripts
(`run_*.py`) plus a few notebooks kept for figures and for the Random Forest ranker.

Two labelled tables drive everything:

- `eval.csv` — designs with a known label (`binder` is 0/1), used to *benchmark* metrics.
- `design.csv` — new designs with unknown label (`binder == "?"`), the ones we *rank*.

---

## 1. Setup

You need Python 3.10 or newer.

```bash
# from this folder (the repository root)
python -m venv .venv && source .venv/bin/activate      # or a conda env
pip install -r requirements.txt

# optional, for an exact reproducible lock on THIS machine:
pip freeze > requirements.lock.txt
```

The upstream developability scripts in `grouped_pyros/` need PyRosetta and are **not**
covered by `requirements.txt`; they run once, before this pipeline, to produce the
prediction CSVs. You only need them to regenerate inputs.

---

## 2. Configuration (paths & dataset selection)

All locations derive from a single `PROJECT_ROOT`. By default it is **this repository
folder**, so a fresh checkout runs with no configuration at all: `data/` and `runs/`
are expected next to the code.

Override any of these with environment variables, or by copying `.env.example` to
`.env` and editing it (real environment variables win over the `.env` file):

| Variable | Meaning | Default |
|---|---|---|
| `PROJECT_ROOT` | root everything derives from | the repository folder |
| `DATA_DIR` | where `raw/`, `processed/`, `qc/` live | `<PROJECT_ROOT>/data` |
| `RUNS_DIR` | where per-run provenance is written | `<PROJECT_ROOT>/runs` |
| `EXTERNAL_DATA_ROOT` | root holding the raw prediction CSVs | `/scicore/home/schwede/barta0000` |
| `DATASET` | which dataset or pooled set to run | `rf` |
| `SCALER` | `standard` or `robust` (used by `run_scale.py`) | `standard` |

`DATASET` picks either one dataset or a pooled **analysis set** (several datasets
combined). The datasets and sets are defined in `config/datasets.py` and
`config/analysis.py`. On the scicore cluster the defaults reproduce the original
absolute paths, so nothing needs to be set there.

Example for another machine:

```bash
export PROJECT_ROOT=/home/me/binder_project
export EXTERNAL_DATA_ROOT=/home/me/binder_project/predictions
export DATASET=rf
```

---

## 3. Data layout

```
<DATA_DIR>/
├── raw/
│   ├── mt_<dataset>.csv          # mastertable (labels/metadata) per dataset
│   └── <dataset>/                # per-dataset working tables (created by the pipeline)
│       ├── merged.csv            # predictions + mastertable, cleaned & merged
│       ├── eval.csv              # labelled rows (binder = 0/1)
│       ├── design.csv            # unlabelled rows (binder = "?")
│       ├── *_aligned.csv         # direction-aligned copies
│       └── *_scaled_aligned.csv  # scaled + aligned copies
├── qc/<dataset>/                 # QC reports
└── processed/evaluation/         # benchmark outputs
```

Raw prediction CSVs themselves live under `EXTERNAL_DATA_ROOT` (they are inputs, not
produced here).

---

## 4. Running the pipeline

Run from the repository root. Each stage reads the `DATASET` you select. Prefixing
with `PYTHONPATH=.` is the robust way to make the `config` / `analysis` packages
importable:

```bash
export DATASET=rf

PYTHONPATH=. python run_data_prep.py     # 1. merge, label, split -> merged/eval/design.csv
PYTHONPATH=. python run_align.py         # 2. direction-align metrics
PYTHONPATH=. python run_scale.py         # 3. scale (standard by default)
PYTHONPATH=. python run_qc.py            # 4. integrity / missingness / non-finite reports
PYTHONPATH=. python run_profiling.py     # 5. PROFILING: composition, class dist, missingness, ranges
PYTHONPATH=. python run_evaluation.py    # 6. BENCHMARK: which metrics separate binders
PYTHONPATH=. python run_composite.py     # 7. composite metric development (dataset-aware selection)
PYTHONPATH=. python run_ranking_rf.py    # 8. RF (PRIMARY ranker): grouped-CV metrics + design p_binder
PYTHONPATH=. python run_filter.py        # 9. filter designs (feasibility: quality + developability gates)
PYTHONPATH=. python run_consensus.py     # 10. RF (primary) + composite -> filter-aware shortlist + disagreement

# NOT a stage -- report-only, nothing downstream reads it. Run by hand if wanted:
PYTHONPATH=. python run_thresholds.py    # data-derived vs reference cutoffs
```

Or run the whole thing in order with the orchestrator (fail-fast, one shared run
directory, per-stage logs):

```bash
DATASET=rf ./run_all.sh
```

The **Random Forest ranker** (the primary ranking method) is `run_ranking_rf.py`, a normal
stage script like every other:

```bash
PYTHONPATH=. python run_ranking_rf.py
```

It trains the RF on all labelled data (`eval.csv`) and scores the unlabelled designs, writing
`data/processed/evaluation/<dataset>/ranking/` — `rf_design_scores.csv` (`sample`, `p_binder`,
consumed by `run_consensus.py`), `rf_cv_metrics.csv` and `rf_importances.csv`. Needs
scikit-learn >= 1.4. Two models are kept distinct: the lineage-grouped CV models used only to
estimate performance, and the final model refit on all labelled rows used only to score designs.

`rf_cv_metrics.csv` reports **pooled out-of-fold** PR-AUC and precision@10% — the same
estimator as the composite and the single-metric benchmark, so the three are directly
comparable. Per-fold mean PR-AUC is also reported (`pr_auc_fold_mean`) but it is a different
quantity and runs optimistic; don't compare it with the others.

The code was extracted verbatim from the "PIPELINE STAGE" cell of
`notebooks/ranking_rf_final.ipynb`, which is kept unchanged for provenance and exploration but
is no longer executed by the pipeline (it needed `nbconvert --allow-errors`, which hid real
failures, plus a kernelspec workaround). Verified equivalent: identical feature importances and
design ranking order.

Notes on the metric set and reporting:

- **Filtering-only metrics.** Developability and energy metrics (SAP, net charge,
  surface hydrophobicity, unsat H-bonds, interface ΔG, ESM3 ΔG, pyRosetta score) are
  kept in the data and used by the feasibility filtering gates (energy mainly for
  nanobodies), but are **excluded from the eval / RF / composite feature set**
  (`config.analysis.FILTER_ONLY_CATEGORIES`, enforced in `get_metric_columns`).
- **`mol_type`.** Each sample carries a `nanobody`/`antibody` label (from the dataset
  config), so benchmarks are reported pooled, per-dataset, and per-mol_type.
- **MCC** is reported only in the RF CV metrics (`mcc@0.5`, on out-of-fold predictions).
  It is deliberately not a single-metric benchmark column: at an in-sample F1-optimal
  threshold on a balanced benchmark it adds nothing over PR-AUC / precision@10%.
- **Thresholds.** `run_thresholds.py` reports the F1-optimal cutoff against each metric's
  literature reference value (`threshold_report.csv`, plus by-category and by-dataset
  breakdowns). Threshold selection is in-sample on the benchmark and therefore optimistic.
  It is report-only: no stage reads its output, and the filter's cutoffs come from
  `config/thresholds.yaml` instead.

Figures are produced from the notebooks at the repository root
(`distributions_plots.ipynb`, `evaluation_plots.ipynb`, `ranking_plots.ipynb`,
`profiles_plots.ipynb`, `composite_plot.ipynb`). These are optional and run manually.

---

## 5. Reproducibility (per-run provenance)

`config/provenance.py` records exactly what a run used. Creating a run directory
snapshots the whole `config/` folder and writes a `manifest.json` with the selected
dataset (and its members), the resolved paths, which environment overrides were set,
the Python and package versions, the platform, and a git commit if the project is a
git checkout. Each stage can append a record of what it produced.

```python
from config.provenance import new_run_dir, record_stage

run = new_run_dir()                                   # runs/<dataset>_<timestamp>/
# ... run a stage ...
record_stage(run, "evaluation", outputs=["rankings.csv"])
```

or from the shell:

```bash
PYTHONPATH=. python -m config.provenance             # prints a new run directory
```

For full reproducibility, keep `runs/<...>/manifest.json` together with the outputs,
and commit a `requirements.lock.txt` (see Setup) for exact package versions. This
project is not yet a git repository; initialising one is recommended so the manifest
can record the exact commit.

---

## 6. Repository layout (top level)

```
config/         paths, dataset definitions, metric metadata & thresholds, provenance
                metric_meta.py = the single source for directions/families/
                categories/thresholds (canonical loaders + get_* helpers)
preprocessing/  data prep, standardize, align, scale, metadata/labelling, metric_meta
analysis/
  distributions/  PROFILING: distributions, model agreement, qc.py (integrity),
                  profiles.py (sample/outlier/threshold views), plots.py
  evaluation/     BENCHMARKING: single-metric benchmark (canonical rankings.csv),
                  effect sizes, redundancy, pass_mask/precision_recall_at, plots.py
  composite/      BENCHMARKING: composite development (dataset-aware selection + grouped CV)
  feasibility/    FILTERING: developability gates + responsiveness (library)
  ranking/        RANKING: RF+composite consensus, plots.py (RF / composite / consensus figures)
  plot_common.py  figure helpers shared by evaluation/plots.py and ranking/plots.py
  figures.py      opt-in figure auto-save (set_figure_dir)
  io.py           load_processed_datasets / load_eval / load_rankings / rankings_for
run_*.py        one script per pipeline stage, in run_all.sh order: data_prep, align,
                scale, qc, profiling, evaluation, composite, ranking_rf, filter, consensus
                (run_thresholds.py and run_rf_importance.py are report-only, not stages)
notebooks/      legacy prep notebooks + the reference RF notebook + experiments
*_plots.ipynb   figure notebooks (root): distributions, evaluation, ranking, profiles, composite
tests/          pytest suite
grouped_pyros/  upstream PyRosetta developability scripts (input generation)
```

---

## 8. Walkthrough for a new user

**Configure & run.** Install dependencies (`pip install -r requirements.txt`).
Configuration is environment variables (or a `.env` at the repo root):
`PROJECT_ROOT` / `DATA_DIR` / `RUNS_DIR` (default to this repo), `EXTERNAL_DATA_ROOT`
(where the raw prediction CSVs live), and `DATASET` — a single dataset (e.g.
`alphaseq`) or a pooled set from `config/analysis.py` `ANALYSIS_SETS` (e.g. `rf`).
Optional: `SCALER` (`standard`/`robust`), `PRECISION_TARGET` / `N_MIN` (filter
threshold), `RUN_RF=0` to skip the RF notebook. Run everything in order:

```bash
DATASET=rf ./run_all.sh
PYTHONPATH=. DATASET=rf python validate_pipeline.py   # check the outputs
```

or run one stage: `PYTHONPATH=. python run_<stage>.py`.

**Where the outputs are** (under `data/`):

- base tables: `data/raw/<dataset>/{merged,eval,design}.csv` (+ `_aligned`, `_scaled_aligned`)
- QC: `data/qc/<dataset>/` ; profiling: `data/processed/profiling/<dataset>/`
- benchmark & downstream: `data/processed/evaluation/<dataset>/` — `rankings.csv`,
  effect sizes, `rankings_by_dataset.csv`, `rankings_by_mol_type.csv`, and the
  subfolders `thresholds/`, `composite/`, `ranking/`, `filter/`, `consensus/`
- provenance per run: `runs/<dataset>_<timestamp>/` (config snapshot + `manifest.json` + logs)

**Understanding the results.** *Filtering* (`.../filter/`): `filter_funnel_eval.csv`
gives precision/recall/N at each funnel step on labelled data (quality → developability
gates), `filter_funnel_design.csv` the N retained on designs,
`filtered_designs.csv` the per-design pass flags, `feasibility_summary.csv` the
developability gate fails per model. *Ranking* (`.../consensus/`): `shortlist.csv` is the
top-k designs by the **RF** (primary) among those that PASS the feasibility filter
(filter-aware), each flagged `consensus` (composite agrees) or `primary_only`; `disagreements.csv` lists where the two methods disagree; `agreement.txt`
has the Spearman/Kendall agreement. The product-composite is complementary — there is no
blended score.

**Tracing one metric** (e.g. `af3_iptm`): its metadata is one line in
`config/metric_data.yaml` (`direction`, `family`, `category`, `scale`); direction
alignment is `preprocessing/align.py` (via `metric_meta.get_direction`); it is selected
as a feature by `metric_meta.get_metric_columns` (interface, not filtering-only); scored
in `run_evaluation.py` → `calculate_all_metrics` → a row in `rankings.csv` (pr_auc,
precision_at_pct, aligned_roc, opt_threshold_raw); its derived-vs-reference cutoff is in
`thresholds/threshold_report.csv`; its gate is `config/thresholds.yaml`; and it appears as
an RF feature in `ranking/rf_importances.csv`.
