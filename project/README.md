# Binder benchmarking & design-ranking pipeline

This pipeline takes structure-prediction metrics for antibody / nanobody designs,
**benchmarks which metrics separate binders from non-binders**, and then uses those
findings to **filter and rank new binder designs**. It grew out of a set of Jupyter
notebooks and is now organised into small, readable Python stage-scripts
(`run_*.py`); the remaining notebooks serve only to inspect figures.

Two labelled tables drive everything:

- `eval.csv` — designs with a known label (`binder` is 0/1), used to *benchmark* metrics.
- `design.csv` — new designs with unknown label (`binder == "?"`), the ones we *rank*.

This file is the how-to-run guide. Two companion documents carry the rest:

- `PIPELINE_SPEC.md` — per-stage contracts (reads / writes / invariants), the configuration
  reference, and the **current headline numbers**.
- `ARCHITECTURE.md` — the narrative and the per-stage diagrams.

---

## 1. Setup

You need Python 3.10 or newer.

```bash
# from this folder (the repository root)
python -m venv .venv && source .venv/bin/activate      # or a conda env
pip install -r requirements.txt

```

---

## 2. Configuration (paths & dataset selection)

All locations derive from a single `PROJECT_ROOT`. By default it is **this repository
folder**, so a fresh checkout runs with no configuration at all: `data/` and `runs/`
are expected next to the code.

Override any of these with environment variables, or by writing them as `KEY=VALUE`
lines into a `.env` file at the repository root (real environment variables win over
the `.env` file):

| Variable | Meaning | Default |
|---|---|---|
| `PROJECT_ROOT` | root everything derives from | the repository folder |
| `DATA_DIR` | where `raw/`, `processed/`, `qc/` live | `<PROJECT_ROOT>/data` |
| `RUNS_DIR` | where per-run provenance is written | `<PROJECT_ROOT>/runs` |
| `EXTERNAL_DATA_ROOT` | root holding the raw prediction CSVs | `.../project/` (a placeholder — set it) |
| `DATASET` | which dataset or pooled set to run | `rf` |
| `SCALER` | `standard` or `robust` (used by `run_scale.py`) | `standard` |
| `SHORTLIST_K` | shortlist size (`run_consensus.py`) | `25` |
| `COMPOSITE_SCORE` | which composite column `run_consensus.py` compares against | `composite_product` |

`DATASET` picks either one dataset or a pooled **analysis set** (several datasets
combined). The datasets and sets are defined in `config/datasets.py` and
`config/analysis.py`. `EXTERNAL_DATA_ROOT` only matters for `run_data_prep.py`, which
reads the raw prediction CSVs; the committed cluster paths were replaced by a
placeholder, so set it to wherever those CSVs live. Everything after `data_prep` runs
from the tables under `DATA_DIR`.

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
to `data/processed/evaluation/<dataset>/ranking/`:

| file | holds |
|---|---|
| `rf_design_scores.csv` | `sample`, `dataset`, `p_binder` — the design ranking, consumed by `run_consensus.py` |
| `rf_cv_metrics.csv` | the lineage-grouped CV metrics (see below) |
| `rf_per_source.csv` | the per-source rows the `_ds_` aggregates are computed from |
| `cv_by_fold.csv` | per fold: n, prevalence, PR-AUC, enrichment, ROC-AUC, composition |
| `rf_oof_scores.csv` | the per-sample out-of-fold predictions the AUCs come from, so the PR / ROC curves match the reported numbers |
| `rf_importances.csv` | impurity importances of the final model |

Needs scikit-learn >= 1.4. Two models are kept distinct: the lineage-grouped CV models used
only to estimate performance, and the final model refit on all labelled rows used only to
score designs. (`run_rf_importance.py` is a separate, report-only permutation-importance run;
it writes `rf_importance_comparison.csv` into the same folder.)

**Per-source columns are the primary numbers; pooled columns are diagnostics.** The labelled
benchmark pools four sources with different prevalences, and a pooled top-10% can be filled
entirely from one of them. So read `ap_norm_ds_mean` — (PR-AUC − prevalence) / (1 − prevalence)
computed *within* each source, then averaged, where 0 is no-skill — together with
`ap_norm_ds_min`, the worst source, as the robustness check. `pr_auc_oof`, `roc_auc_oof` and
`precision_at_10pct` are pooled out-of-fold, kept for comparison with the composite and the
single-metric benchmark, which report the same pooled estimator. `pr_auc_fold_mean` is a
different quantity again (folds differ in prevalence) and is spread information only — don't
compare it across methods. Current figures for all rankers are tabulated in `PIPELINE_SPEC.md` §7.


Notes on the metric set and reporting:

- **Filtering-only metrics.** Three categories are kept in the data but **excluded from the
  eval / RF / composite feature set** (`config.analysis.FILTER_ONLY_CATEGORIES`, enforced in
  `preprocessing.metric_meta.get_metric_columns`): `developability` and `energy` (SAP, net
  charge, surface hydrophobicity, unsat H-bonds, interface ΔG, ESM3 ΔG, pyRosetta score), which
  feed the feasibility gates instead, and `sequence` (MPNN), which is present for only 60 of
  the 133 labelled samples on a 62%-positive subset — it ranked near the top on PR-AUC at an
  ROC-AUC of 0.56, i.e. chance, so it is treated as a filtering signal too. That leaves 46
  predictor metrics out of the 86 annotated numeric columns present for `rf`.
- **`mol_type`.** Each sample carries a `nanobody`/`antibody` label (from the dataset
  config), so benchmarks are reported pooled, per-dataset, and per-mol_type.
- **No F1 / MCC anywhere.** Both are threshold-dependent at a point the ranking task never
  uses, and at an in-sample F1-optimal threshold they add nothing over PR-AUC / precision@10%.
  They are not columns in `rankings.csv` (asserted by `tests/test_threshold_select.py`) and the
  RF stage no longer reports `mcc@0.5` either.
- **The composite** (`run_composite.py`) is the *complementary* ranker, never blended into the
  RF score. Its metric set is chosen deterministically on all labelled rows (best in family by
  cross-source `ap_norm`, with a below-baseline override) — it is **not** voted across folds,
  since the lineage-grouped folds are near single-source and voting on them would leak. The
  selection is recorded in `composite/selected_metrics.csv`; only the standardisation and the
  weights are fitted per training fold, for the reported CV numbers. Note the asymmetry when
  comparing it against the RF: the composite's metric set saw all labelled rows before the
  folds were drawn, so its CV numbers are out-of-fold *given that set*, while the RF's feature
  set is fixed and label-independent. The asymmetry favours the composite.
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
and commit a `requirements.lock.txt` for exact package versions. The project
is a git checkout, so the manifest records the exact commit as well.

The environment matters more than usual here: `StratifiedGroupKFold` partitions the
lineage groups differently across scikit-learn versions, so every out-of-fold number shifts
if the stack changes. The recorded environment is Python 3.11.14, numpy 2.4.2, pandas 3.0.1,
scipy 1.17.1, scikit-learn 1.8.0.

---

## 6. Repository layout (top level)

```
config/         paths, dataset definitions, provenance, palette, and the two config files:
                metric_data.yaml (per-metric direction/family/category/scale/reference)
                and thresholds.yaml (the filter's S1 quality + S2 gate blocks)
preprocessing/  data prep, standardize, align, scale, metadata/labelling, and
                metric_meta.py = the single source for directions/families/categories/
                thresholds (canonical loaders + get_* helpers)
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
plans_audits_reports/   working notes, plans and audits (history, not a contract)
```

The upstream PyRosetta developability scripts that generate the prediction CSVs are not in
this repository.

---

## 7. Walkthrough for a new user

**Configure & run.** Install dependencies (`pip install -r requirements.txt`).
Configuration is environment variables (or a `.env` at the repo root):
`PROJECT_ROOT` / `DATA_DIR` / `RUNS_DIR` (default to this repo), `EXTERNAL_DATA_ROOT`
(where the raw prediction CSVs live), and `DATASET` — a single dataset (e.g.
`alphaseq`) or a pooled set from `config/analysis.py` `ANALYSIS_SETS` (e.g. `rf`).
Optional: `SCALER` (`standard`/`robust`), `SHORTLIST_K`, `COMPOSITE_SCORE`. Every stage is a
plain Python script, so a failure stops the run. Run everything in order:

```bash
DATASET=rf ./run_all.sh
PYTHONPATH=. DATASET=rf python validate_pipeline.py   # 17 checks on the outputs
PYTHONPATH=. python -m pytest -q                      # 27 tests
```

or run one stage: `PYTHONPATH=. python run_<stage>.py`.

**Where the outputs are** (under `data/`):

- base tables: `data/raw/<dataset>/{merged,eval,design}.csv` (+ `_aligned`, `_scaled_aligned`)
- QC: `data/qc/<dataset>/` ; profiling: `data/processed/profiling/<dataset>/`
- benchmark & downstream: `data/processed/evaluation/<dataset>/` — `rankings.csv`,
  effect sizes, `rankings_by_dataset.csv`, `rankings_by_mol_type.csv`, and the
  subfolders `thresholds/`, `composite/`, `ranking/`, `filter/`, `consensus/`
- provenance per run: `runs/<dataset>_<timestamp>/` (config snapshot + `manifest.json` + logs)
- figures, when a notebook opts in with `analysis.figures.set_figure_dir`: `data/figures/`

**Understanding the results.** *Filtering* (`.../filter/`) runs on the **design set only**:
the labelled benchmark pools antibody and nanobody complexes, whose interfaces differ in size,
so the PyRosetta developability metrics are not comparable across it and a funnel computed
there would compare unlike things. `filter_funnel_design.csv` gives the N retained at each
step (S0 all → S1 quality → S2 developability gates) plus how many rows S2 could judge,
`filtered_designs.csv` the per-design flags (`quality_ok`, `developability_status`,
`n_gates_seen`, `passes_filter`), `feasibility_summary.csv` the gate fails per model. Note
that S1 is currently off by decision — `thresholds.yaml` has `quality: columns: {}`, so all
designs pass S1 and S2 does the work. *Ranking* (`.../consensus/`): `shortlist.csv` is the
top-k designs by the **RF** (primary) among those that PASS the feasibility filter
(filter-aware), each flagged `consensus` (composite agrees) or `primary_only`; `disagreements.csv` lists where the two methods disagree; `agreement.txt`
has the Spearman/Kendall agreement. The product-composite is complementary — there is no
blended score.

**Tracing one metric** (e.g. `af3_iptm`): its metadata is one line in
`config/metric_data.yaml` (`direction`, `family`, `category`, `scale`); direction
alignment is `preprocessing/align.py` (via `metric_meta.get_direction`); it is selected
as a feature by `metric_meta.get_metric_columns` (interface, not filtering-only); scored
in `run_evaluation.py` → `calculate_all_metrics` → a row in `rankings.csv` (pooled `pr_auc`,
`precision_at_pct`, `aligned_roc`, `opt_threshold_raw`, plus the per-source `ap_norm_ds_mean` /
`ap_norm_ds_min` and a per-dataset row in `rankings_by_dataset.csv`); its derived-vs-reference cutoff is in
`thresholds/threshold_report.csv`; its gate is `config/thresholds.yaml`; and it appears as
an RF feature in `ranking/rf_importances.csv`.
