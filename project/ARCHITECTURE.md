# Pipeline architecture

Diagrams of what the code actually does, for inspection.


---

## 1. Stage flow

Ten stages, run in order by `run_all.sh`, fail-fast. Blue = produces inputs for later stages,
grey = **report only** (nothing downstream reads it).

```mermaid
flowchart TD
    PRED["predictions CSV<br/>per dataset"] --> DP
    MT["mt_DATASET.csv<br/>labels + metadata"] --> DP

    DP["1 · data_prep"] --> MERGED["merged.csv"]
    DP --> EVAL["eval.csv<br/>labeled"]
    DP --> DESIGN["design.csv<br/>designs"]

    MERGED --> AL["2 · align"]
    EVAL --> AL
    DESIGN --> AL
    AL --> ALIGNED["merged/eval/design_aligned.csv<br/>all metrics direction-aligned"]

    ALIGNED --> SC["3 · scale"]
    SC --> SCALED["merged_scaled_aligned.csv<br/>ONE fit, on the LABELED rows"]

    MERGED --> QC["4 · qc"]
    MERGED --> PROF["5 · profiling"]
    QC --> QCOUT["qc_report · missing_by_sample<br/>nonfinite_report"]
    PROF --> PROFOUT["composition · class_distribution<br/>missingness · metric_ranges"]

    EVAL --> EV["6 · evaluation"]
    ALIGNED --> EV
    EV --> RANK["rankings.csv<br/> sirviving metrics · in_model_set flag"]
    EV --> EVOUT["by_dataset · by_mol_type · cliffs_delta<br/>cohens_d · auroc_pvalue · spearman_correlation"]

    EVAL --> CO["7 · composite"]
    RANK --> CO
    SCALED --> CO
    CO --> COOUT["composite_cv_eval · cv_by_fold<br/>selected_metrics · composite_oof_scores<br/>quality_flags"]
    CO --> CSCORES["composite_scores.csv<br/>consensus · weighted · product"]

    EVAL --> RF["8 · ranking_rf"]
    DESIGN --> RF
    RF --> RFOUT["rf_cv_metrics · cv_by_fold<br/>rf_per_source · rf_oof_scores<br/>rf_importances"]
    RF --> RFSCORES["rf_design_scores.csv<br/>p_binder"]

    DESIGN --> FI["9 · filter"]
    FI --> FIOUT["filter_funnel_design<br/>feasibility_summary"]
    FI --> FFLAGS["filtered_designs.csv<br/>passes_filter"]

    RFSCORES --> CN["10 · consensus"]
    CSCORES --> CN
    FFLAGS --> CN
    CN --> CNOUT["consensus_scores · shortlist<br/>disagreements · agreement.txt"]

    classDef report fill:#eee,stroke:#999,color:#333
    class QCOUT,PROFOUT,EVOUT,COOUT,RFOUT,FIOUT report
```

Note `qc` and `profiling` are terminal: they exist to characterise the data, and **nothing is built
on them**. `evaluation`'s `rankings.csv` is the one descriptive artefact that *is* reused (by
`composite`, via `analysis.io.rankings_for`), so the single-metric benchmark is computed once.


---

## 2. The two metric sets

The pipeline works with two column sets and every stage picks one **on purpose**. Running a
preselection analysis on the already-filtered set would be circular: it is the evidence used to
decide what to exclude.

```mermaid
flowchart LR
    YAML["metric_data.yaml<br/>direction · family · category<br/>reference"] --> ALLSET
    ALLSET["get_all_metric_columns<br/><b>86 metrics</b>"] -->|"minus FILTER_ONLY_CATEGORIES<br/>developability · energy · sequence"| PREDSET["get_metric_columns<br/><b>46 predictors</b>"]

    ALLSET --> D1["align · scale"]
    ALLSET --> D2["qc · profiling"]
    ALLSET --> D3["evaluation<br/>rankings.csv 83 rows"]
    ALLSET --> D4["thresholds<br/><i>on-demand report</i>"]
    ALLSET --> D5["redundancy<br/>spearman · PCA · UMAP"]

    PREDSET --> F1["composite candidates"]
    PREDSET --> F2["RF features"]

    classDef desc fill:#e8f0ff,stroke:#5a7,color:#123
    classDef fit fill:#ffe8e8,stroke:#a55,color:#311
    class D1,D2,D3,D4,D5 desc
    class F1,F2 fit
```

Blue = descriptive / preselection, red = **fitted**. `rankings.csv` covers 83 of them (the 86 less the two-sided window families) and marks
membership with `in_model_set`, so the exclusions are evidenced rather than asserted.
`validate_pipeline.py` asserts predictor ⊂ full and that no filter-only category leaks into the
predictor set.

---

## 3. Configuration

Metric **metadata** and design-**filter** configuration are deliberately separate files, because
windowness is derived from the metadata and feeds the composite candidate pool.

```mermaid
flowchart LR
    MD["metric_data.yaml<br/>metrics:<br/>direction · family · category<br/>reference"]
    TH["thresholds.yaml<br/>quality: S1 (currently empty)<br/>gates: S2 with own numbers"]
    AN["config/analysis.py<br/>MODELS · ANALYSIS_SETS<br/>FILTER_ONLY_CATEGORIES<br/>PRECISION_AT_PERCENT"]

    MD --> L1["load_directions<br/>align"]
    MD --> L2["load_families / load_categories<br/>metric sets · families"]
    MD --> L3["load_reference_values<br/>threshold report"]
    MD --> L4["load_window_families<br/>list-valued reference = band"]

    TH --> L5["load_quality<br/>S1 screen (columns: {} -- no-op)"]
    TH --> L6["load_gates<br/>S2 gates"]

    L4 --> W1["excluded from<br/>monotone benchmark"]
    L4 --> W2["excluded from<br/>composite candidates"]
    L5 --> Q1["run_filter · run_composite"]
    L6 --> Q2["run_filter only"]
    AN --> A1["all stages"]
```

---

## 4. Ranking vs filtering — three artefacts

Filtering never changes the ranking; it only selects from it at the last step. In increasing
restriction:

```mermaid
flowchart TD
    RF["ranking_rf<br/>RF trained on ALL labeled rows"] --> A1["ranking/rf_design_scores.csv<br/><b>every design, unfiltered</b>"]
    CO["composite<br/>product score"] --> A2
    A1 --> A2["consensus/consensus_scores.csv<br/><b>complete ranking + filter flags ATTACHED</b>"]
    FI["filter<br/>design set only"] --> FLAGS["filtered_designs.csv<br/>passes_filter · developability_status<br/>n_gates_seen"]
    FLAGS --> A2
    A2 --> A3["consensus/shortlist.csv<br/><b>top-k RF among passes_filter</b>"]

    classDef final fill:#dff0d8,stroke:#3c763d,color:#1a3c1a
    class A3 final
```

Two models are kept distinct inside `ranking_rf`: the lineage-grouped **CV models**, used only to
estimate performance, and the **final model** refit on all labeled rows, used only to score
designs. Filtering is **design-only** because the labeled benchmark pools antibody and nanobody
complexes, whose interfaces are not comparable.

How the CV numbers are read is itself a decision: **per-source columns are primary, pooled
columns are diagnostics.** `ap_norm_ds_mean` / `ap_norm_ds_min` normalise each source by its own
prevalence and then aggregate, because the four labeled sources differ in prevalence enough
that a pooled top-10% can be filled from one of them. Both the composite and the RF report it
through the same helper (`analysis.composite.per_source_metrics`) so the two are comparable.


---

## 5. Leakage boundaries

```mermaid
flowchart LR
    subgraph safe["fitted per TRAINING FOLD · lineage-grouped StratifiedGroupKFold"]
        S1["composite: standardisation +<br/>per-source ap_norm weights<br/>fit per training fold"]
        S2["RF: refit per fold<br/>pooled out-of-fold predictions"]
    end
    subgraph lab["fitted on ALL LABELED rows"]
        S3["scaler · run_scale"]
        S4["composite metric selection<br/>deterministic, no folds"]
        S5["product-score standardisation"]
    end
    subgraph desc["descriptive · in-sample by design, report only"]
        D1["single-metric benchmark"]
        D2["effect sizes · redundancy"]
    end
    subgraph final["fitted on ALL labeled data<br/>used ONLY to score designs"]
        F1["final RF"]
        F2["final composite"]
    end
```

Lineage groups are `sample.split('_')[0]`; lineage groups over labeled rows. Because lineages nest
inside datasets, the grouped folds are close to **leave-one-source-out**, so the per-fold spread in
`cv_by_fold.csv` is cross-dataset transfer rather than sampling noise.

Known, deliberate exception: metric selection is **not** fold-local. `dataset_aware_select` ranks
families once, on all labeled rows, by prevalence-corrected PR-AUC averaged across sources
deterministically (with no folds and no seed). So the composite's cross-validated numbers are
out-of-fold *given that metric set*, not independent estimates. The RF has no selection step 
(fixed, label-independent features), so its estimate carries no selection bias.


---

## 6. Figure layer

Figures are a separate layer: no stage draws one, and no stage imports matplotlib. Each plot
function takes a pipeline output and renders it, so a figure cannot disagree with the number it
is drawn from.

| module | figures | driven by |
|---|---|---|
| `analysis/distributions/plots.py` | distributions, separability, model agreement, benchmark structure | `distributions_plots.ipynb` |
| `analysis/distributions/profiles.py` | sample / outlier / threshold views (notebook-only) | `profiles_plots.ipynb` |
| `analysis/evaluation/plots.py` | single-metric benchmark: metric bars, ROC/PR curves, quadrant, embeddings, Spearman heatmap | `evaluation_plots.ipynb` |
| `analysis/ranking/plots.py` | RF performance and importances, design scores, RF-vs-composite comparison | `ranking_plots.ipynb` |
| `analysis/plot_common.py` | `_finish` + the shared metric bar-chart machinery | (imported by the two above) |
| *(none)* | composite variants: `composite_plot.ipynb` draws inline from `analysis/composite` + `config/palette`, and writes the `variants/` tables below | `composite_plot.ipynb` |


Color comes from `config/palette.py` and follows the ENTITY, never the position in the figure,
so dropping a series never repaints the survivors. `analysis/figures.py` adds opt-in auto-save:
until a notebook calls `set_figure_dir()`, nothing is written to disk.

