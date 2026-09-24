import pandas as pd
from pathlib import Path


def load_processed_datasets(datasets, root):
    """
    Loads each dataset's "merged.csv" and concatenate into one pooled frame.
    Adds a "dataset" column (= the member name) so pooled analyses can group / colour by it. 
    The existing "source" column (the within-dataset source, used as hue in the separability / agreement plots) is left intact.
    """
    from config.datasets import DATASETS
    dfs = []
    for name in datasets:
        df = pd.read_csv(Path(root) / name / "merged.csv")
        df["dataset"] = name
        mol = DATASETS[name].mol_type if name in DATASETS else None
        if mol is not None and "mol_type" not in df.columns:
            df["mol_type"] = mol          # per-member molecule type for pooled reporting
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Convenience loaders (remove repeated eval/design/rankings)
def _base(cfg):
    from config.paths import RAW_DATA_DIR
    return Path(RAW_DATA_DIR) / cfg.name


def load_eval(cfg) -> pd.DataFrame:
    """Labelled evaluation rows for a dataset/set: eval.csv, else split from merged.csv."""
    from preprocessing.metadata import split_eval_design
    base = _base(cfg)
    p = base / "eval.csv"
    if p.exists():
        return pd.read_csv(p)
    return split_eval_design(pd.read_csv(base / "merged.csv"))[0]


def load_design(cfg):
    """Unlabelled design rows: design.csv, else split from merged.csv, else None."""
    from preprocessing.metadata import split_eval_design
    base = _base(cfg)
    p = base / "design.csv"
    if p.exists():
        return pd.read_csv(p)
    m = base / "merged.csv"
    return split_eval_design(pd.read_csv(m))[1] if m.exists() else None


def load_rankings(cfg):
    """The canonical single-metric benchmark (run_evaluation's rankings.csv), or None
    if it hasn't been produced yet."""
    from config.paths import EVALUATION_DIR
    p = Path(EVALUATION_DIR) / cfg.name / "rankings.csv"
    return pd.read_csv(p) if p.exists() else None


def rankings_for(cfg, df, metrics, directions):
    """The single-metric benchmark for `metrics`: rows from the canonical
    rankings.csv when it exists (benchmark computed once by run_evaluation),
    otherwise computed here as a fallback so the stage still runs standalone."""
    rk = load_rankings(cfg)
    if rk is not None and "metric" in rk.columns:
        return rk[rk["metric"].isin(metrics)].reset_index(drop=True)
    from analysis.evaluation import calculate_all_metrics
    return calculate_all_metrics(df, metrics, directions)
