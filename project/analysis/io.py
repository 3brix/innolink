import pandas as pd
from pathlib import Path


def load_processed_datasets(datasets, root):
    """Pool each dataset's merged.csv, adding a `dataset` column. `source` is left intact."""
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


def load_rankings(cfg):
    """run_evaluation's rankings.csv, or None if it has not been produced yet."""
    from config.paths import EVALUATION_DIR
    p = Path(EVALUATION_DIR) / cfg.name / "rankings.csv"
    return pd.read_csv(p) if p.exists() else None


def rankings_for(cfg, df, metrics, directions):
    """Benchmark rows for `metrics` from rankings.csv, recomputed if that file is absent."""
    rk = load_rankings(cfg)
    if rk is not None and "metric" in rk.columns:
        return rk[rk["metric"].isin(metrics)].reset_index(drop=True)
    from analysis.evaluation import calculate_all_metrics
    return calculate_all_metrics(df, metrics, directions)