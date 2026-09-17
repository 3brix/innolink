import pandas as pd
from pathlib import Path


def load_processed_datasets(datasets, root):
    """
    Loads each dataset's "merged.csv" and concatenate into one pooled frame.
    Adds a "dataset" column (= the member name) so pooled analyses can group / colour by it. 
    The existing "source" column (the within-dataset source, used as hue in the separability / agreement plots) is left intact.
    """
    dfs = []
    for name in datasets:
        df = pd.read_csv(Path(root) / name / "merged.csv")
        df["dataset"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)