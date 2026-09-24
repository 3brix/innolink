"""
Metric redundancy structure.

Treats each METRIC (column) as a point and looks at how metrics relate:
a Spearman correlation matrix (rank-based, robust to non-linear monotonic
relationships) plus PCA and UMAP embeddings of the metrics. Coloring the
embeddings by family or model shows whether the reduction recovers the
expected redundancy (siblings of the same family clustering together).

Expects a direction-ALIGNED table (higher = better): on the raw table a
lower-is-better metric and a higher-is-better metric measuring the same thing
come out anti-correlated (opposite ends of the embedding) despite being
redundant, so alignment is what makes genuine redundancy cluster.

Compute only -- the figures live in plots.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from config.paths import METRIC_YAML
from analysis.distributions.wide import get_numeric_metrics
from analysis.distributions.long import assign_model
from analysis.evaluation.metrics import load_metric_families, get_family


def prepare_metric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric metric columns, inf -> NaN -> median."""
    X = df[get_numeric_metrics(df)].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median())


def spearman_correlation(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Spearman correlation matrix among metrics / metric subset."""
    X = prepare_metric_matrix(df)
    if metrics is not None:
        X = X[[m for m in metrics if m in X.columns]]
    return X.corr(method="spearman")


def label_metrics(metric_names: list[str], color_by: str = "family", metric_yaml=METRIC_YAML) -> pd.Series:
    """ Map each metric to its YAML 'family' or its 'model'. Metrics absent from the YAML fall back to their own name."""
    if color_by == "model":
        values = [assign_model(m) or "other" for m in metric_names]
    else:
        families = load_metric_families(metric_yaml)
        values = [get_family(m, families) or m for m in metric_names]
    return pd.Series(values, index=metric_names)


def _scaled_metric_features(df: pd.DataFrame):
    """Standardized metric-as-row feature matrix; returns (metric_names, X_scaled)."""
    X_feat = prepare_metric_matrix(df).T  # rows = metrics, cols = samples
    return list(X_feat.index), RobustScaler().fit_transform(X_feat.values)


def pca_embedding(df: pd.DataFrame, random_state: int = 42):
    """2D PCA of the metrics"""
    names, X_scaled = _scaled_metric_features(df)
    pca = PCA(n_components=2, random_state=random_state)
    emb = pca.fit_transform(X_scaled)
    return emb, names, pca.explained_variance_ratio_


def umap_embedding(df: pd.DataFrame, n_neighbors: int = 10, min_dist: float = 0.1, random_state: int = 42):
    """2D UMAP of the metrics. Needs umap-learn."""
    from umap import UMAP  # heavy optional dependency, imported lazily
    names, X_scaled = _scaled_metric_features(df)
    emb = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).fit_transform(X_scaled)
    return emb, names