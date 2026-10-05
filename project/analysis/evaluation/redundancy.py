"""Metric redundancy structure: Spearman matrix + PCA / UMAP embeddings of the metrics.

Expects a direction-ALIGNED table (higher = better) -- on the raw table two metrics measuring
the same thing in opposite directions look complementary. Compute only; figures in plots.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config.paths import METRIC_YAML
from analysis.distributions.wide import get_all_numeric_metrics
from preprocessing.metric_meta import get_metric_columns
from analysis.distributions.long import assign_model
from analysis.evaluation.metrics import load_metric_families, get_family


def metric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric metrics with inf -> NaN, no imputation. Full set on purpose: restricting it to
    the already-kept metrics would be circular."""
    return df[get_all_numeric_metrics(df)].replace([np.inf, -np.inf], np.nan)


def prepare_metric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """metric_matrix + median imputation, for PCA / UMAP only.

    Never for correlations: the imputed ties drag Spearman toward zero (one pair measured -0.53
    pairwise-complete, -0.27 imputed), understating redundancy where coverage is worst."""
    X = metric_matrix(df)
    return X.fillna(X.median())


def spearman_correlation(df: pd.DataFrame, metrics: list[str] | None = None,
                         min_periods: int = 30, model_set_only: bool = True) -> pd.DataFrame:
    """Spearman matrix on pairwise-complete observations. Rank-based, so no scaling.

    `min_periods` is the minimum overlap a pair needs; below it the entry is NaN rather than a
    correlation from a handful of rows (coverage ranges ~45-100%). `model_set_only` (default)
    keeps the predictor set -- excluded by category a priori, so not the circularity
    metric_matrix warns about. An explicit `metrics` list is used as given."""
    if metrics is None and model_set_only:
        X = df[get_metric_columns(df)].replace([np.inf, -np.inf], np.nan)
    else:
        X = metric_matrix(df)
    if metrics is not None:
        X = X[[m for m in metrics if m in X.columns]]
    return X.corr(method="spearman", min_periods=min_periods)


def label_metrics(metric_names: list[str], color_by: str = "category", metric_yaml=METRIC_YAML) -> pd.Series:
    """Map each metric to its 'category' (default), 'family' or 'model'.

    'category' is the default because ~34 families overflow the palette's colour slots; use
    'family' only with direct labels, facets, or the paired family styling in plot_embedding."""
    if color_by == "model":
        from config.palette import model_family
        values = [model_family(assign_model(m) or "other") for m in metric_names]
    elif color_by == "category":
        from preprocessing.metric_meta import load_categories, get_category
        categories = load_categories(metric_yaml)
        values = [get_category(m, categories) or "other" for m in metric_names]
    else:
        families = load_metric_families(metric_yaml)
        values = [get_family(m, families) or m for m in metric_names]
    return pd.Series(values, index=metric_names, name=color_by)   # name -> plot_embedding legend title


def _scaled_metric_features(df: pd.DataFrame, rank: bool = False, model_set_only: bool = False):
    """Metric-as-row feature matrix, standardised PER METRIC -> (metric_names, X_scaled).

    Standardise column-wise then transpose: scaling the transposed matrix would standardise each
    SAMPLE across metrics and leave the per-metric scales alone, so PCA would just rank metrics
    by magnitude (per-metric std spans 229,000x). rank=True ranks first -- outlier-proof and
    consistent with the Spearman view. model_set_only keeps the predictor set."""
    X = prepare_metric_matrix(df)
    if model_set_only:
        from preprocessing.metric_meta import get_metric_columns
        keep = set(get_metric_columns(df))
        X = X[[c for c in X.columns if c in keep]]
    if rank:
        X = X.rank(axis=0)
    names = list(X.columns)
    return names, StandardScaler().fit_transform(X.values).T


def pca_embedding(df: pd.DataFrame, random_state: int = 42, rank: bool = False, model_set_only: bool = True):
    """2D PCA of the metrics; variance-based, so per-metric standardisation is required.

    `model_set_only` (default, as in spearman_correlation) embeds the predictor set only."""
    names, X_scaled = _scaled_metric_features(df, rank=rank, model_set_only=model_set_only)
    pca = PCA(n_components=2, random_state=random_state)
    emb = pca.fit_transform(X_scaled)
    return emb, names, pca.explained_variance_ratio_


def umap_embedding(df: pd.DataFrame, n_neighbors: int = 10, min_dist: float = 0.1,
                   random_state: int = 42, rank: bool = False, model_set_only: bool = True):
    """2D UMAP of the metrics; distance-based, so per-metric standardisation is required."""
    from umap import UMAP  # optional dependency
    names, X_scaled = _scaled_metric_features(df, rank=rank, model_set_only=model_set_only)
    emb = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).fit_transform(X_scaled)
    return emb, names