"""
Regression tests for dataset / pooled-set resolution in config.datasets.

Importing config.datasets with the default DATASET runs its collision and
member-validity checks, so a clean import is itself part of the guard.
Run with:  python -m pytest tests/ -q   (from project_current/)
"""

import config.datasets as cd
from config.datasets import DatasetConfig


def test_sets_built_from_analysis_sets():
    assert cd.SETS["antibody"].members == ("alphaseq", "snir")
    assert cd.SETS["antibody"].is_set is True


def test_real_dataset_is_not_a_set():
    assert cd.DATASETS["alphaseq"].is_set is False
    assert cd.DATASETS["alphaseq"].members is None


def test_selectable_is_union_without_collision():
    assert set(cd.SELECTABLE) == set(cd.DATASETS) | set(cd.SETS)
    assert not (set(cd.DATASETS) & set(cd.SETS))


def test_set_members_are_known_datasets():
    for name, cfg in cd.SETS.items():
        for member in cfg.members:
            assert member in cd.DATASETS, f"{name} references unknown dataset {member}"


def test_is_set_property():
    assert DatasetConfig(name="x").is_set is False
    assert DatasetConfig(name="x", members=("a", "b")).is_set is True
