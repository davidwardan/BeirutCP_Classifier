"""Leakage-safe group and spatial splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitSummary:
    seed: int
    group_column: str
    train_rows: int
    validation_rows: int
    test_rows: int
    train_groups: int
    validation_groups: int
    test_groups: int


def _distribution_score(
    full_labels: pd.Series, selected_labels: pd.Series, fraction: float
) -> float:
    full = full_labels.value_counts(normalize=True)
    selected = selected_labels.value_counts(normalize=True).reindex(full.index, fill_value=0)
    distribution_error = float((full - selected).abs().sum())
    size_error = abs(len(selected_labels) / len(full_labels) - fraction)
    missing_penalty = float((selected == 0).sum())
    return distribution_error + 2.0 * size_error + missing_penalty


def choose_holdout_groups(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str,
    fraction: float,
    seed: int,
    candidates: int = 256,
) -> set[str]:
    """Choose whole groups while approximating size and class distribution."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    subset = frame[[group_column, label_column]].dropna().copy()
    subset[group_column] = subset[group_column].astype(str)
    if subset.empty:
        raise ValueError("No eligible rows are available for splitting")

    group_label_counts = subset.groupby(group_column)[label_column].nunique()
    if (group_label_counts == 1).all():
        groups = subset.drop_duplicates(group_column)
        train_groups, holdout_groups = train_test_split(
            groups,
            test_size=fraction,
            random_state=seed,
            stratify=groups[label_column],
        )
        del train_groups
        return set(holdout_groups[group_column].astype(str))

    splitter = GroupShuffleSplit(
        n_splits=candidates, test_size=fraction, random_state=seed
    )
    best_score = float("inf")
    best_groups: set[str] | None = None
    for _, selected_indices in splitter.split(
        subset, subset[label_column], groups=subset[group_column]
    ):
        selected = subset.iloc[selected_indices]
        score = _distribution_score(
            subset[label_column], selected[label_column], fraction
        )
        if score < best_score:
            best_score = score
            best_groups = set(selected[group_column])
    if best_groups is None:
        raise RuntimeError("Unable to produce a grouped split")
    return best_groups


def assign_splits(
    frame: pd.DataFrame,
    seed: int,
    group_column: str,
    label_column: str = "label",
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    common_dataset_column: str = "in_dataset_1",
    image_dataset_column: str = "in_dataset_2",
) -> tuple[pd.DataFrame, SplitSummary]:
    """Create an untouched common test set and group-safe train/validation sets."""
    data = frame.copy()
    data[group_column] = data[group_column].astype(str)
    data["split"] = None

    common = data[data[common_dataset_column].astype(bool) & data[label_column].notna()]
    test_groups = choose_holdout_groups(
        common, group_column, label_column, test_fraction, seed
    )
    in_test_group = data[group_column].isin(test_groups)
    data.loc[in_test_group & data[common_dataset_column].astype(bool), "split"] = "test"
    # In a spatial split, a held-out block can also contain dataset-2-only
    # buildings. Exclude those rows from training without enlarging the common
    # test population used to compare image-only and multimodal models.
    data.loc[
        in_test_group
        & ~data[common_dataset_column].astype(bool)
        & data[image_dataset_column].astype(bool),
        "split",
    ] = "heldout_extra"

    common_remaining = data[
        data[common_dataset_column].astype(bool)
        & data[label_column].notna()
        & data["split"].isna()
    ]
    relative_validation_fraction = validation_fraction / (1.0 - test_fraction)
    validation_groups = choose_holdout_groups(
        common_remaining,
        group_column,
        label_column,
        relative_validation_fraction,
        seed + 1009,
    )

    # Dataset 2 may contain image-only buildings. Allocate some of those to
    # validation as well, but never add them to the common held-out test set.
    extra = data[
        data[image_dataset_column].astype(bool)
        & ~data[common_dataset_column].astype(bool)
        & data[label_column].notna()
        & data["split"].isna()
    ]
    if not extra.empty:
        extra_validation_groups = choose_holdout_groups(
            extra,
            group_column,
            label_column,
            relative_validation_fraction,
            seed + 2017,
        )
        validation_groups |= extra_validation_groups

    data.loc[data[group_column].isin(validation_groups), "split"] = "val"
    eligible = data[image_dataset_column].astype(bool) & data[label_column].notna()
    data.loc[eligible & data["split"].isna(), "split"] = "train"

    split_group_sets = {
        name: set(data.loc[data["split"] == name, group_column])
        for name in ("train", "val", "test")
    }
    if (
        split_group_sets["train"] & split_group_sets["val"]
        or split_group_sets["train"] & split_group_sets["test"]
        or split_group_sets["val"] & split_group_sets["test"]
    ):
        raise AssertionError("Group leakage detected while assigning splits")

    summary = SplitSummary(
        seed=seed,
        group_column=group_column,
        train_rows=int((data["split"] == "train").sum()),
        validation_rows=int((data["split"] == "val").sum()),
        test_rows=int((data["split"] == "test").sum()),
        train_groups=len(split_group_sets["train"]),
        validation_groups=len(split_group_sets["val"]),
        test_groups=len(split_group_sets["test"]),
    )
    return data, summary
