"""Participant-level train/test splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from motionage.preprocessing.constants import ActivityColumns


def get_unique_ids(df: pd.DataFrame, id_column: str = ActivityColumns.ID) -> np.ndarray:
    """Return sorted unique participant IDs from a dataframe."""
    if id_column not in df.columns:
        raise KeyError(f"Column '{id_column}' not found in dataframe.")

    ids = df[id_column].dropna().unique()
    if len(ids) == 0:
        raise ValueError("No participant IDs found after dropping missing values.")

    return np.sort(ids)


def split_train_test_ids(
    participant_ids: np.ndarray | list[Any],
    test_size: float = 0.2,
    random_seed: int = 42,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    """Split participant IDs into sorted train and test ID arrays."""
    _validate_test_size(test_size=test_size)

    ids = np.unique(np.asarray(participant_ids))
    if len(ids) < 2:
        raise ValueError(f"Need at least 2 unique IDs for train/test split, got {len(ids)}.")

    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_seed,
        shuffle=shuffle,
    )

    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError(
            "Train or test split is empty. Adjust test_size or provide more participants."
        )

    return {"train_ids": np.sort(train_ids), "test_ids": np.sort(test_ids)}


def load_precomputed_id_splits(
    split_dir: str | Path,
    id_column: str = ActivityColumns.ID,
) -> dict[str, np.ndarray]:
    """Load precomputed train/val/test participant IDs from CSV files."""
    split_dir = Path(split_dir)
    loaded: dict[str, np.ndarray] = {}

    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}_ids.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing precomputed split file: {csv_path}")

        frame = pd.read_csv(csv_path)
        if id_column not in frame.columns:
            raise KeyError(f"Column '{id_column}' not found in {csv_path}")

        ids = frame[id_column].dropna().to_numpy()
        if len(ids) == 0:
            raise ValueError(f"No IDs found in {csv_path}")

        loaded[split_name] = np.sort(np.unique(ids))

    return loaded


def split_train_test_dataframe(
    df: pd.DataFrame,
    id_column: str = ActivityColumns.ID,
    test_size: float = 0.2,
    random_seed: int = 42,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split row-level dataframes by participant ID to avoid participant leakage."""
    unique_ids = get_unique_ids(df=df, id_column=id_column)
    split_ids = split_train_test_ids(
        participant_ids=unique_ids,
        test_size=test_size,
        random_seed=random_seed,
        shuffle=shuffle,
    )
    train_df = filter_dataframe_by_ids(df=df, ids=split_ids["train_ids"], id_column=id_column)
    test_df = filter_dataframe_by_ids(df=df, ids=split_ids["test_ids"], id_column=id_column)
    return train_df, test_df


def filter_dataframe_by_ids(
    df: pd.DataFrame,
    ids: np.ndarray | list[Any],
    id_column: str = ActivityColumns.ID,
) -> pd.DataFrame:
    """Return a copy of rows where `id_column` is in `ids`."""
    if id_column not in df.columns:
        raise KeyError(f"Column '{id_column}' not found in dataframe.")

    id_set = set(np.asarray(ids).tolist())
    return df[df[id_column].isin(id_set)].copy()


def _validate_test_size(test_size: float) -> None:
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
