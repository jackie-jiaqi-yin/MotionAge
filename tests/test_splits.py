from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motionage.data.splits import (
    filter_dataframe_by_ids,
    get_unique_ids,
    load_precomputed_id_splits,
    split_train_test_dataframe,
    split_train_test_ids,
)


def test_get_unique_ids_returns_sorted_non_missing_ids() -> None:
    frame = pd.DataFrame({"SEQN": [3, 1, np.nan, 2, 1]})

    assert get_unique_ids(frame).tolist() == [1.0, 2.0, 3.0]


def test_split_train_test_ids_is_deterministic_and_disjoint() -> None:
    split = split_train_test_ids([1, 2, 3, 4, 5], test_size=0.4, random_seed=7)

    assert set(split) == {"train_ids", "test_ids"}
    assert set(split["train_ids"]).isdisjoint(set(split["test_ids"]))
    assert sorted([*split["train_ids"], *split["test_ids"]]) == [1, 2, 3, 4, 5]


def test_split_train_test_dataframe_keeps_participants_disjoint() -> None:
    frame = pd.DataFrame({"SEQN": [1, 1, 2, 2, 3, 3, 4, 4], "value": range(8)})

    train_df, test_df = split_train_test_dataframe(frame, test_size=0.25, random_seed=42)

    assert set(train_df["SEQN"]).isdisjoint(set(test_df["SEQN"]))
    assert len(train_df) + len(test_df) == len(frame)


def test_filter_dataframe_by_ids_returns_copy() -> None:
    frame = pd.DataFrame({"SEQN": [1, 2, 3], "value": [10, 20, 30]})

    filtered = filter_dataframe_by_ids(frame, [2, 3])
    filtered.loc[filtered["SEQN"] == 2, "value"] = 999

    assert filtered["SEQN"].tolist() == [2, 3]
    assert frame.loc[frame["SEQN"] == 2, "value"].item() == 20


def test_load_precomputed_id_splits(tmp_path: Path) -> None:
    pd.DataFrame({"SEQN": [3, 1, 1]}).to_csv(tmp_path / "train_ids.csv", index=False)
    pd.DataFrame({"SEQN": [4]}).to_csv(tmp_path / "val_ids.csv", index=False)
    pd.DataFrame({"SEQN": [6, 5]}).to_csv(tmp_path / "test_ids.csv", index=False)

    splits = load_precomputed_id_splits(tmp_path)

    assert splits["train"].tolist() == [1, 3]
    assert splits["val"].tolist() == [4]
    assert splits["test"].tolist() == [5, 6]


def test_split_train_test_ids_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="test_size"):
        split_train_test_ids([1, 2, 3], test_size=1.0)

    with pytest.raises(ValueError, match="at least 2"):
        split_train_test_ids([1], test_size=0.2)
