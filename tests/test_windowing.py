from __future__ import annotations

import pandas as pd
import pytest

from motionage.data.windowing import create_windows, create_windows_for_ids


def _activity_frame() -> pd.DataFrame:
    rows = []
    for participant_id, target in [(101, 0), (202, 1)]:
        for minute_idx in range(5):
            rows.append(
                {
                    "SEQN": participant_id,
                    "PAXN": minute_idx,
                    "PAXHOUR": minute_idx,
                    "PAXDAY": 1,
                    "PAXMINUT": minute_idx,
                    "intensity_mean": float(participant_id + minute_idx),
                    "attention_flag": 1.0,
                    "mortstat": target,
                }
            )
    return pd.DataFrame(rows)


def test_create_windows_participant_targets_and_metadata() -> None:
    result = create_windows(
        _activity_frame(),
        feature_columns=["intensity_mean", "PAXHOUR"],
        target_column="mortstat",
        seq_len=3,
        stride=2,
    )

    assert result["X"].shape == (4, 3, 2)
    assert result["y"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert result["meta"]["id"].tolist() == [101, 101, 202, 202]
    assert result["meta"]["start_idx"].tolist() == [0, 2, 0, 2]


def test_create_windows_last_timestep_target_mode() -> None:
    frame = _activity_frame()
    frame.loc[frame["SEQN"] == 101, "mortstat"] = [0, 0, 1, 1, 1]

    result = create_windows(
        frame[frame["SEQN"] == 101],
        feature_columns=["intensity_mean"],
        target_column="mortstat",
        seq_len=3,
        stride=1,
        target_mode="last_timestep",
    )

    assert result["y"].tolist() == [1.0, 1.0, 1.0]


def test_create_windows_filters_by_attention_ratio() -> None:
    frame = _activity_frame()
    frame.loc[(frame["SEQN"] == 101) & (frame["PAXN"].isin([0, 1])), "attention_flag"] = 0.0

    result = create_windows(
        frame,
        feature_columns=["intensity_mean"],
        target_column="mortstat",
        seq_len=3,
        stride=1,
        min_attention_ratio=0.75,
    )

    assert result["meta"]["id"].tolist() == [101, 202, 202, 202]
    assert result["meta"]["start_idx"].tolist() == [2, 0, 1, 2]


def test_create_windows_returns_empty_arrays_when_no_windows_survive() -> None:
    result = create_windows(
        _activity_frame(),
        feature_columns=["intensity_mean"],
        target_column="mortstat",
        seq_len=10,
    )

    assert result["X"].shape == (0, 10, 1)
    assert result["y"].shape == (0,)
    assert list(result["meta"].columns) == ["id", "start_idx", "end_idx"]


def test_create_windows_rejects_multiple_participant_targets() -> None:
    frame = _activity_frame()
    frame.loc[frame["SEQN"] == 101, "mortstat"] = [0, 0, 1, 1, 1]

    with pytest.raises(ValueError, match="multiple values"):
        create_windows(
            frame[frame["SEQN"] == 101],
            feature_columns=["intensity_mean"],
            target_column="mortstat",
            seq_len=3,
        )


def test_create_windows_for_ids_filters_participants() -> None:
    result = create_windows_for_ids(
        _activity_frame(),
        participant_ids=[202],
        feature_columns=["intensity_mean"],
        target_column="mortstat",
        seq_len=5,
    )

    assert result["X"].shape == (1, 5, 1)
    assert result["meta"]["id"].tolist() == [202]


def test_create_windows_validates_inputs() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        create_windows(
            _activity_frame(),
            feature_columns=["intensity_mean"],
            target_column="mortstat",
            seq_len=0,
        )

    with pytest.raises(KeyError, match="Missing required columns"):
        create_windows(
            _activity_frame(),
            feature_columns=["missing_feature"],
            target_column="mortstat",
            seq_len=3,
        )
