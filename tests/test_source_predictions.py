from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motionage.analysis.motionage.source_predictions import (
    build_public_source_prediction_report_table,
    build_participant_source_predictions,
    logits_to_probabilities,
    summarize_source_predictions,
)


def test_logits_to_probabilities_is_stable_for_extreme_logits() -> None:
    probabilities = logits_to_probabilities(np.array([-1000.0, 0.0, 1000.0]))

    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1.0)


def test_build_participant_source_predictions_averages_window_probabilities() -> None:
    windows = pd.DataFrame(
        {
            "SEQN": ["p1", "p1", "p2", "p2", "p2"],
            "split": ["train", "train", "test", "test", "test"],
            "mortstat_60m": [0, 0, 1, 1, 1],
            "window_logit": [0.0, 1.0, -1.0, 0.0, 1.0],
        }
    )

    participants = build_participant_source_predictions(
        windows,
        id_column="SEQN",
        target_column="mortstat_60m",
        logit_column="window_logit",
    )

    assert participants.columns.tolist() == [
        "SEQN",
        "split",
        "mortstat_60m",
        "participant_probability",
        "participant_logit",
        "n_windows",
    ]
    assert participants["SEQN"].tolist() == ["p1", "p2"]
    assert participants["n_windows"].tolist() == [2, 3]

    expected_p1 = float(np.mean(logits_to_probabilities(np.array([0.0, 1.0]))))
    expected_p2 = float(np.mean(logits_to_probabilities(np.array([-1.0, 0.0, 1.0]))))
    assert participants.loc[participants["SEQN"] == "p1", "participant_probability"].item() == pytest.approx(
        expected_p1
    )
    assert participants.loc[participants["SEQN"] == "p2", "participant_probability"].item() == pytest.approx(
        expected_p2
    )
    assert participants.loc[participants["SEQN"] == "p1", "participant_logit"].item() == pytest.approx(
        np.log(expected_p1 / (1.0 - expected_p1))
    )


def test_build_participant_source_predictions_accepts_probability_column() -> None:
    windows = pd.DataFrame(
        {
            "sample_key": ["p1", "p1", "p2"],
            "partition": ["train", "train", "validation"],
            "window_probability": [0.2, 0.4, 0.8],
        }
    )

    participants = build_participant_source_predictions(
        windows,
        id_column="sample_key",
        split_column="partition",
        probability_column="window_probability",
        include_participant_logit=False,
    )

    assert participants.columns.tolist() == [
        "sample_key",
        "partition",
        "participant_probability",
        "n_windows",
    ]
    assert participants.loc[participants["sample_key"] == "p1", "participant_probability"].item() == pytest.approx(
        0.3
    )
    assert "participant_logit" not in participants.columns


def test_build_participant_source_predictions_rejects_mixed_splits_or_targets() -> None:
    mixed_split = pd.DataFrame(
        {
            "SEQN": ["p1", "p1"],
            "split": ["train", "test"],
            "window_probability": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="exactly one split"):
        build_participant_source_predictions(
            mixed_split,
            id_column="SEQN",
            probability_column="window_probability",
        )

    mixed_target = pd.DataFrame(
        {
            "SEQN": ["p1", "p1"],
            "split": ["train", "train"],
            "mortstat_60m": [0, 1],
            "window_probability": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="exactly one target"):
        build_participant_source_predictions(
            mixed_target,
            id_column="SEQN",
            target_column="mortstat_60m",
            probability_column="window_probability",
        )


def test_summarize_source_predictions_returns_public_aggregate_rows() -> None:
    participants = pd.DataFrame(
        {
            "SEQN": ["p1", "p2", "p3", "p4"],
            "split": ["train", "train", "test", "test"],
            "mortstat_60m": [0, 1, 0, 1],
            "participant_probability": [0.2, 0.6, 0.1, 0.9],
            "participant_logit": [-1.386, 0.405, -2.197, 2.197],
            "n_windows": [2, 4, 3, 5],
        }
    )

    rows = summarize_source_predictions(participants, target_column="mortstat_60m")

    assert rows == [
        {
            "split": "test",
            "n": 2,
            "events": 1,
            "non_events": 1,
            "event_rate": 0.5,
            "n_windows": 8,
            "mean_windows_per_participant": 4.0,
            "participant_probability_mean": 0.5,
            "participant_probability_std": 0.4,
            "participant_probability_min": 0.1,
            "participant_probability_max": 0.9,
        },
        {
            "split": "train",
            "n": 2,
            "events": 1,
            "non_events": 1,
            "event_rate": 0.5,
            "n_windows": 6,
            "mean_windows_per_participant": 3.0,
            "participant_probability_mean": 0.4,
            "participant_probability_std": 0.2,
            "participant_probability_min": 0.2,
            "participant_probability_max": 0.6,
        },
    ]
    for row in rows:
        assert "SEQN" not in row
        assert "participant_logit" not in row


def test_summarize_source_predictions_supports_targetless_local_replay() -> None:
    participants = pd.DataFrame(
        {
            "sample_key": ["p1", "p2"],
            "partition": ["train", "validation"],
            "participant_probability": [0.25, 0.75],
            "n_windows": [2, 6],
        }
    )

    rows = summarize_source_predictions(
        participants,
        split_column="partition",
        target_column=None,
    )

    assert rows == [
        {
            "partition": "train",
            "n": 1,
            "n_windows": 2,
            "mean_windows_per_participant": 2.0,
            "participant_probability_mean": 0.25,
            "participant_probability_std": 0.0,
            "participant_probability_min": 0.25,
            "participant_probability_max": 0.25,
        },
        {
            "partition": "validation",
            "n": 1,
            "n_windows": 6,
            "mean_windows_per_participant": 6.0,
            "participant_probability_mean": 0.75,
            "participant_probability_std": 0.0,
            "participant_probability_min": 0.75,
            "participant_probability_max": 0.75,
        },
    ]


def test_build_public_source_prediction_report_table_labels_models_without_identifiers() -> None:
    common_columns = {
        "sample_key": ["p1", "p2"],
        "split": ["test", "test"],
        "mortstat_60m": [0, 1],
        "participant_logit": [-1.0, 1.0],
        "checkpoint_path": ["local_run_a.pt", "local_run_a.pt"],
    }
    rows = build_public_source_prediction_report_table(
        {
            "GRU": pd.DataFrame(
                {
                    **common_columns,
                    "participant_probability": [0.2, 0.6],
                    "n_windows": [2, 4],
                }
            ),
            "LSTM": pd.DataFrame(
                {
                    **common_columns,
                    "participant_probability": [0.3, 0.7],
                    "n_windows": [1, 3],
                }
            ),
            "Transformer": pd.DataFrame(
                {
                    **common_columns,
                    "participant_probability": [0.1, 0.9],
                    "n_windows": [5, 5],
                }
            ),
        },
        target_column="mortstat_60m",
    )

    assert rows == [
        {
            "source_model": "GRU",
            "split": "test",
            "n": 2,
            "events": 1,
            "non_events": 1,
            "event_rate": 0.5,
            "n_windows": 6,
            "mean_windows_per_participant": 3.0,
            "participant_probability_mean": 0.4,
            "participant_probability_std": 0.2,
            "participant_probability_min": 0.2,
            "participant_probability_max": 0.6,
        },
        {
            "source_model": "LSTM",
            "split": "test",
            "n": 2,
            "events": 1,
            "non_events": 1,
            "event_rate": 0.5,
            "n_windows": 4,
            "mean_windows_per_participant": 2.0,
            "participant_probability_mean": 0.5,
            "participant_probability_std": 0.2,
            "participant_probability_min": 0.3,
            "participant_probability_max": 0.7,
        },
        {
            "source_model": "Transformer",
            "split": "test",
            "n": 2,
            "events": 1,
            "non_events": 1,
            "event_rate": 0.5,
            "n_windows": 10,
            "mean_windows_per_participant": 5.0,
            "participant_probability_mean": 0.5,
            "participant_probability_std": 0.4,
            "participant_probability_min": 0.1,
            "participant_probability_max": 0.9,
        },
    ]
    for row in rows:
        assert "sample_key" not in row
        assert "participant_logit" not in row
        assert "checkpoint_path" not in row


def test_method_docs_include_source_prediction_boundary() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("docs", "method.md").read_text()
    reports_text = Path(__file__).resolve().parents[1].joinpath("docs", "reports", "README.md").read_text()

    assert "build_participant_source_predictions" in text
    assert "summarize_source_predictions" in reports_text
    assert "synthetic or local run outputs only" in text
