from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motionage.analysis.motionage.source_predictions import (
    build_participant_source_predictions,
    logits_to_probabilities,
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


def test_method_docs_include_source_prediction_boundary() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("docs", "method.md").read_text()

    assert "build_participant_source_predictions" in text
    assert "synthetic or local run outputs only" in text
