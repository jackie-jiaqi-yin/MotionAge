from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motionage.evaluation.predictions import (
    aggregate_to_participant,
    align_meta_to_predictions,
    build_public_binary_evaluation_table,
    evaluate_binary_probability_splits,
    participant_prediction_frame,
)


def test_align_meta_to_predictions_accepts_exact_length_and_optional_prefix() -> None:
    meta = pd.DataFrame({"id": [101, 101, 102, 103], "start_idx": [0, 2, 0, 0]})

    exact = align_meta_to_predictions(meta, sample_count=4)
    prefix = align_meta_to_predictions(meta, sample_count=3, allow_prefix=True)

    assert exact.equals(meta)
    assert prefix["id"].tolist() == [101, 101, 102]
    assert prefix.index.tolist() == [0, 1, 2]


def test_aggregate_to_participant_averages_windows_and_preserves_first_target() -> None:
    meta = pd.DataFrame({"id": [101, 101, 102], "start_idx": [0, 2, 0]})

    probabilities, targets = aggregate_to_participant(
        np.asarray([0.2, 0.6, 0.9]),
        np.asarray([0, 0, 1]),
        meta,
    )

    assert probabilities.tolist() == pytest.approx([0.4, 0.9])
    assert targets.tolist() == [0, 1]


def test_aggregate_to_participant_rejects_unaligned_metadata() -> None:
    meta = pd.DataFrame({"id": [101, 101, 102, 102]})

    with pytest.raises(ValueError, match="metadata length must match"):
        aggregate_to_participant(np.asarray([0.2, 0.4, 0.8]), np.asarray([0, 0, 1]), meta)


def test_participant_prediction_frame_handles_metadata_and_synthetic_ids() -> None:
    meta = pd.DataFrame({"id": [101, 101, 102]})

    with_meta = participant_prediction_frame(
        np.asarray([0.2, 0.6, 0.9]),
        np.asarray([0, 0, 1]),
        meta=meta,
        split="test",
    )
    without_meta = participant_prediction_frame(np.asarray([0.3, 0.7]), np.asarray([0, 1]))

    assert with_meta.to_dict("records") == [
        {"id": 101, "probability": 0.4, "target": 0, "split": "test"},
        {"id": 102, "probability": 0.9, "target": 1, "split": "test"},
    ]
    assert without_meta["id"].tolist() == [0, 1]
    assert without_meta["probability"].tolist() == [0.3, 0.7]
    assert "split" not in without_meta.columns


def test_evaluate_binary_probability_splits_selects_validation_threshold() -> None:
    results = evaluate_binary_probability_splits(
        {
            "train": (np.asarray([0.1, 0.8, 0.2, 0.7]), np.asarray([0, 1, 0, 1])),
            "val": (np.asarray([0.1, 0.4, 0.6, 0.8]), np.asarray([0, 0, 1, 1])),
            "test": (np.asarray([0.2, 0.6, 0.7, 0.3]), np.asarray([0, 1, 1, 0])),
        },
        threshold_metric="balanced_accuracy",
    )

    assert results["meta"]["selected_threshold"] == pytest.approx(0.5)
    assert results["meta"]["selected_threshold_score"] == pytest.approx(1.0)
    assert results["meta"]["threshold_selection_source"] == "validation"
    assert results["test"]["n"] == 4
    assert results["test"]["events"] == 2
    assert results["test"]["non_events"] == 2
    assert results["test"]["event_rate"] == pytest.approx(0.5)
    assert results["test"]["balanced_accuracy"] == pytest.approx(1.0)


def test_evaluate_binary_probability_splits_uses_provided_threshold_without_validation() -> None:
    results = evaluate_binary_probability_splits(
        {
            "train": (np.asarray([0.1, 0.8]), np.asarray([0, 1])),
            "test": (np.asarray([0.2, 0.7]), np.asarray([0, 1])),
        },
        selected_threshold=0.7,
        selected_threshold_score=0.83,
        threshold_selection_source="winner_validation_selected_threshold",
    )

    assert results["meta"]["selected_threshold"] == pytest.approx(0.7)
    assert results["meta"]["selected_threshold_score"] == pytest.approx(0.83)
    assert results["meta"]["threshold_selection_source"] == "winner_validation_selected_threshold"
    assert results["test"]["threshold"] == pytest.approx(0.7)


def test_build_public_binary_evaluation_table_adds_threshold_metadata() -> None:
    results = evaluate_binary_probability_splits(
        {
            "train": (np.asarray([0.1, 0.8, 0.2, 0.7]), np.asarray([0, 1, 0, 1])),
            "val": (np.asarray([0.1, 0.4, 0.6, 0.8]), np.asarray([0, 0, 1, 1])),
            "test": (np.asarray([0.2, 0.6, 0.7, 0.3]), np.asarray([0, 1, 1, 0])),
        },
        threshold_metric="balanced_accuracy",
    )
    results["test"]["raw_probability_rows"] = [0.2, 0.6, 0.7, 0.3]

    rows = build_public_binary_evaluation_table(results, model="Transformer")

    assert [row["split"] for row in rows] == ["train", "val", "test"]
    assert all(row["model"] == "Transformer" for row in rows)
    assert all(row["threshold_metric"] == "balanced_accuracy" for row in rows)
    assert all(row["threshold_selection_source"] == "validation" for row in rows)
    assert all(row["selected_threshold"] == pytest.approx(0.5) for row in rows)
    assert all(row["selected_threshold_score"] == pytest.approx(1.0) for row in rows)
    assert rows[-1]["n"] == 4
    assert rows[-1]["events"] == 2
    assert rows[-1]["balanced_accuracy"] == pytest.approx(1.0)
    assert "raw_probability_rows" not in rows[-1]


def test_build_public_binary_evaluation_table_resolves_model_id_to_public_label() -> None:
    results = evaluate_binary_probability_splits(
        {
            "test": (np.asarray([0.2, 0.6, 0.7, 0.3]), np.asarray([0, 1, 1, 0])),
        },
        selected_threshold=0.5,
        threshold_selection_source="fixed_threshold",
    )

    rows = build_public_binary_evaluation_table(
        results,
        model_id="transformer_level1_latefusion",
        model_labels={"transformer_level1_latefusion": "Transformer MotionAge-FRC"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "Transformer MotionAge-FRC"
    assert row["split"] == "test"
    assert row["n"] == 4
    assert row["events"] == 2
    assert row["non_events"] == 2
    assert row["event_rate"] == pytest.approx(0.5)
    assert row["auroc"] == pytest.approx(1.0)
    assert row["auprc"] == pytest.approx(1.0)
    assert row["logloss"] == pytest.approx(results["test"]["logloss"])
    assert row["brier"] == pytest.approx(results["test"]["brier"])
    assert row["threshold"] == pytest.approx(0.5)
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert row["threshold_metric"] == "balanced_accuracy"
    assert row["selected_threshold"] == pytest.approx(0.5)
    assert np.isnan(row["selected_threshold_score"])
    assert row["threshold_selection_source"] == "fixed_threshold"
    assert "model_id" not in row


def test_build_public_binary_evaluation_table_requires_public_model_labels_for_model_ids() -> None:
    results = evaluate_binary_probability_splits(
        {
            "test": (np.asarray([0.2, 0.6, 0.7, 0.3]), np.asarray([0, 1, 1, 0])),
        },
        selected_threshold=0.5,
    )

    with pytest.raises(ValueError, match="Missing public model labels"):
        build_public_binary_evaluation_table(
            results,
            model_id="gru_level1_latefusion",
        )
