"""Prediction alignment and participant-level evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from motionage.evaluation.metrics import compute_all_metrics, select_binary_threshold


def align_meta_to_predictions(
    meta: pd.DataFrame,
    sample_count: int,
    *,
    allow_prefix: bool = False,
) -> pd.DataFrame:
    """Align window metadata to the number of model predictions."""
    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative, got {sample_count}.")

    if len(meta) == sample_count:
        return meta
    if allow_prefix and len(meta) > sample_count:
        return meta.iloc[:sample_count].reset_index(drop=True).copy()

    raise ValueError(
        "Window metadata length must match the number of predictions "
        f"(got metadata={len(meta)}, predictions={sample_count}, allow_prefix={allow_prefix})."
    )


def aggregate_to_participant(
    predictions: np.ndarray,
    targets: np.ndarray,
    meta: pd.DataFrame,
    *,
    id_column: str = "id",
) -> tuple[np.ndarray, np.ndarray]:
    """Average window predictions by participant and keep each participant's first target."""
    prediction_arr = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(targets).reshape(-1)
    if prediction_arr.shape[0] != target_arr.shape[0]:
        raise ValueError(
            "Predictions and targets must have the same length "
            f"(got predictions={len(prediction_arr)}, targets={len(target_arr)})."
        )
    if id_column not in meta.columns:
        raise KeyError(f"Metadata must include participant id column: {id_column}")

    aligned_meta = align_meta_to_predictions(meta, prediction_arr.shape[0])
    frame = pd.DataFrame(
        {
            "probability": prediction_arr,
            "target": target_arr,
            id_column: aligned_meta[id_column].to_numpy(),
        }
    )
    grouped = (
        frame.groupby(id_column, sort=False, dropna=False)
        .agg(probability=("probability", "mean"), target=("target", "first"))
        .reset_index()
    )
    return grouped["probability"].to_numpy(dtype=float), grouped["target"].to_numpy()


def participant_prediction_frame(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    meta: pd.DataFrame | None = None,
    split: str | None = None,
    id_column: str = "id",
) -> pd.DataFrame:
    """Build a participant-level prediction frame for reporting or artifact replay."""
    probability_arr = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(targets).reshape(-1)
    if probability_arr.shape[0] != target_arr.shape[0]:
        raise ValueError(
            "Probabilities and targets must have the same length "
            f"(got probabilities={len(probability_arr)}, targets={len(target_arr)})."
        )

    if meta is None or meta.empty:
        frame = pd.DataFrame(
            {
                id_column: np.arange(probability_arr.shape[0], dtype=np.int64),
                "probability": probability_arr,
                "target": target_arr,
            }
        )
    else:
        participant_probabilities, participant_targets = aggregate_to_participant(
            probability_arr,
            target_arr,
            meta,
            id_column=id_column,
        )
        participant_ids = (
            align_meta_to_predictions(meta, probability_arr.shape[0])
            .drop_duplicates(subset=[id_column], keep="first")[id_column]
            .to_numpy()
        )
        frame = pd.DataFrame(
            {
                id_column: participant_ids,
                "probability": participant_probabilities,
                "target": participant_targets,
            }
        )

    if split is not None:
        frame["split"] = str(split)
    return frame


def evaluate_binary_probability_splits(
    split_probabilities: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    threshold_metric: str = "balanced_accuracy",
    selected_threshold: float | None = None,
    selected_threshold_score: float | None = None,
    threshold_selection_source: str | None = None,
) -> dict[str, dict[str, float | str]]:
    """Evaluate binary probability arrays for train/validation/test-style splits."""
    prepared = {
        split_name: (np.asarray(probabilities).reshape(-1), np.asarray(targets).reshape(-1))
        for split_name, (probabilities, targets) in split_probabilities.items()
    }
    if not prepared:
        raise ValueError("At least one split is required.")

    threshold = 0.5
    threshold_score = float("nan")
    selection_source = threshold_selection_source or "default_0.5"
    if selected_threshold is not None:
        threshold = float(selected_threshold)
        if selected_threshold_score is not None:
            threshold_score = float(selected_threshold_score)
        selection_source = threshold_selection_source or "provided_threshold"
    elif "val" in prepared:
        val_probabilities, val_targets = prepared["val"]
        threshold, threshold_score = select_binary_threshold(
            val_targets,
            val_probabilities,
            metric=threshold_metric,
        )
        selection_source = threshold_selection_source or "validation"

    results: dict[str, dict[str, float | str]] = {
        split_name: compute_all_metrics(
            targets,
            probabilities,
            task_type="binary_classification",
            threshold=threshold,
        )
        for split_name, (probabilities, targets) in prepared.items()
    }
    results["meta"] = {
        "task_type": "binary_classification",
        "threshold_metric": str(threshold_metric),
        "selected_threshold": float(threshold),
        "selected_threshold_score": float(threshold_score),
        "threshold_selection_source": str(selection_source),
    }
    return results
