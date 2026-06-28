"""Evaluation utilities for MotionAge models and reports."""

from motionage.evaluation.metrics import (
    binary_precision_recall_curve_rows,
    binary_probability_metrics,
    binary_roc_curve_rows,
    binary_threshold_metrics,
    binary_threshold_sweep,
    compute_all_metrics,
    logits_to_probabilities,
    select_binary_threshold,
)
from motionage.evaluation.predictions import (
    aggregate_to_participant,
    align_meta_to_predictions,
    evaluate_binary_probability_splits,
    participant_prediction_frame,
)

__all__ = [
    "aggregate_to_participant",
    "align_meta_to_predictions",
    "binary_precision_recall_curve_rows",
    "binary_probability_metrics",
    "binary_roc_curve_rows",
    "binary_threshold_metrics",
    "binary_threshold_sweep",
    "compute_all_metrics",
    "evaluate_binary_probability_splits",
    "logits_to_probabilities",
    "participant_prediction_frame",
    "select_binary_threshold",
]
