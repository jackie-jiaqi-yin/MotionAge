"""Evaluation utilities for MotionAge models and reports."""

from motionage.evaluation.metrics import (
    build_public_benchmark_sensitivity_table,
    build_public_binary_evaluation_row,
    binary_precision_recall_curve_rows,
    binary_probability_metrics,
    binary_roc_curve_rows,
    binary_target_summary,
    binary_threshold_metrics,
    binary_threshold_sweep,
    compute_all_metrics,
    logits_to_probabilities,
    select_binary_threshold,
)
from motionage.evaluation.predictions import (
    aggregate_to_participant,
    align_meta_to_predictions,
    build_public_binary_evaluation_table,
    evaluate_binary_probability_splits,
    participant_prediction_frame,
)
from motionage.evaluation.secondary import evaluate_secondary_feature_sets

__all__ = [
    "aggregate_to_participant",
    "align_meta_to_predictions",
    "build_public_benchmark_sensitivity_table",
    "build_public_binary_evaluation_row",
    "binary_precision_recall_curve_rows",
    "binary_probability_metrics",
    "binary_roc_curve_rows",
    "binary_target_summary",
    "binary_threshold_metrics",
    "binary_threshold_sweep",
    "build_public_binary_evaluation_table",
    "compute_all_metrics",
    "evaluate_binary_probability_splits",
    "evaluate_secondary_feature_sets",
    "logits_to_probabilities",
    "participant_prediction_frame",
    "select_binary_threshold",
]
