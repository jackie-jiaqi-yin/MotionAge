from __future__ import annotations

import math

import numpy as np
import pytest

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


def test_logits_to_probabilities_is_stable_for_extreme_logits() -> None:
    probabilities = logits_to_probabilities(np.asarray([-1000.0, 0.0, 1000.0]))

    assert probabilities[0] == pytest.approx(1.0 / (1.0 + math.exp(60.0)))
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1.0 / (1.0 + math.exp(-60.0)))


def test_binary_probability_metrics_filters_invalid_rows_and_reports_fallbacks() -> None:
    metrics = binary_probability_metrics(
        np.asarray([0, 1, 1, np.nan]),
        np.asarray([0.1, 0.9, np.nan, 0.4]),
    )

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["positive_rate"] == pytest.approx(0.5)
    assert metrics["logloss"] > 0.0
    assert metrics["brier"] > 0.0

    one_class = binary_probability_metrics(np.asarray([1, 1]), np.asarray([0.2, 0.8]))
    assert one_class["auroc"] == pytest.approx(0.5)
    assert one_class["auprc"] == pytest.approx(1.0)


def test_binary_threshold_metrics_and_selection_use_validation_scores() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    y_prob = np.asarray([0.1, 0.4, 0.6, 0.8])

    metrics = binary_threshold_metrics(y_true, y_prob, threshold=0.5)
    threshold, score = select_binary_threshold(y_true, y_prob, metric="balanced_accuracy")

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert threshold == pytest.approx(0.5)
    assert score == pytest.approx(1.0)


def test_binary_threshold_sweep_and_curve_rows_have_stable_schemas() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    y_prob = np.asarray([0.1, 0.4, 0.6, 0.8])

    sweep = binary_threshold_sweep(y_true, y_prob)
    pr_rows = binary_precision_recall_curve_rows(y_true, y_prob)
    roc_rows = binary_roc_curve_rows(y_true, y_prob)

    assert [row["threshold"] for row in sweep] == sorted(row["threshold"] for row in sweep)
    assert {0.0, 0.5, 1.0}.issubset({row["threshold"] for row in sweep})
    assert set(sweep[0]) == {"threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1"}
    assert set(pr_rows[0]) == {"precision", "recall", "threshold"}
    assert set(roc_rows[0]) == {"fpr", "tpr", "threshold"}


def test_compute_all_metrics_supports_binary_and_regression_tasks() -> None:
    binary = compute_all_metrics(
        np.asarray([0, 1]),
        np.asarray([0.2, 0.8]),
        task_type="binary_classification",
        threshold=0.5,
    )
    regression = compute_all_metrics(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.5, 2.5]),
        task_type="regression",
    )

    assert binary["auroc"] == pytest.approx(1.0)
    assert binary["threshold"] == pytest.approx(0.5)
    assert regression["mae"] == pytest.approx(1.0 / 3.0)
    assert regression["rmse"] == pytest.approx(math.sqrt(0.5 / 3.0))
    assert regression["median_ae"] == pytest.approx(0.5)
    assert regression["r2"] == pytest.approx(0.75)
