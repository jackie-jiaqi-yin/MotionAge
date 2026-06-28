from __future__ import annotations

import pytest

from motionage.training.task import (
    BINARY_CLASSIFICATION,
    REGRESSION,
    is_binary_classification,
    resolve_task_type,
    scheduler_mode,
    selection_metric_direction,
    selection_metric_name,
    threshold_metric_name,
)


def test_resolve_task_type_defaults_to_regression() -> None:
    assert resolve_task_type({}) == REGRESSION
    assert resolve_task_type({"task": {"type": "continuous"}}) == REGRESSION


def test_resolve_task_type_accepts_binary_aliases() -> None:
    assert resolve_task_type({"task": {"type": "binary"}}) == BINARY_CLASSIFICATION
    assert resolve_task_type({"task": {"type": "classification"}}) == BINARY_CLASSIFICATION
    assert is_binary_classification({"task": {"type": "binary_classification"}}) is True


def test_binary_selection_metric_controls_direction_and_scheduler_mode() -> None:
    config = {"task": {"type": "binary_classification", "selection_metric": "logloss"}}

    assert selection_metric_name(config) == "logloss"
    assert selection_metric_direction(config) == "minimize"
    assert scheduler_mode(config) == "min"


def test_binary_selection_metric_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="selection_metric"):
        selection_metric_name({"task": {"type": "binary", "selection_metric": "accuracy"}})


def test_regression_and_threshold_metric_defaults() -> None:
    assert selection_metric_name({}) == "mae"
    assert selection_metric_direction({}) == "minimize"
    assert scheduler_mode({}) == "min"
    assert threshold_metric_name({}) == "balanced_accuracy"
    assert threshold_metric_name({"task": {"threshold_metric": "f1"}}) == "f1"
