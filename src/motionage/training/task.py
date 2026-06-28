"""Task-type helpers for shared training and evaluation code."""

from __future__ import annotations

from typing import Any, Literal

TaskType = Literal["regression", "binary_classification"]

REGRESSION: TaskType = "regression"
BINARY_CLASSIFICATION: TaskType = "binary_classification"

_BINARY_SELECTION_METRIC_DIRECTIONS: dict[str, Literal["minimize", "maximize"]] = {
    "auroc": "maximize",
    "auprc": "maximize",
    "logloss": "minimize",
    "brier": "minimize",
}


def resolve_task_type(config: dict[str, Any]) -> TaskType:
    """Resolve task type from config, defaulting to regression."""
    task_config = config.get("task")
    if not isinstance(task_config, dict):
        return REGRESSION

    raw = str(task_config.get("type", REGRESSION)).strip().lower()
    if raw in {"regression", "continuous"}:
        return REGRESSION
    if raw in {"binary", "binary_classification", "classification"}:
        return BINARY_CLASSIFICATION
    raise ValueError(f"Unsupported task.type '{raw}'.")


def is_binary_classification(config: dict[str, Any]) -> bool:
    """Return whether a config declares a binary classification task."""
    return resolve_task_type(config) == BINARY_CLASSIFICATION


def selection_metric_name(config: dict[str, Any]) -> str:
    """Return the primary validation metric used for model selection."""
    if not is_binary_classification(config):
        return "mae"

    task_config = config.get("task")
    if not isinstance(task_config, dict):
        return "auroc"
    metric = str(task_config.get("selection_metric", "auroc")).strip().lower()
    if metric not in _BINARY_SELECTION_METRIC_DIRECTIONS:
        raise ValueError(
            f"Unsupported binary task.selection_metric '{metric}'. "
            f"Choose one of: {sorted(_BINARY_SELECTION_METRIC_DIRECTIONS)}."
        )
    return metric


def selection_metric_direction(config: dict[str, Any]) -> Literal["minimize", "maximize"]:
    """Return whether the selection metric should be minimized or maximized."""
    if is_binary_classification(config):
        return _BINARY_SELECTION_METRIC_DIRECTIONS[selection_metric_name(config)]
    return "minimize"


def scheduler_mode(config: dict[str, Any]) -> Literal["min", "max"]:
    """Return PyTorch scheduler mode for the selection metric."""
    if is_binary_classification(config):
        return "max" if selection_metric_direction(config) == "maximize" else "min"
    return "min"


def threshold_metric_name(config: dict[str, Any]) -> str:
    """Return the threshold-selection metric for binary tasks."""
    task_config = config.get("task")
    if not isinstance(task_config, dict):
        return "balanced_accuracy"
    return str(task_config.get("threshold_metric", "balanced_accuracy"))
