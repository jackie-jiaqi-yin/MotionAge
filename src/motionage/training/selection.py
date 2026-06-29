"""Metric-selection helpers for early stopping and checkpointing."""

from __future__ import annotations

from typing import Literal

MetricDirection = Literal["minimize", "maximize"]


def initial_best_metric(direction: MetricDirection) -> float:
    """Return the sentinel best value for a metric direction."""
    if direction == "maximize":
        return float("-inf")
    if direction == "minimize":
        return float("inf")
    raise ValueError(f"Unsupported selection metric direction: {direction!r}.")


def metric_improved(value: float, best_value: float, direction: MetricDirection) -> bool:
    """Return whether a metric value improves over the current best value."""
    if direction == "maximize":
        return value > best_value
    if direction == "minimize":
        return value < best_value
    raise ValueError(f"Unsupported selection metric direction: {direction!r}.")
