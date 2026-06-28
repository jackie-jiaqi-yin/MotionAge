from __future__ import annotations

import math

import pytest

import motionage.training as training


def test_initial_best_metric_uses_direction_specific_sentinel() -> None:
    assert hasattr(training, "initial_best_metric")

    assert training.initial_best_metric("minimize") == math.inf
    assert training.initial_best_metric("maximize") == -math.inf


def test_metric_improved_handles_minimize_and_maximize_directions() -> None:
    assert hasattr(training, "metric_improved")

    assert training.metric_improved(0.8, 1.0, "minimize") is True
    assert training.metric_improved(1.0, 1.0, "minimize") is False
    assert training.metric_improved(1.2, 1.0, "minimize") is False

    assert training.metric_improved(0.9, 0.8, "maximize") is True
    assert training.metric_improved(0.8, 0.8, "maximize") is False
    assert training.metric_improved(0.7, 0.8, "maximize") is False


def test_selection_helpers_reject_unknown_direction() -> None:
    assert hasattr(training, "initial_best_metric")
    assert hasattr(training, "metric_improved")

    with pytest.raises(ValueError, match="direction"):
        training.initial_best_metric("largest")
    with pytest.raises(ValueError, match="direction"):
        training.metric_improved(1.0, 0.0, "largest")
