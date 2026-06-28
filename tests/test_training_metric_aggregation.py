from __future__ import annotations

import pytest

import motionage.training as training


def test_build_train_epoch_metrics_averages_regression_totals() -> None:
    assert hasattr(training, "build_train_epoch_metrics")

    metrics = training.build_train_epoch_metrics(
        total_loss=6.0,
        total_grad_norm=1.5,
        n_batches=3,
        task_type=training.REGRESSION,
        total_abs_error=8.0,
        total_targets=4,
    )

    assert metrics == {"loss": 2.0, "avg_grad_norm": 0.5, "mae": 2.0}


def test_build_train_epoch_metrics_uses_safe_denominators_for_empty_regression_epoch() -> None:
    assert hasattr(training, "build_train_epoch_metrics")

    metrics = training.build_train_epoch_metrics(
        total_loss=0.0,
        total_grad_norm=0.0,
        n_batches=0,
        task_type=training.REGRESSION,
        total_abs_error=0.0,
        total_targets=0,
    )

    assert metrics == {"loss": 0.0, "avg_grad_norm": 0.0, "mae": 0.0}


def test_build_train_epoch_metrics_omits_mae_for_binary_classification() -> None:
    assert hasattr(training, "build_train_epoch_metrics")

    metrics = training.build_train_epoch_metrics(
        total_loss=1.2,
        total_grad_norm=0.4,
        n_batches=2,
        task_type=training.BINARY_CLASSIFICATION,
        total_abs_error=99.0,
        total_targets=10,
    )

    assert metrics == {"loss": 0.6, "avg_grad_norm": 0.2}


def test_build_train_epoch_metrics_rejects_unknown_task_type() -> None:
    assert hasattr(training, "build_train_epoch_metrics")

    with pytest.raises(ValueError, match="Unsupported task_type"):
        training.build_train_epoch_metrics(
            total_loss=1.0,
            total_grad_norm=0.1,
            n_batches=1,
            task_type="unsupported",
            total_abs_error=1.0,
            total_targets=1,
        )
