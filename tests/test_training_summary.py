from __future__ import annotations

import math

import pytest

import motionage.training as training


def test_build_training_summary_includes_regression_fields() -> None:
    assert hasattr(training, "build_training_summary")

    summary = training.build_training_summary(
        task_type=training.REGRESSION,
        selection_metric_name="mae",
        selection_metric_direction="minimize",
        best_val_metric=3.2,
        best_val_loss=12.5,
        best_epoch=7,
        epochs_trained=11,
        best_eval_metrics={"mae": 3.2, "r2": 0.48},
    )

    assert summary == {
        "task_type": training.REGRESSION,
        "selection_metric_name": "mae",
        "selection_metric_direction": "minimize",
        "best_val_metric": 3.2,
        "best_val_loss": 12.5,
        "best_epoch": 7,
        "epochs_trained": 11,
        "best_val_mae": 3.2,
        "best_val_r2": 0.48,
    }


def test_build_training_summary_includes_binary_fields() -> None:
    assert hasattr(training, "build_training_summary")

    summary = training.build_training_summary(
        task_type=training.BINARY_CLASSIFICATION,
        selection_metric_name="auroc",
        selection_metric_direction="maximize",
        best_val_metric=0.83,
        best_val_loss=0.51,
        best_epoch=5,
        epochs_trained=9,
        best_eval_metrics={
            "auroc": 0.83,
            "auprc": 0.58,
            "logloss": 0.51,
            "brier": 0.16,
        },
    )

    assert summary == {
        "task_type": training.BINARY_CLASSIFICATION,
        "selection_metric_name": "auroc",
        "selection_metric_direction": "maximize",
        "best_val_metric": 0.83,
        "best_val_loss": 0.51,
        "best_epoch": 5,
        "epochs_trained": 9,
        "best_val_auroc": 0.83,
        "best_val_auprc": 0.58,
        "best_val_logloss": 0.51,
        "best_val_brier": 0.16,
    }


def test_build_fixed_epoch_training_summary_marks_no_validation_fit() -> None:
    assert hasattr(training, "build_fixed_epoch_training_summary")

    summary = training.build_fixed_epoch_training_summary(
        task_type=training.REGRESSION,
        selection_metric_name="mae",
        selection_metric_direction="minimize",
        best_val_metric=float("nan"),
        best_val_loss=float("nan"),
        best_epoch=20,
        epochs_trained=10,
        best_eval_metrics={},
        fixed_epochs=20,
        start_epoch=11,
        resumed_from_epoch=10,
    )

    assert summary["fit_mode"] == "fixed_epochs_no_validation"
    assert summary["fixed_epochs"] == 20
    assert summary["start_epoch"] == 11
    assert summary["resumed_from_epoch"] == 10
    assert summary["best_epoch"] == 20
    assert summary["epochs_trained"] == 10
    assert math.isnan(summary["best_val_metric"])
    assert math.isnan(summary["best_val_loss"])
    assert math.isnan(summary["best_val_mae"])
    assert math.isnan(summary["best_val_r2"])


def test_training_summary_rejects_unknown_task_type() -> None:
    assert hasattr(training, "build_training_summary")

    with pytest.raises(ValueError, match="Unsupported task_type"):
        training.build_training_summary(
            task_type="unsupported",
            selection_metric_name="mae",
            selection_metric_direction="minimize",
            best_val_metric=3.2,
            best_val_loss=12.5,
            best_epoch=7,
            epochs_trained=11,
            best_eval_metrics={"mae": 3.2, "r2": 0.48},
        )
