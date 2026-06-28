from __future__ import annotations

import csv
import math

import pytest

import motionage.training as training


def test_nan_validation_metrics_returns_task_specific_nan_fields() -> None:
    assert hasattr(training, "nan_validation_metrics")

    regression_metrics = training.nan_validation_metrics(
        task_type=training.REGRESSION,
        selection_metric_name="mae",
    )
    binary_metrics = training.nan_validation_metrics(
        task_type=training.BINARY_CLASSIFICATION,
        selection_metric_name="auroc",
    )

    assert set(regression_metrics) == {"mae", "r2"}
    assert set(binary_metrics) == {"auroc", "auprc", "logloss", "brier"}
    assert all(math.isnan(value) for value in regression_metrics.values())
    assert all(math.isnan(value) for value in binary_metrics.values())


def test_nan_validation_metrics_preserves_custom_selection_metric() -> None:
    assert hasattr(training, "nan_validation_metrics")

    metrics = training.nan_validation_metrics(
        task_type=training.REGRESSION,
        selection_metric_name="custom_score",
    )

    assert set(metrics) == {"mae", "r2", "custom_score"}
    assert math.isnan(metrics["custom_score"])


def test_build_training_log_row_includes_common_and_regression_fields() -> None:
    assert hasattr(training, "build_training_log_row")

    row = training.build_training_log_row(
        epoch=3,
        freeze_stage_label="head-only",
        train_metrics={"loss": 1.2, "avg_grad_norm": 0.3, "mae": 4.5},
        val_metrics={"mae": 3.4, "r2": 0.6},
        val_loss=1.1,
        lr_min=1.0e-5,
        lr_max=1.0e-3,
        trainable_param_count=17,
        task_type=training.REGRESSION,
        selection_metric_name="mae",
    )

    assert row == {
        "epoch": 3,
        "train_loss": 1.2,
        "avg_grad_norm": 0.3,
        "val_loss": 1.1,
        "lr_min": 1.0e-5,
        "lr_max": 1.0e-3,
        "freeze_stage": "head-only",
        "trainable_param_count": 17,
        "selection_metric_name": "mae",
        "selection_metric_value": 3.4,
        "train_mae": 4.5,
        "val_mae": 3.4,
        "val_r2": 0.6,
    }


def test_build_training_log_row_includes_binary_fields() -> None:
    assert hasattr(training, "build_training_log_row")

    row = training.build_training_log_row(
        epoch=4,
        freeze_stage_label="all-layers",
        train_metrics={"loss": 0.7, "avg_grad_norm": 0.9},
        val_metrics={"auroc": 0.82, "auprc": 0.52, "logloss": 0.46, "brier": 0.14},
        val_loss=0.46,
        lr_min=2.0e-6,
        lr_max=5.0e-4,
        trainable_param_count=99,
        task_type=training.BINARY_CLASSIFICATION,
        selection_metric_name="auroc",
    )

    assert row == {
        "epoch": 4,
        "train_loss": 0.7,
        "avg_grad_norm": 0.9,
        "val_loss": 0.46,
        "lr_min": 2.0e-6,
        "lr_max": 5.0e-4,
        "freeze_stage": "all-layers",
        "trainable_param_count": 99,
        "selection_metric_name": "auroc",
        "selection_metric_value": 0.82,
        "val_auroc": 0.82,
        "val_auprc": 0.52,
        "val_logloss": 0.46,
        "val_brier": 0.14,
    }


def test_build_training_log_row_rejects_unknown_task_type() -> None:
    assert hasattr(training, "build_training_log_row")

    with pytest.raises(ValueError, match="Unsupported task_type"):
        training.build_training_log_row(
            epoch=1,
            freeze_stage_label="all-layers",
            train_metrics={"loss": 0.7, "avg_grad_norm": 0.9},
            val_metrics={"mae": 3.4, "r2": 0.6},
            val_loss=0.46,
            lr_min=2.0e-6,
            lr_max=5.0e-4,
            trainable_param_count=99,
            task_type="unsupported",
            selection_metric_name="mae",
        )


def test_training_log_fieldnames_follow_first_row_order() -> None:
    assert hasattr(training, "training_log_fieldnames")

    rows = [
        {"epoch": 1, "train_loss": 1.2, "val_loss": 1.1},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 0.9},
    ]

    assert training.training_log_fieldnames(rows) == ("epoch", "train_loss", "val_loss")
    assert training.training_log_fieldnames([]) == ()


def test_write_training_log_csv_writes_header_and_rows(tmp_path) -> None:
    assert hasattr(training, "write_training_log_csv")

    rows = [
        {"epoch": 1, "train_loss": 1.2, "val_loss": 1.1},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 0.9},
    ]
    path = tmp_path / "metrics" / "training_log.csv"

    wrote = training.write_training_log_csv(rows, path)

    assert wrote is True
    with path.open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == [
            {"epoch": "1", "train_loss": "1.2", "val_loss": "1.1"},
            {"epoch": "2", "train_loss": "1.0", "val_loss": "0.9"},
        ]


def test_write_training_log_csv_skips_empty_logs(tmp_path) -> None:
    assert hasattr(training, "write_training_log_csv")

    path = tmp_path / "metrics" / "training_log.csv"

    wrote = training.write_training_log_csv([], path)

    assert wrote is False
    assert not path.exists()
