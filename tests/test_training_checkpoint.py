from __future__ import annotations

import math

import pytest
import torch

import motionage.training as training


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)


def make_optimizer_and_scheduler(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    return optimizer, scheduler


def test_build_checkpoint_payload_includes_shared_and_regression_fields() -> None:
    assert hasattr(training, "build_checkpoint_payload")
    model = TinyModel()
    optimizer, scheduler = make_optimizer_and_scheduler(model)

    payload = training.build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        task_type=training.REGRESSION,
        selection_metric_name="mae",
        selection_metric_direction="minimize",
        best_val_metric=1.2,
        best_val_loss=1.44,
        best_eval_metrics={"mae": 1.2, "r2": 0.4},
        best_epoch=2,
        patience_counter=1,
        training_log=[{"epoch": 1, "train_loss": 2.0}],
    )

    assert payload["epoch"] == 3
    assert set(payload["model_state_dict"]) == set(model.state_dict())
    assert "state" in payload["optimizer_state_dict"]
    assert "best" in payload["scheduler_state_dict"]
    assert payload["task_type"] == training.REGRESSION
    assert payload["selection_metric_name"] == "mae"
    assert payload["selection_metric_direction"] == "minimize"
    assert payload["best_val_metric"] == 1.2
    assert payload["best_val_loss"] == 1.44
    assert payload["best_eval_metrics"] == {"mae": 1.2, "r2": 0.4}
    assert payload["best_epoch"] == 2
    assert payload["patience_counter"] == 1
    assert payload["training_log"] == [{"epoch": 1, "train_loss": 2.0}]
    assert payload["val_mae"] == 1.2
    assert payload["val_r2"] == 0.4


def test_build_checkpoint_payload_includes_binary_validation_fields() -> None:
    assert hasattr(training, "build_checkpoint_payload")
    model = TinyModel()
    optimizer, scheduler = make_optimizer_and_scheduler(model)

    payload = training.build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        task_type=training.BINARY_CLASSIFICATION,
        selection_metric_name="auroc",
        selection_metric_direction="maximize",
        best_val_metric=0.81,
        best_val_loss=0.52,
        best_eval_metrics={
            "auroc": 0.81,
            "auprc": 0.44,
            "logloss": 0.52,
            "brier": 0.18,
        },
        best_epoch=4,
        patience_counter=0,
        training_log=[],
    )

    assert payload["task_type"] == training.BINARY_CLASSIFICATION
    assert payload["selection_metric_direction"] == "maximize"
    assert payload["best_val_metric"] == 0.81
    assert payload["val_auroc"] == 0.81
    assert payload["val_auprc"] == 0.44
    assert payload["val_logloss"] == 0.52
    assert payload["val_brier"] == 0.18


def test_build_checkpoint_resume_state_extracts_regression_fields() -> None:
    assert hasattr(training, "build_checkpoint_resume_state")

    state = training.build_checkpoint_resume_state(
        {
            "epoch": 8,
            "best_val_metric": 1.3,
            "best_val_loss": 1.69,
            "best_eval_metrics": {"mae": 1.3, "r2": 0.42},
            "best_epoch": 6,
            "patience_counter": 2,
            "training_log": [{"epoch": 1, "train_loss": 2.0}],
            "val_mae": 1.3,
            "val_r2": 0.42,
        },
        task_type=training.REGRESSION,
    )

    assert state == {
        "resumed_from_epoch": 8,
        "best_val_metric": 1.3,
        "best_val_loss": 1.69,
        "best_eval_metrics": {"mae": 1.3, "r2": 0.42},
        "best_epoch": 6,
        "patience_counter": 2,
        "training_log": [{"epoch": 1, "train_loss": 2.0}],
        "best_val_mae": 1.3,
        "best_val_r2": 0.42,
    }


def test_build_checkpoint_resume_state_extracts_binary_fields() -> None:
    assert hasattr(training, "build_checkpoint_resume_state")

    state = training.build_checkpoint_resume_state(
        {
            "epoch": 9,
            "best_val_metric": 0.83,
            "best_val_loss": 0.48,
            "best_eval_metrics": {"auroc": 0.83, "auprc": 0.51, "logloss": 0.48, "brier": 0.16},
            "best_epoch": 7,
            "patience_counter": 1,
            "training_log": [],
            "val_auroc": 0.83,
            "val_auprc": 0.51,
            "val_logloss": 0.48,
            "val_brier": 0.16,
        },
        task_type=training.BINARY_CLASSIFICATION,
    )

    assert state["resumed_from_epoch"] == 9
    assert state["best_val_metric"] == 0.83
    assert state["best_eval_metrics"]["auprc"] == 0.51
    assert state["best_val_auroc"] == 0.83
    assert state["best_val_auprc"] == 0.51
    assert state["best_val_logloss"] == 0.48
    assert state["best_val_brier"] == 0.16


def test_build_checkpoint_resume_state_uses_safe_defaults_for_missing_values() -> None:
    assert hasattr(training, "build_checkpoint_resume_state")

    state = training.build_checkpoint_resume_state({}, task_type=training.REGRESSION)

    assert state["resumed_from_epoch"] == 0
    assert math.isnan(state["best_val_metric"])
    assert math.isnan(state["best_val_loss"])
    assert state["best_eval_metrics"] == {}
    assert state["best_epoch"] == 0
    assert state["patience_counter"] == 0
    assert state["training_log"] == []
    assert math.isnan(state["best_val_mae"])
    assert math.isnan(state["best_val_r2"])


def test_build_checkpoint_resume_state_rejects_unknown_task_type() -> None:
    assert hasattr(training, "build_checkpoint_resume_state")

    with pytest.raises(ValueError, match="Unsupported task_type"):
        training.build_checkpoint_resume_state({}, task_type="unsupported")
