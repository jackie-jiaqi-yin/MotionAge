from __future__ import annotations

import pytest
import torch

import motionage.training as training


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 3)
        self.head = torch.nn.Linear(3, 1)


def config(
    *,
    task: dict[str, object] | None = None,
    parameter_groups: object | None = None,
) -> dict[str, object]:
    training_cfg: dict[str, object] = {
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-2,
        "scheduler_patience": 3,
        "scheduler_factor": 0.25,
    }
    if parameter_groups is not None:
        training_cfg["parameter_groups"] = parameter_groups
    payload: dict[str, object] = {"training": training_cfg}
    if task is not None:
        payload["task"] = task
    return payload


def test_build_optimizer_and_scheduler_creates_default_adamw_and_min_scheduler() -> None:
    assert hasattr(training, "build_optimizer_and_scheduler")
    model = TinyModel()

    optimizer, scheduler, summaries = training.build_optimizer_and_scheduler(model, config())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1.0e-2)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(1.0e-3)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "min"
    assert scheduler.patience == 3
    assert scheduler.factor == pytest.approx(0.25)
    assert summaries[0].name == "default"


def test_build_optimizer_and_scheduler_preserves_module_parameter_groups() -> None:
    assert hasattr(training, "build_optimizer_and_scheduler")
    model = TinyModel()

    optimizer, scheduler, summaries = training.build_optimizer_and_scheduler(
        model,
        config(parameter_groups=[{"name": "encoder_slow", "module_prefixes": "encoder", "lr_scale": 0.1}]),
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1.0e-3)
    assert scheduler.mode == "min"
    assert [summary.name for summary in summaries] == ["encoder_slow", "default"]


def test_build_optimizer_and_scheduler_uses_task_aware_scheduler_mode() -> None:
    assert hasattr(training, "build_optimizer_and_scheduler")
    model = TinyModel()

    _, auroc_scheduler, _ = training.build_optimizer_and_scheduler(
        model,
        config(task={"type": "binary_classification", "selection_metric": "auroc"}),
    )
    _, logloss_scheduler, _ = training.build_optimizer_and_scheduler(
        model,
        config(task={"type": "binary_classification", "selection_metric": "logloss"}),
    )

    assert auroc_scheduler.mode == "max"
    assert logloss_scheduler.mode == "min"


def test_restore_frozen_group_lrs_resets_only_fully_frozen_groups() -> None:
    assert hasattr(training, "restore_frozen_group_lrs")
    model = TinyModel()
    optimizer, _, _ = training.build_optimizer_and_scheduler(
        model,
        config(parameter_groups=[{"name": "encoder_slow", "module_prefixes": "encoder", "lr_scale": 0.1}]),
    )
    training.apply_freeze_stage(
        model,
        training.FreezeStage(start_epoch=2, freeze_module_prefixes=("encoder",), label="lock_encoder"),
    )
    optimizer.param_groups[0]["lr"] = 1.0e-6
    optimizer.param_groups[1]["lr"] = 2.0e-3

    restored = training.restore_frozen_group_lrs(optimizer)

    assert restored == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2.0e-3)


def test_reset_plateau_scheduler_for_stage_transition_resets_plateau_state() -> None:
    assert hasattr(training, "reset_plateau_scheduler_for_stage_transition")
    model = TinyModel()
    optimizer, scheduler, _ = training.build_optimizer_and_scheduler(model, config())
    optimizer.param_groups[0]["lr"] = 4.0e-4
    scheduler.best = 0.5
    scheduler.num_bad_epochs = 2
    scheduler.cooldown_counter = 1
    scheduler._last_lr = [8.0e-4]

    reset = training.reset_plateau_scheduler_for_stage_transition(scheduler)

    assert reset is True
    assert scheduler.best == scheduler.mode_worse
    assert scheduler.num_bad_epochs == 0
    assert scheduler.cooldown_counter == 0
    assert scheduler._last_lr[0] == pytest.approx(4.0e-4)


def test_reset_plateau_scheduler_for_stage_transition_skips_other_schedulers() -> None:
    assert hasattr(training, "reset_plateau_scheduler_for_stage_transition")
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)

    assert training.reset_plateau_scheduler_for_stage_transition(scheduler) is False
