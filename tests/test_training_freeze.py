from __future__ import annotations

import pytest
import torch

import motionage.training as training


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
        self.head = torch.nn.Linear(3, 1)


def test_parse_freeze_schedule_normalizes_and_sorts_stages() -> None:
    assert hasattr(training, "FreezeStage")
    assert hasattr(training, "parse_freeze_schedule")

    stages = training.parse_freeze_schedule(
        [
            {"start_epoch": 4, "freeze_modules": ["encoder", "encoder"], "label": "lock_encoder"},
            {"start_epoch": 1, "freeze_module_prefixes": None},
            {"start_epoch": 8, "freeze_module_prefixes": "head"},
        ]
    )

    assert stages == (
        training.FreezeStage(start_epoch=1, freeze_module_prefixes=tuple(), label="all_trainable"),
        training.FreezeStage(start_epoch=4, freeze_module_prefixes=("encoder",), label="lock_encoder"),
        training.FreezeStage(start_epoch=8, freeze_module_prefixes=("head",), label="freeze_head"),
    )


def test_parse_freeze_schedule_rejects_invalid_entries() -> None:
    assert hasattr(training, "parse_freeze_schedule")

    assert training.parse_freeze_schedule(None) == tuple()
    with pytest.raises(ValueError, match="must be a list"):
        training.parse_freeze_schedule({"start_epoch": 1})
    with pytest.raises(ValueError, match="must be a mapping"):
        training.parse_freeze_schedule(["encoder"])
    with pytest.raises(ValueError, match="start_epoch"):
        training.parse_freeze_schedule([{"start_epoch": 0}])
    with pytest.raises(ValueError, match="Duplicate"):
        training.parse_freeze_schedule([{"start_epoch": 1}, {"start_epoch": 1}])


def test_resolve_freeze_stage_selects_latest_stage_at_epoch() -> None:
    assert hasattr(training, "FreezeStage")
    assert hasattr(training, "resolve_freeze_stage")
    schedule = (
        training.FreezeStage(start_epoch=3, freeze_module_prefixes=("encoder",), label="lock_encoder"),
        training.FreezeStage(start_epoch=6, freeze_module_prefixes=tuple(), label="unfreeze"),
    )

    assert training.resolve_freeze_stage(schedule, 1) == training.FreezeStage(
        start_epoch=1,
        freeze_module_prefixes=tuple(),
        label="all_trainable",
    )
    assert training.resolve_freeze_stage(schedule, 3).label == "lock_encoder"
    assert training.resolve_freeze_stage(schedule, 7).label == "unfreeze"


def test_apply_freeze_stage_toggles_matching_module_prefixes() -> None:
    assert hasattr(training, "FreezeStage")
    assert hasattr(training, "apply_freeze_stage")
    assert hasattr(training, "count_trainable_parameters")
    model = TinyModel()

    summary = training.apply_freeze_stage(
        model,
        training.FreezeStage(start_epoch=2, freeze_module_prefixes=("encoder",), label="lock_encoder"),
    )

    assert summary["frozen_tensor_count"] == 2
    assert summary["trainable_tensor_count"] == 2
    assert summary["frozen_param_count"] == sum(param.numel() for param in model.encoder.parameters())
    assert summary["trainable_param_count"] == training.count_trainable_parameters(model)
    assert all(not param.requires_grad for param in model.encoder.parameters())
    assert all(param.requires_grad for param in model.head.parameters())

    unfreeze_summary = training.apply_freeze_stage(
        model,
        training.FreezeStage(start_epoch=3, freeze_module_prefixes=tuple(), label="all_trainable"),
    )

    assert unfreeze_summary["frozen_tensor_count"] == 0
    assert all(param.requires_grad for param in model.parameters())


def test_apply_freeze_stage_rejects_unknown_module_prefix() -> None:
    assert hasattr(training, "FreezeStage")
    assert hasattr(training, "apply_freeze_stage")
    model = TinyModel()

    with pytest.raises(ValueError, match="unknown module prefixes"):
        training.apply_freeze_stage(
            model,
            training.FreezeStage(start_epoch=1, freeze_module_prefixes=("missing",), label="bad"),
        )
