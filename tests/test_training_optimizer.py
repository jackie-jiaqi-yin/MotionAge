from __future__ import annotations

import pytest
import torch

import motionage.training as training


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 3)
        self.head = torch.nn.Linear(3, 1)


def base_config(parameter_groups: object | None = None) -> dict[str, object]:
    training_cfg: dict[str, object] = {
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-2,
    }
    if parameter_groups is not None:
        training_cfg["parameter_groups"] = parameter_groups
    return {"training": training_cfg}


def group_param_count(group: dict[str, object]) -> int:
    params = group["params"]
    assert isinstance(params, list)
    return sum(param.numel() for param in params)


def test_build_optimizer_parameter_groups_uses_single_default_group_without_overrides() -> None:
    assert hasattr(training, "OptimizerGroupSummary")
    assert hasattr(training, "build_optimizer_parameter_groups")
    model = TinyModel()

    groups, summaries = training.build_optimizer_parameter_groups(model, base_config())

    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(1.0e-3)
    assert groups[0]["weight_decay"] == pytest.approx(1.0e-2)
    assert groups[0]["initial_lr"] == pytest.approx(1.0e-3)
    assert group_param_count(groups[0]) == sum(param.numel() for param in model.parameters())
    assert summaries == [
        training.OptimizerGroupSummary(
            name="default",
            lr=1.0e-3,
            weight_decay=1.0e-2,
            param_count=sum(param.numel() for param in model.parameters()),
            tensor_count=4,
            module_prefixes=tuple(),
        )
    ]


def test_build_optimizer_parameter_groups_applies_module_prefix_overrides_and_default_remainder() -> None:
    assert hasattr(training, "OptimizerGroupSummary")
    assert hasattr(training, "build_optimizer_parameter_groups")
    model = TinyModel()

    groups, summaries = training.build_optimizer_parameter_groups(
        model,
        base_config(
            [
                {
                    "name": "encoder_slow",
                    "module_prefixes": ["encoder", "encoder"],
                    "lr_scale": 0.1,
                    "weight_decay": 0.0,
                }
            ]
        ),
    )

    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1.0e-4)
    assert groups[0]["weight_decay"] == pytest.approx(0.0)
    assert groups[0]["initial_lr"] == pytest.approx(1.0e-4)
    assert group_param_count(groups[0]) == sum(param.numel() for param in model.encoder.parameters())
    assert groups[1]["lr"] == pytest.approx(1.0e-3)
    assert group_param_count(groups[1]) == sum(param.numel() for param in model.head.parameters())
    assert summaries[0] == training.OptimizerGroupSummary(
        name="encoder_slow",
        lr=1.0e-4,
        weight_decay=0.0,
        param_count=sum(param.numel() for param in model.encoder.parameters()),
        tensor_count=2,
        module_prefixes=("encoder",),
    )
    assert summaries[1].name == "default"


def test_build_optimizer_parameter_groups_accepts_explicit_learning_rate() -> None:
    assert hasattr(training, "build_optimizer_parameter_groups")
    model = TinyModel()

    groups, summaries = training.build_optimizer_parameter_groups(
        model,
        base_config([{"name": "head_fast", "module_prefixes": "head", "lr": 2.0e-3}]),
    )

    assert groups[0]["lr"] == pytest.approx(2.0e-3)
    assert summaries[0].module_prefixes == ("head",)


def test_build_optimizer_parameter_groups_rejects_invalid_or_overlapping_groups() -> None:
    assert hasattr(training, "build_optimizer_parameter_groups")
    model = TinyModel()

    with pytest.raises(ValueError, match="must be a list"):
        training.build_optimizer_parameter_groups(model, base_config({"module_prefixes": "encoder"}))
    with pytest.raises(ValueError, match="must be a mapping"):
        training.build_optimizer_parameter_groups(model, base_config(["encoder"]))
    with pytest.raises(ValueError, match="at least one module_prefix"):
        training.build_optimizer_parameter_groups(model, base_config([{"name": "empty"}]))
    with pytest.raises(ValueError, match="unknown module prefixes"):
        training.build_optimizer_parameter_groups(model, base_config([{"module_prefixes": "missing"}]))
    with pytest.raises(ValueError, match="reuses parameters"):
        training.build_optimizer_parameter_groups(
            model,
            base_config(
                [
                    {"name": "encoder_a", "module_prefixes": "encoder"},
                    {"name": "encoder_b", "module_prefixes": "encoder"},
                ]
            ),
        )
