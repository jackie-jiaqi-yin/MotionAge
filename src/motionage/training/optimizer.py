"""Optimizer parameter-group helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from motionage.training.freeze import (
    _matches_module_prefix,
    _normalize_prefixes,
    _validate_prefix_matches,
)


@dataclass(frozen=True)
class OptimizerGroupSummary:
    """Human-readable summary of an optimizer parameter group."""

    name: str
    lr: float
    weight_decay: float
    param_count: int
    tensor_count: int
    module_prefixes: tuple[str, ...]


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[OptimizerGroupSummary]]:
    """Build AdamW-compatible parameter groups from optional module-prefix config."""
    train_cfg = config["training"]
    base_lr = float(train_cfg["learning_rate"])
    base_weight_decay = float(train_cfg["weight_decay"])
    raw_groups = train_cfg.get("parameter_groups")
    named_params = list(model.named_parameters())
    module_names = [name for name, _ in model.named_modules()]

    if not raw_groups:
        default_params = [param for _, param in named_params]
        return (
            [
                {
                    "params": default_params,
                    "lr": base_lr,
                    "weight_decay": base_weight_decay,
                    "initial_lr": base_lr,
                }
            ],
            [
                OptimizerGroupSummary(
                    name="default",
                    lr=base_lr,
                    weight_decay=base_weight_decay,
                    param_count=int(sum(param.numel() for param in default_params)),
                    tensor_count=len(default_params),
                    module_prefixes=tuple(),
                )
            ],
        )

    if not isinstance(raw_groups, list):
        raise ValueError("training.parameter_groups must be a list when provided.")

    assigned_names: set[str] = set()
    param_groups: list[dict[str, Any]] = []
    summaries: list[OptimizerGroupSummary] = []

    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError(f"training.parameter_groups[{index}] must be a mapping.")

        prefixes = _normalize_prefixes(raw_group.get("module_prefixes"))
        if not prefixes:
            raise ValueError(f"training.parameter_groups[{index}] must define at least one module_prefix.")

        matches = [
            (name, param)
            for name, param in named_params
            if any(_matches_module_prefix(name, prefix) for prefix in prefixes)
        ]
        _validate_prefix_matches(
            prefixes=prefixes,
            names=[name for name, _ in named_params],
            module_names=module_names,
            context=f"training.parameter_groups[{index}]",
        )

        duplicate_names = [name for name, _ in matches if name in assigned_names]
        if duplicate_names:
            raise ValueError(
                f"training.parameter_groups[{index}] reuses parameters already assigned "
                f"to another group: {duplicate_names[:5]}"
            )

        params = [param for _, param in matches]
        name = str(raw_group.get("name", f"group_{index}"))
        lr = float(raw_group["lr"]) if "lr" in raw_group else base_lr * float(raw_group.get("lr_scale", 1.0))
        weight_decay = float(raw_group.get("weight_decay", base_weight_decay))

        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
                "initial_lr": lr,
            }
        )
        summaries.append(
            OptimizerGroupSummary(
                name=name,
                lr=lr,
                weight_decay=weight_decay,
                param_count=int(sum(param.numel() for param in params)),
                tensor_count=len(params),
                module_prefixes=prefixes,
            )
        )
        assigned_names.update(name for name, _ in matches)

    remaining = [(name, param) for name, param in named_params if name not in assigned_names]
    if remaining:
        params = [param for _, param in remaining]
        param_groups.append(
            {
                "params": params,
                "lr": base_lr,
                "weight_decay": base_weight_decay,
                "initial_lr": base_lr,
            }
        )
        summaries.append(
            OptimizerGroupSummary(
                name="default",
                lr=base_lr,
                weight_decay=base_weight_decay,
                param_count=int(sum(param.numel() for param in params)),
                tensor_count=len(params),
                module_prefixes=tuple(),
            )
        )

    return param_groups, summaries
