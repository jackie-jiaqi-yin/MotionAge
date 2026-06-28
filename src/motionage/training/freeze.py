"""Staged module-freezing utilities for fine-tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class FreezeStage:
    """Single staged-freezing configuration."""

    start_epoch: int
    freeze_module_prefixes: tuple[str, ...]
    label: str


def parse_freeze_schedule(raw_schedule: Any) -> tuple[FreezeStage, ...]:
    """Normalize an optional staged-freezing config into sorted stages."""
    if raw_schedule in (None, [], ()):
        return tuple()
    if not isinstance(raw_schedule, list):
        raise ValueError("training.freeze_schedule must be a list when provided.")

    stages: list[FreezeStage] = []
    seen_epochs: set[int] = set()
    for index, raw_stage in enumerate(raw_schedule):
        if not isinstance(raw_stage, dict):
            raise ValueError(f"training.freeze_schedule[{index}] must be a mapping.")

        start_epoch = int(raw_stage.get("start_epoch", 1))
        if start_epoch < 1:
            raise ValueError("training.freeze_schedule start_epoch must be >= 1.")
        if start_epoch in seen_epochs:
            raise ValueError(f"Duplicate freeze schedule start_epoch detected: {start_epoch}")
        seen_epochs.add(start_epoch)

        prefixes = _normalize_prefixes(
            raw_stage.get("freeze_module_prefixes", raw_stage.get("freeze_modules"))
        )
        label = str(raw_stage.get("label") or _default_freeze_stage_label(prefixes))
        stages.append(
            FreezeStage(
                start_epoch=start_epoch,
                freeze_module_prefixes=prefixes,
                label=label,
            )
        )

    return tuple(sorted(stages, key=lambda stage: stage.start_epoch))


def resolve_freeze_stage(
    freeze_schedule: Sequence[FreezeStage],
    epoch: int,
) -> FreezeStage:
    """Resolve the active freeze stage for an epoch."""
    active = FreezeStage(start_epoch=1, freeze_module_prefixes=tuple(), label="all_trainable")
    for stage in freeze_schedule:
        if stage.start_epoch <= epoch:
            active = stage
        else:
            break
    return active


def apply_freeze_stage(model: torch.nn.Module, stage: FreezeStage) -> dict[str, int]:
    """Toggle ``requires_grad`` based on module-prefix matches."""
    named_params = list(model.named_parameters())
    names = [name for name, _ in named_params]
    module_names = [name for name, _ in model.named_modules()]
    _validate_prefix_matches(
        prefixes=stage.freeze_module_prefixes,
        names=names,
        module_names=module_names,
        context=f"freeze stage '{stage.label}'",
    )

    frozen_param_count = 0
    frozen_tensor_count = 0
    trainable_param_count = 0
    trainable_tensor_count = 0

    for name, param in named_params:
        should_freeze = any(_matches_module_prefix(name, prefix) for prefix in stage.freeze_module_prefixes)
        param.requires_grad = not should_freeze
        if should_freeze:
            frozen_param_count += int(param.numel())
            frozen_tensor_count += 1
        else:
            trainable_param_count += int(param.numel())
            trainable_tensor_count += 1

    return {
        "frozen_param_count": frozen_param_count,
        "frozen_tensor_count": frozen_tensor_count,
        "trainable_param_count": trainable_param_count,
        "trainable_tensor_count": trainable_tensor_count,
    }


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count currently trainable parameters."""
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _normalize_prefixes(raw_prefixes: Any) -> tuple[str, ...]:
    if raw_prefixes in (None, "", "null"):
        return tuple()
    if isinstance(raw_prefixes, str):
        values = [raw_prefixes]
    else:
        values = list(raw_prefixes)

    normalized: list[str] = []
    for raw_prefix in values:
        prefix = str(raw_prefix).strip()
        if prefix and prefix not in normalized:
            normalized.append(prefix)
    return tuple(normalized)


def _matches_module_prefix(parameter_name: str, prefix: str) -> bool:
    return parameter_name == prefix or parameter_name.startswith(f"{prefix}.")


def _validate_prefix_matches(
    *,
    prefixes: Sequence[str],
    names: Sequence[str],
    module_names: Sequence[str],
    context: str,
) -> None:
    unmatched = [
        prefix
        for prefix in prefixes
        if not any(_matches_module_prefix(name, prefix) for name in names) and prefix not in module_names
    ]
    if unmatched:
        raise ValueError(f"{context} references unknown module prefixes: {unmatched}")


def _default_freeze_stage_label(prefixes: Sequence[str]) -> str:
    if not prefixes:
        return "all_trainable"
    return "freeze_" + "_".join(prefix.replace(".", "_") for prefix in prefixes)
