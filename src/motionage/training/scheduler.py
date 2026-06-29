"""Optimizer and learning-rate scheduler construction helpers."""

from __future__ import annotations

from typing import Any

import torch

from motionage.training.optimizer import OptimizerGroupSummary, build_optimizer_parameter_groups
from motionage.training.task import scheduler_mode


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    list[OptimizerGroupSummary],
]:
    """Create an AdamW optimizer and task-aware ReduceLROnPlateau scheduler."""
    train_cfg = config["training"]
    param_groups, group_summaries = build_optimizer_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode(config),
        patience=int(train_cfg["scheduler_patience"]),
        factor=float(train_cfg["scheduler_factor"]),
    )
    return optimizer, scheduler, group_summaries


def restore_frozen_group_lrs(optimizer: torch.optim.Optimizer) -> int:
    """Restore configured initial learning rates for fully frozen parameter groups."""
    restored_count = 0
    for group in optimizer.param_groups:
        initial_lr = group.get("initial_lr")
        params = group.get("params", [])
        if initial_lr is None or not params:
            continue
        if all(not param.requires_grad for param in params):
            group["lr"] = float(initial_lr)
            restored_count += 1
    return restored_count


def reset_plateau_scheduler_for_stage_transition(scheduler: object) -> bool:
    """Reset ReduceLROnPlateau tracking when a new freeze stage starts."""
    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return False

    scheduler.best = scheduler.mode_worse
    scheduler.num_bad_epochs = 0
    scheduler.cooldown_counter = 0
    scheduler._last_lr = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
    return True
