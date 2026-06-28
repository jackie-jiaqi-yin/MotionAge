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
