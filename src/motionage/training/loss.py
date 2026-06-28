"""Task-aware loss construction helpers."""

from __future__ import annotations

from typing import Any

import torch

from motionage.training.task import BINARY_CLASSIFICATION, resolve_task_type


def build_loss_function(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Create a PyTorch loss function for regression or binary classification."""
    if resolve_task_type(config) == BINARY_CLASSIFICATION:
        task_cfg = config.get("task", {})
        pos_weight = task_cfg.get("pos_weight") if isinstance(task_cfg, dict) else None
        if pos_weight is None:
            return torch.nn.BCEWithLogitsLoss()
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight), device=device, dtype=torch.float32)
        )
    return torch.nn.MSELoss()
