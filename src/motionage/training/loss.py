"""Task-aware loss construction helpers."""

from __future__ import annotations

from typing import Any

import torch

from motionage.training.task import BINARY_CLASSIFICATION, REGRESSION, TaskType, resolve_task_type


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


def compute_validation_loss(
    *,
    predictions: Any,
    targets: Any,
    task_type: TaskType | str,
    criterion: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
) -> float:
    """Compute validation loss from model outputs and targets."""
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    pred_tensor = torch.as_tensor(predictions, dtype=torch.float32, device=resolved_device)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=resolved_device)

    if task_type == REGRESSION:
        return float(torch.mean((target_tensor - pred_tensor) ** 2).item())

    if task_type == BINARY_CLASSIFICATION:
        loss_function = criterion if criterion is not None else torch.nn.BCEWithLogitsLoss()
        return float(loss_function(pred_tensor, target_tensor).item())

    raise ValueError(f"Unsupported task_type '{task_type}'.")
