"""Helpers for converting dataloader batches into model forward inputs."""

from __future__ import annotations

from typing import Any

import torch


def build_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move required sequence tensors and optional static covariates to device."""
    inputs = {
        "intensity": batch["intensity"].to(device),
        "hour_idx": batch["hour_idx"].to(device),
        "day_idx": batch["day_idx"].to(device),
        "mask": batch["mask"].to(device),
    }
    for key in ("static_num", "static_cat", "static_num_missing"):
        if key in batch:
            inputs[key] = batch[key].to(device)
    return inputs
