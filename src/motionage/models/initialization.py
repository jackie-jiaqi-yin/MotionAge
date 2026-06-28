"""Checkpoint-based model initialization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class InitializationSummary:
    """Summary of checkpoint initialization behavior."""

    applied: bool
    checkpoint_path: str | None = None
    source_tensor_count: int = 0
    target_tensor_count: int = 0
    loaded_tensor_count: int = 0
    loaded_param_count: int = 0
    untouched_target_count: int = 0
    skipped_missing_count: int = 0
    skipped_shape_count: int = 0


def initialize_model_from_config(
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> InitializationSummary:
    """Optionally initialize a model from `experiment.init_checkpoint`."""
    experiment_config = dict(config.get("experiment", {}))
    checkpoint_path = experiment_config.get("init_checkpoint")
    if checkpoint_path in (None, "", "null"):
        return InitializationSummary(applied=False)

    path = Path(str(checkpoint_path))
    if not path.exists():
        raise FileNotFoundError(f"experiment.init_checkpoint not found: {path}")

    strict = bool(experiment_config.get("init_strict", False))
    return initialize_model_from_checkpoint(
        model=model,
        checkpoint_path=path,
        device=device,
        strict=strict,
    )


def initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = False,
) -> InitializationSummary:
    """Initialize model weights from a checkpoint state dictionary."""
    path = Path(checkpoint_path)
    loaded = torch.load(path, map_location=device, weights_only=True)
    source_state = _normalize_state_dict(loaded)
    model_state = model.state_dict()

    if strict:
        model.load_state_dict(source_state, strict=True)
        loaded_param_count = int(sum(tensor.numel() for tensor in source_state.values()))
        return InitializationSummary(
            applied=True,
            checkpoint_path=str(path),
            source_tensor_count=len(source_state),
            target_tensor_count=len(model_state),
            loaded_tensor_count=len(source_state),
            loaded_param_count=loaded_param_count,
        )

    compatible: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    for key, tensor in source_state.items():
        target = model_state.get(key)
        if target is None:
            skipped_missing += 1
            continue
        if target.shape != tensor.shape:
            skipped_shape += 1
            continue
        compatible[key] = tensor

    model.load_state_dict(compatible, strict=False)
    loaded_param_count = int(sum(tensor.numel() for tensor in compatible.values()))
    return InitializationSummary(
        applied=True,
        checkpoint_path=str(path),
        source_tensor_count=len(source_state),
        target_tensor_count=len(model_state),
        loaded_tensor_count=len(compatible),
        loaded_param_count=loaded_param_count,
        untouched_target_count=len(model_state) - len(compatible),
        skipped_missing_count=skipped_missing,
        skipped_shape_count=skipped_shape,
    )


def _normalize_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
        state = raw["model_state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise TypeError("Checkpoint must be a state_dict or a dict containing 'model_state_dict'.")

    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        normalized[key.replace("gru.", "recurrent.")] = tensor
    return normalized
