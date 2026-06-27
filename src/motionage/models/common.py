"""Shared helpers for MotionAge model modules."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

PredictionMode = Literal["residual", "late_fusion"]


def normalize_prediction_mode(prediction_mode: str) -> PredictionMode:
    """Normalize and validate a static-covariate prediction mode."""
    normalized = prediction_mode.lower()
    if normalized not in {"residual", "late_fusion"}:
        raise ValueError(
            f"prediction_mode must be 'residual' or 'late_fusion', got '{prediction_mode}'."
        )
    return normalized  # type: ignore[return-value]


def validate_categorical_cardinalities(
    categorical_cardinalities: list[int] | tuple[int, ...] | None,
) -> list[int]:
    """Return positive integer categorical cardinalities."""
    values = [int(value) for value in (categorical_cardinalities or [])]
    for idx, cardinality in enumerate(values):
        if cardinality <= 0:
            raise ValueError(f"categorical_cardinalities[{idx}] must be positive, got {cardinality}.")
    return values


def build_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    """Build either a linear layer or a small 2-layer MLP depending on hidden_dim."""
    if hidden_dim <= 0:
        return nn.Linear(input_dim, output_dim)

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def zero_last_linear(module: nn.Module) -> None:
    """Zero-initialize the last linear layer of a residual delta head."""
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linear_layers:
        raise ValueError("Expected at least one linear layer to zero-initialize.")
    last_linear = linear_layers[-1]
    nn.init.zeros_(last_linear.weight)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


def validate_sequence_inputs(
    *,
    intensity: torch.Tensor,
    hour_idx: torch.Tensor,
    day_idx: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate shared wearable sequence input shapes."""
    if intensity.ndim != 2:
        raise ValueError(f"intensity must have shape [batch, seq], got {tuple(intensity.shape)}.")
    expected_shape = intensity.shape
    for name, tensor in {"hour_idx": hour_idx, "day_idx": day_idx, "mask": mask}.items():
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}.")


def validate_static_inputs(
    *,
    static_num: torch.Tensor,
    static_cat: torch.Tensor,
    static_num_missing: torch.Tensor,
    num_numeric_features: int,
    num_categorical_features: int,
) -> None:
    """Validate shared static covariate input shapes."""
    if static_num.ndim != 2:
        raise ValueError(f"static_num must have shape [batch, features], got {tuple(static_num.shape)}.")
    if static_num_missing.shape != static_num.shape:
        raise ValueError(
            "static_num_missing must match static_num shape, "
            f"got {tuple(static_num_missing.shape)} and {tuple(static_num.shape)}."
        )
    if static_num.shape[1] != num_numeric_features:
        raise ValueError(
            f"static_num has {static_num.shape[1]} columns, expected {num_numeric_features}."
        )
    if static_cat.ndim != 2:
        raise ValueError(f"static_cat must have shape [batch, features], got {tuple(static_cat.shape)}.")
    if static_cat.shape[0] != static_num.shape[0]:
        raise ValueError("static_cat and static_num must have the same batch size.")
    if static_cat.shape[1] != num_categorical_features:
        raise ValueError(
            f"static_cat has {static_cat.shape[1]} columns, expected {num_categorical_features}."
        )
