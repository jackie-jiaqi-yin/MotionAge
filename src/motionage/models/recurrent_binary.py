"""Mask-aware recurrent classifiers for mortality prediction."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from motionage.models.common import (
    PredictionMode,
    build_mlp,
    normalize_prediction_mode,
    validate_categorical_cardinalities,
    validate_sequence_inputs,
    validate_static_inputs,
    zero_last_linear,
)

RecurrentType = Literal["gru", "lstm"]


def _resolve_recurrent_class(recurrent_type: str) -> type[nn.GRU] | type[nn.LSTM]:
    normalized = recurrent_type.lower()
    if normalized == "gru":
        return nn.GRU
    if normalized == "lstm":
        return nn.LSTM
    raise ValueError(f"Unsupported recurrent_type '{recurrent_type}'. Expected 'gru' or 'lstm'.")


class MaskedRecurrentClassifier(nn.Module):
    """Recurrent encoder with masked temporal pooling and a single-logit head."""

    def __init__(
        self,
        recurrent_type: RecurrentType,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        hour_emb_dim: int = 8,
        day_emb_dim: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if hour_emb_dim <= 0:
            raise ValueError(f"hour_emb_dim must be positive, got {hour_emb_dim}.")
        if day_emb_dim <= 0:
            raise ValueError(f"day_emb_dim must be positive, got {day_emb_dim}.")

        recurrent_cls = _resolve_recurrent_class(recurrent_type)
        self.recurrent_type = recurrent_type
        input_size = 1 + hour_emb_dim + day_emb_dim

        self.hour_embedding = nn.Embedding(num_embeddings=24, embedding_dim=hour_emb_dim)
        self.day_embedding = nn.Embedding(num_embeddings=7, embedding_dim=day_emb_dim)
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        intensity: torch.Tensor,
        hour_idx: torch.Tensor,
        day_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode_wearable(
            intensity=intensity,
            hour_idx=hour_idx,
            day_idx=day_idx,
            mask=mask,
        )
        return self.classifier(pooled).squeeze(-1)

    def encode_wearable(
        self,
        intensity: torch.Tensor,
        hour_idx: torch.Tensor,
        day_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the temporal wearable sequence into a pooled hidden state."""
        validate_sequence_inputs(intensity=intensity, hour_idx=hour_idx, day_idx=day_idx, mask=mask)
        intensity_input = intensity.unsqueeze(-1)
        hour_emb = self.hour_embedding(hour_idx)
        day_emb = self.day_embedding(day_idx)

        x = torch.cat([intensity_input, hour_emb, day_emb], dim=-1)
        h_seq, _ = self.recurrent(x)

        pool_mask = mask.unsqueeze(-1).to(h_seq.dtype)
        masked_h = h_seq * pool_mask
        denom = pool_mask.sum(dim=1).clamp(min=1.0)
        return masked_h.sum(dim=1) / denom


class MaskedRecurrentCovariateClassifier(nn.Module):
    """Recurrent encoder with a static-covariate branch and binary fusion head."""

    def __init__(
        self,
        recurrent_type: RecurrentType,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        hour_emb_dim: int = 8,
        day_emb_dim: int = 4,
        num_numeric_features: int = 0,
        categorical_cardinalities: list[int] | tuple[int, ...] | None = None,
        categorical_embedding_dim: int = 4,
        covariate_hidden_dim: int = 64,
        covariate_output_dim: int = 32,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.1,
        prediction_mode: PredictionMode = "late_fusion",
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if num_numeric_features < 0:
            raise ValueError(f"num_numeric_features must be non-negative, got {num_numeric_features}.")
        if categorical_embedding_dim <= 0:
            raise ValueError(f"categorical_embedding_dim must be positive, got {categorical_embedding_dim}.")
        if covariate_output_dim <= 0:
            raise ValueError(f"covariate_output_dim must be positive, got {covariate_output_dim}.")

        recurrent_cls = _resolve_recurrent_class(recurrent_type)
        self.recurrent_type = recurrent_type
        self.hidden_size = hidden_size
        self.num_numeric_features = int(num_numeric_features)
        self.categorical_cardinalities = validate_categorical_cardinalities(categorical_cardinalities)
        self.prediction_mode = normalize_prediction_mode(prediction_mode)

        input_size = 1 + hour_emb_dim + day_emb_dim
        self.hour_embedding = nn.Embedding(num_embeddings=24, embedding_dim=hour_emb_dim)
        self.day_embedding = nn.Embedding(num_embeddings=7, embedding_dim=day_emb_dim)
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier: nn.Linear | None = None
        if self.prediction_mode == "residual":
            self.classifier = nn.Linear(hidden_size, 1)

        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=cardinality, embedding_dim=categorical_embedding_dim)
                for cardinality in self.categorical_cardinalities
            ]
        )

        categorical_width = len(self.categorical_cardinalities) * categorical_embedding_dim
        static_input_dim = (2 * self.num_numeric_features) + categorical_width
        if static_input_dim <= 0:
            raise ValueError("The covariate model requires at least one static numeric or categorical feature.")

        self.covariate_encoder = build_mlp(
            input_dim=static_input_dim,
            hidden_dim=covariate_hidden_dim,
            output_dim=covariate_output_dim,
            dropout=fusion_dropout,
        )
        self.fusion_mlp = build_mlp(
            input_dim=hidden_size + covariate_output_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=1,
            dropout=fusion_dropout,
        )

        if self.prediction_mode == "residual" and zero_init_residual:
            zero_last_linear(self.fusion_mlp)

    def forward(
        self,
        intensity: torch.Tensor,
        hour_idx: torch.Tensor,
        day_idx: torch.Tensor,
        mask: torch.Tensor,
        static_num: torch.Tensor,
        static_cat: torch.Tensor,
        static_num_missing: torch.Tensor,
    ) -> torch.Tensor:
        h_pool = self.encode_wearable(
            intensity=intensity,
            hour_idx=hour_idx,
            day_idx=day_idx,
            mask=mask,
        )
        s_cov = self.encode_covariates(
            static_num=static_num,
            static_cat=static_cat,
            static_num_missing=static_num_missing,
        )
        fused = torch.cat([h_pool, s_cov], dim=-1)

        if self.prediction_mode == "late_fusion":
            return self.fusion_mlp(fused).squeeze(-1)

        if self.classifier is None:
            raise RuntimeError("prediction_mode='residual' requires a classifier head.")
        base_logit = self.classifier(h_pool).squeeze(-1)
        delta_logit = self.fusion_mlp(fused).squeeze(-1)
        return base_logit + delta_logit

    def encode_wearable(
        self,
        intensity: torch.Tensor,
        hour_idx: torch.Tensor,
        day_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the temporal wearable sequence into a pooled hidden state."""
        validate_sequence_inputs(intensity=intensity, hour_idx=hour_idx, day_idx=day_idx, mask=mask)
        intensity_input = intensity.unsqueeze(-1)
        hour_emb = self.hour_embedding(hour_idx)
        day_emb = self.day_embedding(day_idx)

        x = torch.cat([intensity_input, hour_emb, day_emb], dim=-1)
        h_seq, _ = self.recurrent(x)

        pool_mask = mask.unsqueeze(-1).to(h_seq.dtype)
        masked_h = h_seq * pool_mask
        denom = pool_mask.sum(dim=1).clamp(min=1.0)
        return masked_h.sum(dim=1) / denom

    def encode_covariates(
        self,
        static_num: torch.Tensor,
        static_cat: torch.Tensor,
        static_num_missing: torch.Tensor,
    ) -> torch.Tensor:
        """Encode static covariates into a compact fused representation."""
        validate_static_inputs(
            static_num=static_num,
            static_cat=static_cat,
            static_num_missing=static_num_missing,
            num_numeric_features=self.num_numeric_features,
            num_categorical_features=len(self.categorical_cardinalities),
        )
        parts: list[torch.Tensor] = [static_num, static_num_missing]

        embedded_parts: list[torch.Tensor] = []
        for index, embedding in enumerate(self.categorical_embeddings):
            embedded_parts.append(embedding(static_cat[:, index]))
        if embedded_parts:
            parts.append(torch.cat(embedded_parts, dim=-1))

        covariate_input = torch.cat(parts, dim=-1)
        return self.covariate_encoder(covariate_input)
