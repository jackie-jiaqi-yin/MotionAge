"""Transformer binary classifier with participant-level static covariate fusion."""

from __future__ import annotations

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


class MaskedTransformerCovariateClassifier(nn.Module):
    """Transformer encoder with a static-covariate branch and binary fusion head."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        hour_emb_dim: int = 8,
        day_emb_dim: int = 4,
        intensity_proj_dim: int = 16,
        max_seq_len: int = 1008,
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

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if nhead <= 0:
            raise ValueError(f"nhead must be positive, got {nhead}.")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        if num_numeric_features < 0:
            raise ValueError(f"num_numeric_features must be non-negative, got {num_numeric_features}.")
        if categorical_embedding_dim <= 0:
            raise ValueError(f"categorical_embedding_dim must be positive, got {categorical_embedding_dim}.")
        if covariate_output_dim <= 0:
            raise ValueError(f"covariate_output_dim must be positive, got {covariate_output_dim}.")

        self.max_seq_len = max_seq_len
        self.num_numeric_features = int(num_numeric_features)
        self.categorical_cardinalities = validate_categorical_cardinalities(categorical_cardinalities)
        self.prediction_mode = normalize_prediction_mode(prediction_mode)

        self.hour_embedding = nn.Embedding(num_embeddings=24, embedding_dim=hour_emb_dim)
        self.day_embedding = nn.Embedding(num_embeddings=7, embedding_dim=day_emb_dim)
        self.intensity_projection = nn.Linear(1, intensity_proj_dim)
        self.input_projection = nn.Linear(intensity_proj_dim + hour_emb_dim + day_emb_dim, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.classifier: nn.Linear | None = None
        if self.prediction_mode == "residual":
            self.classifier = nn.Linear(d_model, 1)

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
            input_dim=d_model + covariate_output_dim,
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
        """Encode the temporal wearable sequence into a pooled representation."""
        validate_sequence_inputs(intensity=intensity, hour_idx=hour_idx, day_idx=day_idx, mask=mask)
        batch_size, seq_len = intensity.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        intensity_proj = self.intensity_projection(intensity.unsqueeze(-1))
        hour_emb = self.hour_embedding(hour_idx)
        day_emb = self.day_embedding(day_idx)

        x = torch.cat([intensity_proj, hour_emb, day_emb], dim=-1)
        x = self.input_projection(x)

        position_ids = torch.arange(seq_len, device=intensity.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(position_ids)
        x = self.input_dropout(self.input_norm(x))

        # attention_flag marks low-coverage intervals, not true padding.
        # Keep those timesteps in the contextual encoder and only exclude them
        # from the pooled summary so Transformer and recurrent models share the
        # same mask contract.
        h = self.encoder(x)

        pool_mask = mask.unsqueeze(-1).to(h.dtype)
        masked_h = h * pool_mask
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
