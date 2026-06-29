"""Transformer binary classifier for MotionAge mortality prediction."""

from __future__ import annotations

import torch
from torch import nn

from motionage.models.common import validate_sequence_inputs


class MaskedTransformerClassifier(nn.Module):
    """Transformer encoder with temporal embeddings and masked mean pooling."""

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

        self.max_seq_len = max_seq_len
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
        self.classifier = nn.Linear(d_model, 1)

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
