"""LSTM binary classifier for MotionAge mortality prediction."""

from __future__ import annotations

from motionage.models.recurrent_binary import MaskedRecurrentClassifier


class MaskedLSTMClassifier(MaskedRecurrentClassifier):
    """LSTM encoder with masked temporal pooling and a single-logit classifier head."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        hour_emb_dim: int = 8,
        day_emb_dim: int = 4,
    ) -> None:
        super().__init__(
            recurrent_type="lstm",
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            hour_emb_dim=hour_emb_dim,
            day_emb_dim=day_emb_dim,
        )
