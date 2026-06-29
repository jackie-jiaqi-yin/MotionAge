"""GRU binary classifier with participant-level static covariate fusion."""

from __future__ import annotations

from motionage.models.common import PredictionMode
from motionage.models.recurrent_binary import MaskedRecurrentCovariateClassifier


class MaskedGRUCovariateClassifier(MaskedRecurrentCovariateClassifier):
    """GRU encoder with a static-covariate branch and binary fusion head."""

    def __init__(
        self,
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
        super().__init__(
            recurrent_type="gru",
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            hour_emb_dim=hour_emb_dim,
            day_emb_dim=day_emb_dim,
            num_numeric_features=num_numeric_features,
            categorical_cardinalities=categorical_cardinalities,
            categorical_embedding_dim=categorical_embedding_dim,
            covariate_hidden_dim=covariate_hidden_dim,
            covariate_output_dim=covariate_output_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            prediction_mode=prediction_mode,
            zero_init_residual=zero_init_residual,
        )
