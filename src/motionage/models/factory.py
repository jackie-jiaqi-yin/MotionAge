"""Model registry and factory utilities."""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from motionage.models.gru_binary import MaskedGRUClassifier
from motionage.models.gru_covariate_binary import MaskedGRUCovariateClassifier
from motionage.models.lstm_binary import MaskedLSTMClassifier
from motionage.models.lstm_covariate_binary import MaskedLSTMCovariateClassifier
from motionage.models.transformer_binary import MaskedTransformerClassifier
from motionage.models.transformer_covariate_binary import MaskedTransformerCovariateClassifier

ModelBuilder = Callable[[dict[str, Any]], nn.Module]


def _extract_model_params(model_cfg: dict[str, Any]) -> dict[str, Any]:
    if "params" in model_cfg:
        params = model_cfg["params"]
        if not isinstance(params, dict):
            raise ValueError("model.params must be a dictionary when provided.")
        return params
    return {key: value for key, value in model_cfg.items() if key != "type"}


def _required_kwargs(model_cfg: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise KeyError(f"Missing model config key(s): {', '.join(missing)}")
    return {key: model_cfg[key] for key in required}


def _build_gru_binary(model_cfg: dict[str, Any]) -> nn.Module:
    return MaskedGRUClassifier(
        **_required_kwargs(model_cfg, ("hidden_size", "num_layers", "dropout", "hour_emb_dim", "day_emb_dim"))
    )


def _build_lstm_binary(model_cfg: dict[str, Any]) -> nn.Module:
    return MaskedLSTMClassifier(
        **_required_kwargs(model_cfg, ("hidden_size", "num_layers", "dropout", "hour_emb_dim", "day_emb_dim"))
    )


def _build_recurrent_covariates_binary(
    model_cfg: dict[str, Any],
    cls: type[MaskedGRUCovariateClassifier] | type[MaskedLSTMCovariateClassifier],
) -> nn.Module:
    required = (
        "hidden_size",
        "num_layers",
        "dropout",
        "hour_emb_dim",
        "day_emb_dim",
        "num_numeric_features",
        "categorical_cardinalities",
    )
    optional_defaults = {
        "categorical_embedding_dim": 4,
        "covariate_hidden_dim": 64,
        "covariate_output_dim": 32,
        "fusion_hidden_dim": 64,
        "fusion_dropout": 0.1,
        "prediction_mode": "late_fusion",
        "zero_init_residual": True,
    }
    kwargs = _required_kwargs(model_cfg, required)
    for key, default in optional_defaults.items():
        kwargs[key] = model_cfg.get(key, default)
    return cls(**kwargs)


def _build_gru_covariates_binary(model_cfg: dict[str, Any]) -> nn.Module:
    return _build_recurrent_covariates_binary(model_cfg, MaskedGRUCovariateClassifier)


def _build_lstm_covariates_binary(model_cfg: dict[str, Any]) -> nn.Module:
    return _build_recurrent_covariates_binary(model_cfg, MaskedLSTMCovariateClassifier)


def _build_transformer_binary(model_cfg: dict[str, Any]) -> nn.Module:
    return MaskedTransformerClassifier(
        **_required_kwargs(
            model_cfg,
            (
                "d_model",
                "nhead",
                "num_layers",
                "dim_feedforward",
                "dropout",
                "hour_emb_dim",
                "day_emb_dim",
                "intensity_proj_dim",
                "max_seq_len",
            ),
        )
    )


def _build_transformer_covariates_binary(model_cfg: dict[str, Any]) -> nn.Module:
    required = (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "hour_emb_dim",
        "day_emb_dim",
        "intensity_proj_dim",
        "max_seq_len",
        "num_numeric_features",
        "categorical_cardinalities",
    )
    optional_defaults = {
        "categorical_embedding_dim": 4,
        "covariate_hidden_dim": 64,
        "covariate_output_dim": 32,
        "fusion_hidden_dim": 64,
        "fusion_dropout": 0.1,
        "prediction_mode": "late_fusion",
        "zero_init_residual": True,
    }
    kwargs = _required_kwargs(model_cfg, required)
    for key, default in optional_defaults.items():
        kwargs[key] = model_cfg.get(key, default)
    return MaskedTransformerCovariateClassifier(**kwargs)


_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "gru_binary": _build_gru_binary,
    "gru_covariates_binary": _build_gru_covariates_binary,
    "lstm_binary": _build_lstm_binary,
    "lstm_covariates_binary": _build_lstm_covariates_binary,
    "transformer_binary": _build_transformer_binary,
    "transformer_covariates_binary": _build_transformer_covariates_binary,
}

_MODEL_FAMILY_CATALOG: tuple[dict[str, object], ...] = (
    {
        "model_type": "gru_binary",
        "family": "GRU",
        "public_label": "GRU",
        "task": "fixed_horizon_mortality_binary_classification",
        "sequence_encoder": "masked_recurrent",
        "uses_static_covariates": False,
        "covariate_prediction_modes": (),
    },
    {
        "model_type": "gru_covariates_binary",
        "family": "GRU",
        "public_label": "GRU + covariates",
        "task": "fixed_horizon_mortality_binary_classification",
        "sequence_encoder": "masked_recurrent",
        "uses_static_covariates": True,
        "covariate_prediction_modes": ("late_fusion", "residual"),
    },
    {
        "model_type": "lstm_binary",
        "family": "LSTM",
        "public_label": "LSTM",
        "task": "fixed_horizon_mortality_binary_classification",
        "sequence_encoder": "masked_recurrent",
        "uses_static_covariates": False,
        "covariate_prediction_modes": (),
    },
    {
        "model_type": "lstm_covariates_binary",
        "family": "LSTM",
        "public_label": "LSTM + covariates",
        "task": "fixed_horizon_mortality_binary_classification",
        "sequence_encoder": "masked_recurrent",
        "uses_static_covariates": True,
        "covariate_prediction_modes": ("late_fusion", "residual"),
    },
    {
        "model_type": "transformer_binary",
        "family": "Transformer",
        "public_label": "Transformer",
        "task": "fixed_horizon_mortality_binary_classification",
        "sequence_encoder": "masked_transformer",
        "uses_static_covariates": False,
        "covariate_prediction_modes": (),
    },
    {
        "model_type": "transformer_covariates_binary",
        "family": "Transformer",
        "public_label": "Transformer + covariates",
        "task": "fixed_horizon_mortality_binary_classification",
        "sequence_encoder": "masked_transformer",
        "uses_static_covariates": True,
        "covariate_prediction_modes": ("late_fusion", "residual"),
    },
)


def list_available_model_types() -> tuple[str, ...]:
    """Return sorted available model type keys."""
    return tuple(sorted(_MODEL_REGISTRY.keys()))


def public_model_family_catalog() -> tuple[dict[str, object], ...]:
    """Return public-safe model family rows for publication-facing summaries."""
    return tuple(dict(row) for row in _MODEL_FAMILY_CATALOG)


def public_model_architecture_summary(config: dict[str, Any]) -> dict[str, object]:
    """Return allowlisted architecture metadata for a model config."""
    model_type = resolve_model_type(config)
    model_cfg = config["model"]
    params = _extract_model_params(model_cfg)
    catalog_row = _model_catalog_row(model_type)
    model = build_model(config)

    row: dict[str, object] = {
        "model_type": model_type,
        "family": catalog_row["family"],
        "public_label": catalog_row["public_label"],
        "task": catalog_row["task"],
        "sequence_encoder": catalog_row["sequence_encoder"],
        "uses_static_covariates": catalog_row["uses_static_covariates"],
        "prediction_mode": str(params.get("prediction_mode", "")),
        "num_numeric_features": int(params.get("num_numeric_features", 0)),
        "categorical_feature_count": len(params.get("categorical_cardinalities", [])),
        "num_layers": int(params["num_layers"]),
    }
    if "hidden_size" in params:
        row["hidden_size"] = int(params["hidden_size"])
    if "d_model" in params:
        row["d_model"] = int(params["d_model"])
    if "nhead" in params:
        row["nhead"] = int(params["nhead"])
    if "max_seq_len" in params:
        row["max_seq_len"] = int(params["max_seq_len"])

    row["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    row["trainable_parameter_count"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return row


def resolve_model_type(config: dict[str, Any]) -> str:
    """Resolve and validate `model.type` from a config dictionary."""
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Config must include a 'model' dictionary.")

    model_type = str(model_cfg.get("type", "")).lower()
    if not model_type:
        available = ", ".join(list_available_model_types())
        raise ValueError(f"Config must set model.type explicitly. Available: {available}")
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(list_available_model_types())
        raise ValueError(f"Unknown model.type '{model_type}'. Available: {available}")
    return model_type


def build_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate a registered model from a resolved config dictionary."""
    model_cfg = config["model"]
    model_type = resolve_model_type(config)
    params = _extract_model_params(model_cfg)
    return _MODEL_REGISTRY[model_type](params)


def _model_catalog_row(model_type: str) -> dict[str, object]:
    for row in _MODEL_FAMILY_CATALOG:
        if row["model_type"] == model_type:
            return dict(row)
    raise KeyError(f"Model type {model_type!r} is missing from the public model catalog.")
