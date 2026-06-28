from __future__ import annotations

import copy

import pytest
import torch

from motionage.models import (
    MaskedGRUClassifier,
    MaskedGRUCovariateClassifier,
    MaskedLSTMClassifier,
    MaskedLSTMCovariateClassifier,
    MaskedTransformerClassifier,
    MaskedTransformerCovariateClassifier,
    build_model,
    list_available_model_types,
)


def _recurrent_config(model_type: str) -> dict:
    return {
        "model": {
            "type": model_type,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "hour_emb_dim": 4,
            "day_emb_dim": 2,
        }
    }


def _recurrent_covariate_config(model_type: str, prediction_mode: str = "late_fusion") -> dict:
    config = _recurrent_config(model_type)
    config["model"].update(
        {
            "num_numeric_features": 3,
            "categorical_cardinalities": [5, 4],
            "categorical_embedding_dim": 3,
            "covariate_hidden_dim": 8,
            "covariate_output_dim": 6,
            "fusion_hidden_dim": 8,
            "fusion_dropout": 0.0,
            "prediction_mode": prediction_mode,
            "zero_init_residual": True,
        }
    )
    return config


def _transformer_config(model_type: str = "transformer_binary") -> dict:
    return {
        "model": {
            "type": model_type,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "hour_emb_dim": 4,
            "day_emb_dim": 2,
            "intensity_proj_dim": 4,
            "max_seq_len": 16,
        }
    }


def _transformer_covariate_config(prediction_mode: str = "late_fusion") -> dict:
    config = _transformer_config(model_type="transformer_covariates_binary")
    config["model"].update(
        {
            "num_numeric_features": 3,
            "categorical_cardinalities": [5, 4],
            "categorical_embedding_dim": 3,
            "covariate_hidden_dim": 8,
            "covariate_output_dim": 6,
            "fusion_hidden_dim": 8,
            "fusion_dropout": 0.0,
            "prediction_mode": prediction_mode,
            "zero_init_residual": True,
        }
    )
    return config


def _sequence_inputs(batch_size: int = 3, seq_len: int = 10) -> dict[str, torch.Tensor]:
    mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    if seq_len >= 4:
        mask[:, 3::4] = 0.0
    return {
        "intensity": torch.rand(batch_size, seq_len),
        "hour_idx": torch.randint(low=0, high=24, size=(batch_size, seq_len)),
        "day_idx": torch.randint(low=0, high=7, size=(batch_size, seq_len)),
        "mask": mask,
    }


def _static_inputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "static_num": torch.rand(batch_size, 3),
        "static_cat": torch.tensor([[0, 1], [2, 3], [4, 0]], dtype=torch.long)[:batch_size],
        "static_num_missing": torch.zeros(batch_size, 3),
    }


def test_factory_lists_explicit_binary_model_types() -> None:
    assert list_available_model_types() == (
        "gru_binary",
        "gru_covariates_binary",
        "lstm_binary",
        "lstm_covariates_binary",
        "transformer_binary",
        "transformer_covariates_binary",
    )


@pytest.mark.parametrize(
    ("model_type", "expected_class"),
    [
        ("gru_binary", MaskedGRUClassifier),
        ("lstm_binary", MaskedLSTMClassifier),
        ("transformer_binary", MaskedTransformerClassifier),
    ],
)
def test_binary_model_factory_and_forward_shape(model_type: str, expected_class: type[torch.nn.Module]) -> None:
    config = _transformer_config(model_type) if model_type == "transformer_binary" else _recurrent_config(model_type)
    model = build_model(config)

    output = model(**_sequence_inputs())

    assert isinstance(model, expected_class)
    assert output.shape == (3,)


@pytest.mark.parametrize(
    ("model_type", "expected_class"),
    [
        ("gru_covariates_binary", MaskedGRUCovariateClassifier),
        ("lstm_covariates_binary", MaskedLSTMCovariateClassifier),
        ("transformer_covariates_binary", MaskedTransformerCovariateClassifier),
    ],
)
@pytest.mark.parametrize("prediction_mode", ["late_fusion", "residual"])
def test_covariate_model_factory_and_forward_shape(
    model_type: str,
    expected_class: type[torch.nn.Module],
    prediction_mode: str,
) -> None:
    if model_type == "transformer_covariates_binary":
        config = _transformer_covariate_config(prediction_mode)
    else:
        config = _recurrent_covariate_config(model_type, prediction_mode)
    model = build_model(config)

    output = model(**_sequence_inputs(), **_static_inputs())

    assert isinstance(model, expected_class)
    assert output.shape == (3,)


def test_lstm_binary_uses_classifier_head_not_age_regression_head() -> None:
    model = build_model(_recurrent_config("lstm_binary"))

    assert isinstance(model, MaskedLSTMClassifier)
    assert hasattr(model, "classifier")
    assert not hasattr(model, "head")


@pytest.mark.parametrize("model_type", ["gru_covariates_binary", "lstm_covariates_binary"])
def test_residual_recurrent_covariate_head_zero_init_starts_from_base_logit(model_type: str) -> None:
    model = build_model(_recurrent_covariate_config(model_type, prediction_mode="residual"))
    model.eval()
    sequence = _sequence_inputs()
    static = _static_inputs()

    with torch.no_grad():
        output = model(**sequence, **static)
        pooled = model.encode_wearable(**sequence)
        base_output = model.classifier(pooled).squeeze(-1)

    assert torch.allclose(output, base_output)


def test_transformer_does_not_treat_mask_as_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    model = build_model(_transformer_config())
    captured: dict[str, object] = {}
    original_forward = model.encoder.forward

    def spy_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = dict(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.encoder, "forward", spy_forward)

    output = model(**_sequence_inputs())

    assert output.shape == (3,)
    assert "src_key_padding_mask" not in captured["kwargs"]


def test_model_params_nested_config_is_supported() -> None:
    config = {
        "model": {
            "type": "lstm_binary",
            "params": copy.deepcopy(_recurrent_config("lstm_binary")["model"]),
        }
    }
    model = build_model(config)

    assert isinstance(model, MaskedLSTMClassifier)


def test_ambiguous_lstm_model_type_is_rejected() -> None:
    config = _recurrent_config("lstm")

    with pytest.raises(ValueError, match="Unknown model.type 'lstm'"):
        build_model(config)


def test_transformer_rejects_sequence_longer_than_max_seq_len() -> None:
    model = build_model(_transformer_config())
    inputs = _sequence_inputs(seq_len=17)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(**inputs)


def test_covariate_model_requires_static_features() -> None:
    config = _recurrent_covariate_config("lstm_covariates_binary")
    config["model"]["num_numeric_features"] = 0
    config["model"]["categorical_cardinalities"] = []

    with pytest.raises(ValueError, match="requires at least one static"):
        build_model(config)


def test_static_covariate_shape_validation() -> None:
    model = build_model(_recurrent_covariate_config("gru_covariates_binary"))
    static = _static_inputs()
    static["static_cat"] = torch.zeros(3, 1, dtype=torch.long)

    with pytest.raises(ValueError, match="static_cat has 1 columns"):
        model(**_sequence_inputs(), **static)
