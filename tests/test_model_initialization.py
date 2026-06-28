from __future__ import annotations

from pathlib import Path

import pytest
import torch

from motionage.models import build_model
from motionage.models.initialization import (
    InitializationSummary,
    initialize_model_from_checkpoint,
    initialize_model_from_config,
)


def _recurrent_config(model_type: str, hidden_size: int = 8) -> dict:
    return {
        "model": {
            "type": model_type,
            "hidden_size": hidden_size,
            "num_layers": 1,
            "dropout": 0.0,
            "hour_emb_dim": 4,
            "day_emb_dim": 2,
        }
    }


def test_initialize_model_from_checkpoint_loads_matching_gru_tensors(tmp_path: Path) -> None:
    source = build_model(_recurrent_config("gru_binary"))
    checkpoint_path = tmp_path / "gru.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

    target = build_model(_recurrent_config("gru_binary"))
    summary = initialize_model_from_checkpoint(
        target,
        checkpoint_path,
        device=torch.device("cpu"),
        strict=False,
    )

    assert isinstance(summary, InitializationSummary)
    assert summary.applied is True
    assert summary.loaded_tensor_count == summary.source_tensor_count
    assert summary.skipped_missing_count == 0
    assert summary.skipped_shape_count == 0
    assert torch.allclose(target.hour_embedding.weight, source.hour_embedding.weight)
    assert torch.allclose(target.recurrent.weight_ih_l0, source.recurrent.weight_ih_l0)
    assert torch.allclose(target.classifier.weight, source.classifier.weight)


def test_initialize_model_from_checkpoint_partially_warm_starts_lstm_from_gru(
    tmp_path: Path,
) -> None:
    source = build_model(_recurrent_config("gru_binary"))
    checkpoint_path = tmp_path / "gru.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

    target = build_model(_recurrent_config("lstm_binary"))
    summary = initialize_model_from_checkpoint(
        target,
        checkpoint_path,
        device=torch.device("cpu"),
        strict=False,
    )

    assert summary.applied is True
    assert 0 < summary.loaded_tensor_count < summary.source_tensor_count
    assert summary.skipped_shape_count > 0
    assert torch.allclose(target.hour_embedding.weight, source.hour_embedding.weight)
    assert torch.allclose(target.classifier.weight, source.classifier.weight)


def test_initialize_model_from_checkpoint_normalizes_legacy_gru_key_prefix(
    tmp_path: Path,
) -> None:
    source = build_model(_recurrent_config("gru_binary"))
    legacy_state = {
        key.replace("recurrent.", "gru."): value
        for key, value in source.state_dict().items()
    }
    checkpoint_path = tmp_path / "legacy_gru.pt"
    torch.save({"model_state_dict": legacy_state}, checkpoint_path)

    target = build_model(_recurrent_config("gru_binary"))
    summary = initialize_model_from_checkpoint(
        target,
        checkpoint_path,
        device=torch.device("cpu"),
        strict=False,
    )

    assert summary.loaded_tensor_count == summary.source_tensor_count
    assert summary.skipped_missing_count == 0
    assert summary.skipped_shape_count == 0
    assert torch.allclose(target.recurrent.weight_ih_l0, source.recurrent.weight_ih_l0)


def test_initialize_model_from_config_noops_without_checkpoint() -> None:
    model = build_model(_recurrent_config("gru_binary"))

    summary = initialize_model_from_config(model, {"experiment": {}}, device=torch.device("cpu"))

    assert summary == InitializationSummary(applied=False)


def test_initialize_model_from_config_raises_for_missing_checkpoint(tmp_path: Path) -> None:
    model = build_model(_recurrent_config("gru_binary"))

    with pytest.raises(FileNotFoundError, match="init_checkpoint"):
        initialize_model_from_config(
            model,
            {"experiment": {"init_checkpoint": str(tmp_path / "missing.pt")}},
            device=torch.device("cpu"),
        )
