from __future__ import annotations

import torch

import motionage.training as training


def test_build_model_inputs_moves_required_wearable_tensors_to_device() -> None:
    assert hasattr(training, "build_model_inputs")
    device = torch.device("cpu")
    batch = {
        "intensity": torch.tensor([[0.1, 0.2]]),
        "hour_idx": torch.tensor([[1, 2]]),
        "day_idx": torch.tensor([[3, 4]]),
        "mask": torch.tensor([[True, False]]),
    }

    inputs = training.build_model_inputs(batch, device)

    assert set(inputs) == {"intensity", "hour_idx", "day_idx", "mask"}
    for key, expected in batch.items():
        assert inputs[key].device == device
        assert torch.equal(inputs[key], expected)


def test_build_model_inputs_keeps_optional_static_covariates_when_present() -> None:
    assert hasattr(training, "build_model_inputs")
    device = torch.device("cpu")
    batch = {
        "intensity": torch.tensor([[0.1, 0.2]]),
        "hour_idx": torch.tensor([[1, 2]]),
        "day_idx": torch.tensor([[3, 4]]),
        "mask": torch.tensor([[True, True]]),
        "static_num": torch.tensor([[63.0, 1.0]]),
        "static_cat": torch.tensor([[2, 0]]),
        "static_num_missing": torch.tensor([[False, True]]),
    }

    inputs = training.build_model_inputs(batch, device)

    assert set(inputs) == set(batch)
    for key in ("static_num", "static_cat", "static_num_missing"):
        assert inputs[key].device == device
        assert torch.equal(inputs[key], batch[key])
