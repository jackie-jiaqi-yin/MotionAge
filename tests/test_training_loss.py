from __future__ import annotations

import pytest
import torch

import motionage.training as training


def test_build_loss_function_uses_mse_for_regression() -> None:
    assert hasattr(training, "build_loss_function")

    loss = training.build_loss_function({"task": {"type": "regression"}}, torch.device("cpu"))

    assert isinstance(loss, torch.nn.MSELoss)
    pred = torch.tensor([1.0, 3.0])
    target = torch.tensor([2.0, 1.0])
    assert loss(pred, target).item() == pytest.approx(2.5)


def test_build_loss_function_uses_unweighted_bce_for_binary_task_without_pos_weight() -> None:
    assert hasattr(training, "build_loss_function")

    loss = training.build_loss_function({"task": {"type": "binary_classification"}}, torch.device("cpu"))

    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)
    assert loss.pos_weight is None


def test_build_loss_function_applies_binary_pos_weight_on_device() -> None:
    assert hasattr(training, "build_loss_function")

    loss = training.build_loss_function(
        {"task": {"type": "binary", "pos_weight": "2.5"}},
        torch.device("cpu"),
    )

    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)
    assert loss.pos_weight is not None
    assert loss.pos_weight.device == torch.device("cpu")
    assert loss.pos_weight.dtype == torch.float32
    assert loss.pos_weight.item() == pytest.approx(2.5)


def test_build_loss_function_propagates_unknown_task_type_errors() -> None:
    assert hasattr(training, "build_loss_function")

    with pytest.raises(ValueError, match="Unsupported task.type"):
        training.build_loss_function({"task": {"type": "multiclass"}}, torch.device("cpu"))
