from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import motionage.training as training


class SumIntensityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saw_static_covariates = False

    def forward(
        self,
        *,
        intensity: torch.Tensor,
        hour_idx: torch.Tensor,
        day_idx: torch.Tensor,
        mask: torch.Tensor,
        static_num: torch.Tensor | None = None,
        static_cat: torch.Tensor | None = None,
        static_num_missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hour_idx, day_idx, mask
        self.saw_static_covariates = (
            static_num is not None and static_cat is not None and static_num_missing is not None
        )
        return intensity.sum(dim=1)


def test_predict_collects_model_outputs_and_targets_from_limited_batches() -> None:
    assert hasattr(training, "predict")
    model = SumIntensityModel()
    loader = DataLoader(
        [
            {
                "intensity": torch.tensor([1.0, 2.0]),
                "hour_idx": torch.tensor([0, 1]),
                "day_idx": torch.tensor([2, 3]),
                "mask": torch.tensor([1.0, 1.0]),
                "static_num": torch.tensor([0.5]),
                "static_cat": torch.tensor([1]),
                "static_num_missing": torch.tensor([0.0]),
                "y": torch.tensor(0.0),
            },
            {
                "intensity": torch.tensor([3.0, 4.0]),
                "hour_idx": torch.tensor([4, 5]),
                "day_idx": torch.tensor([6, 0]),
                "mask": torch.tensor([1.0, 0.0]),
                "static_num": torch.tensor([1.5]),
                "static_cat": torch.tensor([2]),
                "static_num_missing": torch.tensor([1.0]),
                "y": torch.tensor(1.0),
            },
            {
                "intensity": torch.tensor([9.0, 9.0]),
                "hour_idx": torch.tensor([1, 1]),
                "day_idx": torch.tensor([1, 1]),
                "mask": torch.tensor([1.0, 1.0]),
                "y": torch.tensor(9.0),
            },
        ],
        batch_size=2,
    )

    preds, targets = training.predict(model, loader, torch.device("cpu"), limit_batches=1)

    np.testing.assert_allclose(preds, np.array([3.0, 7.0], dtype=np.float32))
    np.testing.assert_allclose(targets, np.array([0.0, 1.0], dtype=np.float32))
    assert model.saw_static_covariates is True


def test_aggregate_to_participant_averages_windows_and_keeps_first_target() -> None:
    assert hasattr(training, "aggregate_to_participant")
    preds, targets = training.aggregate_to_participant(
        preds=np.array([0.2, 0.4, 0.9], dtype=np.float32),
        targets=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        meta=pd.DataFrame({"id": [101, 101, 202]}),
    )

    np.testing.assert_allclose(preds, np.array([0.3, 0.9]))
    np.testing.assert_allclose(targets, np.array([0.0, 1.0]))


def test_align_meta_to_predictions_requires_matching_length_unless_prefix_allowed() -> None:
    assert hasattr(training, "align_meta_to_predictions")
    meta = pd.DataFrame({"id": [10, 11, 12], "start_idx": [0, 0, 0]})

    aligned = training.align_meta_to_predictions(meta, 2, allow_prefix=True)

    assert aligned["id"].tolist() == [10, 11]
    with pytest.raises(ValueError, match="metadata length"):
        training.align_meta_to_predictions(meta, 2)
    with pytest.raises(ValueError, match="non-negative"):
        training.align_meta_to_predictions(meta, -1)


def test_logits_to_probabilities_applies_sigmoid_to_known_logits() -> None:
    assert hasattr(training, "logits_to_probabilities")

    probabilities = training.logits_to_probabilities(np.array([0.0, np.log(3.0), -np.log(3.0)]))

    np.testing.assert_allclose(probabilities, np.array([0.5, 0.75, 0.25]), rtol=1.0e-7)


def test_logits_to_probabilities_clips_extreme_logits_to_finite_probabilities() -> None:
    assert hasattr(training, "logits_to_probabilities")

    probabilities = training.logits_to_probabilities(np.array([-1000.0, 1000.0]))

    assert probabilities.dtype == np.float64
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] > 0.0
    assert np.all(probabilities <= 1.0)
    assert np.all(probabilities >= 0.0)


def test_prepare_validation_outputs_converts_binary_logits_before_aggregation() -> None:
    assert hasattr(training, "prepare_validation_outputs")

    prepared_preds, prepared_targets = training.prepare_validation_outputs(
        predictions=np.array([0.0, np.log(3.0), -np.log(3.0)]),
        targets=np.array([0.0, 0.0, 1.0]),
        task_type=training.BINARY_CLASSIFICATION,
        meta=pd.DataFrame({"id": [101, 101, 202]}),
    )

    np.testing.assert_allclose(prepared_preds, np.array([0.625, 0.25]))
    np.testing.assert_allclose(prepared_targets, np.array([0.0, 1.0]))


def test_prepare_validation_outputs_aggregates_regression_predictions_without_sigmoid() -> None:
    assert hasattr(training, "prepare_validation_outputs")

    prepared_preds, prepared_targets = training.prepare_validation_outputs(
        predictions=np.array([10.0, 14.0, 20.0]),
        targets=np.array([9.0, 9.0, 18.0]),
        task_type=training.REGRESSION,
        meta=pd.DataFrame({"id": [101, 101, 202]}),
    )

    np.testing.assert_allclose(prepared_preds, np.array([12.0, 20.0]))
    np.testing.assert_allclose(prepared_targets, np.array([9.0, 18.0]))


def test_prepare_validation_outputs_returns_flat_arrays_without_metadata() -> None:
    assert hasattr(training, "prepare_validation_outputs")

    prepared_preds, prepared_targets = training.prepare_validation_outputs(
        predictions=np.array([[0.0], [np.log(3.0)]]),
        targets=np.array([[0.0], [1.0]]),
        task_type=training.BINARY_CLASSIFICATION,
    )

    np.testing.assert_allclose(prepared_preds, np.array([0.5, 0.75]))
    np.testing.assert_allclose(prepared_targets, np.array([0.0, 1.0]))


def test_prepare_validation_outputs_rejects_unknown_task_type() -> None:
    assert hasattr(training, "prepare_validation_outputs")

    with pytest.raises(ValueError, match="Unsupported task_type"):
        training.prepare_validation_outputs(
            predictions=np.array([1.0]),
            targets=np.array([1.0]),
            task_type="unsupported",
        )
