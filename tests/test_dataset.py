from __future__ import annotations

import numpy as np
import pytest
import torch

from motionage.data.dataset import FitbitSequenceDataset, FitbitSequenceWithCovariatesDataset


def test_sequence_dataset_returns_tensor_item() -> None:
    dataset = FitbitSequenceDataset(
        intensity=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        hour_idx=np.array([[0, 1, 2]], dtype=np.int64),
        day_idx=np.array([[1, 1, 1]], dtype=np.int64),
        mask=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        targets=np.array([1.0], dtype=np.float32),
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["intensity"].shape == (3,)
    assert item["hour_idx"].dtype == torch.long
    assert item["mask"].tolist() == [1.0, 1.0, 0.0]
    assert item["y"].item() == pytest.approx(1.0)


def test_covariate_dataset_adds_static_features() -> None:
    dataset = FitbitSequenceWithCovariatesDataset(
        intensity=np.array([[0.1, 0.2]], dtype=np.float32),
        hour_idx=np.array([[0, 1]], dtype=np.int64),
        day_idx=np.array([[1, 1]], dtype=np.int64),
        mask=np.array([[1.0, 1.0]], dtype=np.float32),
        targets=np.array([0.0], dtype=np.float32),
        static_num=np.array([[65.0, 22.5]], dtype=np.float32),
        static_cat=np.array([[1, 3]], dtype=np.int64),
        static_num_missing=np.array([[0.0, 1.0]], dtype=np.float32),
    )

    item = dataset[0]

    assert item["static_num"].tolist() == [65.0, 22.5]
    assert item["static_cat"].tolist() == [1, 3]
    assert item["static_num_missing"].tolist() == [0.0, 1.0]


def test_dataset_rejects_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError, match="hour_idx and targets"):
        FitbitSequenceDataset(
            intensity=np.zeros((2, 3), dtype=np.float32),
            hour_idx=np.zeros((1, 3), dtype=np.int64),
            day_idx=np.zeros((2, 3), dtype=np.int64),
            mask=np.ones((2, 3), dtype=np.float32),
            targets=np.zeros(2, dtype=np.float32),
        )
