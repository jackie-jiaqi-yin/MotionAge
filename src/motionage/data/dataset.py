"""Dataset definitions for sequence modeling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FitbitSequenceDataset(Dataset):
    """Dataset returning tensors for mask-aware activity sequence models."""

    def __init__(
        self,
        intensity: np.ndarray,
        hour_idx: np.ndarray,
        day_idx: np.ndarray,
        mask: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if len(intensity) != len(targets):
            raise ValueError("intensity and targets must have same sample count")
        if len(hour_idx) != len(targets):
            raise ValueError("hour_idx and targets must have same sample count")
        if len(day_idx) != len(targets):
            raise ValueError("day_idx and targets must have same sample count")
        if len(mask) != len(targets):
            raise ValueError("mask and targets must have same sample count")

        self.intensity = torch.as_tensor(intensity, dtype=torch.float32)
        self.hour_idx = torch.as_tensor(hour_idx, dtype=torch.long)
        self.day_idx = torch.as_tensor(day_idx, dtype=torch.long)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "intensity": self.intensity[idx],
            "hour_idx": self.hour_idx[idx],
            "day_idx": self.day_idx[idx],
            "mask": self.mask[idx],
            "y": self.targets[idx],
        }


class FitbitSequenceWithCovariatesDataset(FitbitSequenceDataset):
    """Sequence dataset variant with participant-level static covariates."""

    def __init__(
        self,
        intensity: np.ndarray,
        hour_idx: np.ndarray,
        day_idx: np.ndarray,
        mask: np.ndarray,
        targets: np.ndarray,
        static_num: np.ndarray,
        static_cat: np.ndarray,
        static_num_missing: np.ndarray,
    ) -> None:
        super().__init__(
            intensity=intensity,
            hour_idx=hour_idx,
            day_idx=day_idx,
            mask=mask,
            targets=targets,
        )

        sample_count = len(targets)
        if len(static_num) != sample_count:
            raise ValueError("static_num and targets must have same sample count")
        if len(static_cat) != sample_count:
            raise ValueError("static_cat and targets must have same sample count")
        if len(static_num_missing) != sample_count:
            raise ValueError("static_num_missing and targets must have same sample count")

        self.static_num = torch.as_tensor(static_num, dtype=torch.float32)
        self.static_cat = torch.as_tensor(static_cat, dtype=torch.long)
        self.static_num_missing = torch.as_tensor(static_num_missing, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["static_num"] = self.static_num[idx]
        item["static_cat"] = self.static_cat[idx]
        item["static_num_missing"] = self.static_num_missing[idx]
        return item
