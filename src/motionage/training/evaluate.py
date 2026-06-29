"""Prediction and participant-level aggregation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from motionage.training.model_inputs import build_model_inputs
from motionage.training.task import BINARY_CLASSIFICATION, REGRESSION, TaskType


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a dataloader and return predictions with targets."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        pred = model(**build_model_inputs(batch, device))
        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(batch["y"].detach().cpu().numpy())

    if not all_preds:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(all_preds), np.concatenate(all_targets)


def aggregate_to_participant(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-window predictions back to participant-level rows."""
    preds_arr = np.asarray(preds).reshape(-1)
    targets_arr = np.asarray(targets).reshape(-1)
    if preds_arr.shape[0] != targets_arr.shape[0]:
        raise ValueError(
            "Predictions and targets must have the same length for participant aggregation "
            f"(got preds={preds_arr.shape[0]}, targets={targets_arr.shape[0]})."
        )

    aligned_meta = align_meta_to_predictions(meta, preds_arr.shape[0])
    df = pd.DataFrame({"pred": preds_arr, "target": targets_arr, "id": aligned_meta["id"].values})
    grouped = df.groupby("id").agg({"pred": "mean", "target": "first"}).reset_index()
    return grouped["pred"].values, grouped["target"].values


def align_meta_to_predictions(
    meta: pd.DataFrame,
    sample_count: int,
    *,
    allow_prefix: bool = False,
) -> pd.DataFrame:
    """Align per-window metadata to the number of model predictions."""
    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative, got {sample_count}.")

    meta_len = len(meta)
    if meta_len == sample_count:
        return meta
    if allow_prefix and meta_len > sample_count:
        return meta.iloc[:sample_count].reset_index(drop=True).copy()

    raise ValueError(
        "Window metadata length must match the number of predictions "
        f"(got meta={meta_len}, predictions={sample_count}, allow_prefix={allow_prefix})."
    )


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities with a numerically stable sigmoid."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    clipped = np.clip(logits_arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def prepare_validation_outputs(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    task_type: TaskType | str,
    meta: pd.DataFrame | None = None,
    allow_prefix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare predictions and targets for validation metric computation."""
    preds_arr = np.asarray(predictions).reshape(-1)
    targets_arr = np.asarray(targets).reshape(-1)

    if task_type == BINARY_CLASSIFICATION:
        preds_arr = logits_to_probabilities(preds_arr)
    elif task_type != REGRESSION:
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    if meta is not None and len(meta) > 0:
        aligned_meta = align_meta_to_predictions(meta, len(targets_arr), allow_prefix=allow_prefix)
        return aggregate_to_participant(preds_arr, targets_arr, aligned_meta)

    return preds_arr, targets_arr
