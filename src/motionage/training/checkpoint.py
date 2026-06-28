"""Checkpoint payload construction helpers."""

from __future__ import annotations

from typing import Any

import torch

from motionage.training.task import BINARY_CLASSIFICATION, REGRESSION


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    task_type: str,
    selection_metric_name: str,
    selection_metric_direction: str,
    best_val_metric: float,
    best_val_loss: float,
    best_eval_metrics: dict[str, float],
    best_epoch: int,
    patience_counter: int,
    training_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the serializable state payload used for model checkpoints."""
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "task_type": task_type,
        "selection_metric_name": str(selection_metric_name),
        "selection_metric_direction": str(selection_metric_direction),
        "best_val_metric": float(best_val_metric),
        "best_val_loss": float(best_val_loss),
        "best_eval_metrics": dict(best_eval_metrics),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "training_log": [dict(row) for row in training_log],
    }

    if task_type == REGRESSION:
        payload.update(
            {
                "val_mae": float(best_eval_metrics.get("mae", float("nan"))),
                "val_r2": float(best_eval_metrics.get("r2", float("nan"))),
            }
        )
    elif task_type == BINARY_CLASSIFICATION:
        payload.update(
            {
                "val_auroc": float(best_eval_metrics.get("auroc", float("nan"))),
                "val_auprc": float(best_eval_metrics.get("auprc", float("nan"))),
                "val_logloss": float(best_eval_metrics.get("logloss", float("nan"))),
                "val_brier": float(best_eval_metrics.get("brier", float("nan"))),
            }
        )
    return payload


def build_checkpoint_resume_state(
    checkpoint: dict[str, Any],
    *,
    task_type: str,
) -> dict[str, Any]:
    """Extract trainer resume metadata from a checkpoint payload."""
    state: dict[str, Any] = {
        "resumed_from_epoch": int(checkpoint.get("epoch", 0)),
        "best_val_metric": float(checkpoint.get("best_val_metric", float("nan"))),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "best_eval_metrics": dict(checkpoint.get("best_eval_metrics", {})),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "patience_counter": int(checkpoint.get("patience_counter", 0)),
        "training_log": [dict(row) for row in checkpoint.get("training_log", [])],
    }

    if task_type == REGRESSION:
        state.update(
            {
                "best_val_mae": float(checkpoint.get("val_mae", float("nan"))),
                "best_val_r2": float(checkpoint.get("val_r2", float("nan"))),
            }
        )
        return state

    if task_type == BINARY_CLASSIFICATION:
        state.update(
            {
                "best_val_auroc": float(checkpoint.get("val_auroc", float("nan"))),
                "best_val_auprc": float(checkpoint.get("val_auprc", float("nan"))),
                "best_val_logloss": float(checkpoint.get("val_logloss", float("nan"))),
                "best_val_brier": float(checkpoint.get("val_brier", float("nan"))),
            }
        )
        return state

    raise ValueError(f"Unsupported task_type '{task_type}'.")
