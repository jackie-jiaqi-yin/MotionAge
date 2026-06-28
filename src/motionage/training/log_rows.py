"""Training log row helpers for MotionAge experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from motionage.training.task import BINARY_CLASSIFICATION, REGRESSION, TaskType


def nan_validation_metrics(
    *,
    task_type: TaskType | str,
    selection_metric_name: str,
) -> dict[str, float]:
    """Return NaN validation metrics for fixed-epoch refits without validation."""
    if task_type == REGRESSION:
        metrics = {"mae": float("nan"), "r2": float("nan")}
    elif task_type == BINARY_CLASSIFICATION:
        metrics = {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
        }
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    metrics[selection_metric_name] = float("nan")
    return metrics


def build_training_log_row(
    *,
    epoch: int,
    freeze_stage_label: str,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    val_loss: float,
    lr_min: float,
    lr_max: float,
    trainable_param_count: int,
    task_type: TaskType | str,
    selection_metric_name: str,
) -> dict[str, Any]:
    """Build the per-epoch training log row used by training reports."""
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "avg_grad_norm": train_metrics["avg_grad_norm"],
        "val_loss": val_loss,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "freeze_stage": freeze_stage_label,
        "trainable_param_count": trainable_param_count,
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": float(val_metrics[selection_metric_name]),
    }

    if task_type == REGRESSION:
        row.update(
            {
                "train_mae": train_metrics["mae"],
                "val_mae": float(val_metrics["mae"]),
                "val_r2": float(val_metrics["r2"]),
            }
        )
        return row

    if task_type == BINARY_CLASSIFICATION:
        row.update(
            {
                "val_auroc": float(val_metrics["auroc"]),
                "val_auprc": float(val_metrics["auprc"]),
                "val_logloss": float(val_metrics["logloss"]),
                "val_brier": float(val_metrics["brier"]),
            }
        )
        return row

    raise ValueError(f"Unsupported task_type '{task_type}'.")


def training_log_fieldnames(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return CSV fieldnames from the first log row."""
    if not rows:
        return ()
    return tuple(rows[0].keys())


def write_training_log_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> bool:
    """Write a non-empty training log to CSV and return whether a file was written."""
    if not rows:
        return False

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=training_log_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    return True
