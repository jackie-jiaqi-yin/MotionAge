"""Training log row helpers for MotionAge experiments."""

from __future__ import annotations

import csv
import math
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


def build_public_training_log_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-epoch training logs without exposing the full local log."""
    if not rows:
        return {"logged_epoch_count": 0}

    ordered_rows = sorted(rows, key=lambda row: int(row["epoch"]))
    first_row = ordered_rows[0]
    final_row = ordered_rows[-1]

    summary: dict[str, Any] = {
        "logged_epoch_count": len(ordered_rows),
        "first_logged_epoch": int(first_row["epoch"]),
        "last_logged_epoch": int(final_row["epoch"]),
    }

    freeze_stages = _ordered_unique_text_values(row.get("freeze_stage") for row in ordered_rows)
    if freeze_stages:
        summary["freeze_stages"] = freeze_stages

    trainable_counts = [
        int(row["trainable_param_count"])
        for row in ordered_rows
        if row.get("trainable_param_count") is not None
    ]
    if trainable_counts:
        summary["trainable_param_count_min"] = min(trainable_counts)
        summary["trainable_param_count_max"] = max(trainable_counts)

    lr_mins = _numeric_values(row.get("lr_min") for row in ordered_rows)
    lr_maxes = _numeric_values(row.get("lr_max") for row in ordered_rows)
    if lr_mins:
        summary["lr_min"] = min(lr_mins)
    if lr_maxes:
        summary["lr_max"] = max(lr_maxes)

    for source_key, output_key in (
        ("train_loss", "final_train_loss"),
        ("val_loss", "final_val_loss"),
    ):
        if source_key in final_row:
            summary[output_key] = final_row[source_key]

    selection_metric_names = _ordered_unique_text_values(
        row.get("selection_metric_name") for row in ordered_rows
    )
    if selection_metric_names:
        summary["selection_metric_name"] = (
            selection_metric_names[0] if len(selection_metric_names) == 1 else "mixed"
        )

    selection_metric_values = _numeric_values(
        row.get("selection_metric_value") for row in ordered_rows
    )
    if selection_metric_values:
        summary["selection_metric_value_min"] = min(selection_metric_values)
        summary["selection_metric_value_max"] = max(selection_metric_values)

    return summary


def _ordered_unique_text_values(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value not in (None, "")))


def _numeric_values(values: Sequence[Any]) -> list[float]:
    numeric_values: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric_value = float(value)
        if math.isnan(numeric_value):
            continue
        numeric_values.append(numeric_value)
    return numeric_values
