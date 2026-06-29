"""Training summary helpers for MotionAge reports."""

from __future__ import annotations

from typing import Any, Mapping

from motionage.training.selection import MetricDirection
from motionage.training.task import BINARY_CLASSIFICATION, REGRESSION, TaskType


DEFAULT_PUBLIC_MODEL_FAMILY_LABELS = {
    "gru": "GRU",
    "GRU": "GRU",
    "lstm": "LSTM",
    "LSTM": "LSTM",
    "transformer": "Transformer",
    "Transformer": "Transformer",
}


def build_training_summary(
    *,
    task_type: TaskType | str,
    selection_metric_name: str,
    selection_metric_direction: MetricDirection | str,
    best_val_metric: float,
    best_val_loss: float,
    best_epoch: int,
    epochs_trained: int,
    best_eval_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Build the common training summary returned by training workflows."""
    summary: dict[str, Any] = {
        "task_type": task_type,
        "selection_metric_name": selection_metric_name,
        "selection_metric_direction": selection_metric_direction,
        "best_val_metric": best_val_metric,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
    }
    summary.update(_task_metric_summary(task_type, best_eval_metrics))
    return summary


def build_fixed_epoch_training_summary(
    *,
    task_type: TaskType | str,
    selection_metric_name: str,
    selection_metric_direction: MetricDirection | str,
    best_val_metric: float,
    best_val_loss: float,
    best_epoch: int,
    epochs_trained: int,
    best_eval_metrics: Mapping[str, float],
    fixed_epochs: int,
    start_epoch: int,
    resumed_from_epoch: int,
) -> dict[str, Any]:
    """Build a summary for fixed-epoch refits that do not use validation."""
    summary = build_training_summary(
        task_type=task_type,
        selection_metric_name=selection_metric_name,
        selection_metric_direction=selection_metric_direction,
        best_val_metric=best_val_metric,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        best_eval_metrics=best_eval_metrics,
    )
    summary.update(
        {
            "fit_mode": "fixed_epochs_no_validation",
            "fixed_epochs": int(fixed_epochs),
            "start_epoch": int(start_epoch),
            "resumed_from_epoch": int(resumed_from_epoch),
        }
    )
    return summary


def build_public_training_report_row(
    summary: Mapping[str, Any],
    *,
    model_labels: Mapping[str, str] | None = None,
    model_family_labels: Mapping[str, str] | None = None,
    require_public_model_labels: bool = True,
) -> dict[str, Any]:
    """Build a public-safe aggregate training report row from a summary."""
    fit_mode = str(summary.get("fit_mode", "validation_selected"))
    public_row: dict[str, Any] = {
        "task_type": summary.get("task_type"),
        "fit_mode": fit_mode,
        "validation_used": fit_mode != "fixed_epochs_no_validation",
        "selection_metric_name": summary.get("selection_metric_name"),
        "selection_metric_direction": summary.get("selection_metric_direction"),
        "best_val_metric": summary.get("best_val_metric"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_trained": summary.get("epochs_trained"),
    }
    if "model_id" in summary:
        public_row["model"] = _public_training_label(
            str(summary["model_id"]),
            model_labels or {},
            label_type="model",
            required=require_public_model_labels,
        )
    if "model_family" in summary:
        family_labels = dict(DEFAULT_PUBLIC_MODEL_FAMILY_LABELS)
        if model_family_labels:
            family_labels.update(model_family_labels)
        public_row["model_family"] = _public_training_label(
            str(summary["model_family"]),
            family_labels,
            label_type="model family",
            required=require_public_model_labels,
        )
    for optional_key in ("fixed_epochs", "start_epoch", "resumed_from_epoch"):
        if optional_key in summary:
            public_row[optional_key] = summary[optional_key]
    for key, value in summary.items():
        if key.startswith("best_val_") and key not in public_row:
            public_row[key] = value
    return public_row


def _public_training_label(
    value: str,
    labels: Mapping[str, str],
    *,
    label_type: str,
    required: bool,
) -> str:
    label = labels.get(value)
    if label:
        return label
    if not required:
        return value
    raise ValueError(f"Missing public {label_type} labels for value: {value}")


def _task_metric_summary(
    task_type: TaskType | str,
    best_eval_metrics: Mapping[str, float],
) -> dict[str, float]:
    if task_type == REGRESSION:
        return {
            "best_val_mae": float(best_eval_metrics.get("mae", float("nan"))),
            "best_val_r2": float(best_eval_metrics.get("r2", float("nan"))),
        }
    if task_type == BINARY_CLASSIFICATION:
        return {
            "best_val_auroc": float(best_eval_metrics.get("auroc", float("nan"))),
            "best_val_auprc": float(best_eval_metrics.get("auprc", float("nan"))),
            "best_val_logloss": float(best_eval_metrics.get("logloss", float("nan"))),
            "best_val_brier": float(best_eval_metrics.get("brier", float("nan"))),
        }
    raise ValueError(f"Unsupported task_type '{task_type}'.")
