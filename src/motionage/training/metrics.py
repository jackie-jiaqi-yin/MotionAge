"""Training metric aggregation helpers."""

from __future__ import annotations

from motionage.training.task import BINARY_CLASSIFICATION, REGRESSION, TaskType


def build_train_epoch_metrics(
    *,
    total_loss: float,
    total_grad_norm: float,
    n_batches: int,
    task_type: TaskType | str,
    total_abs_error: float = 0.0,
    total_targets: int = 0,
) -> dict[str, float]:
    """Build train-epoch metrics from accumulated batch totals."""
    batch_count = max(int(n_batches), 1)
    metrics = {
        "loss": float(total_loss) / batch_count,
        "avg_grad_norm": float(total_grad_norm) / batch_count,
    }

    if task_type == REGRESSION:
        target_count = max(int(total_targets), 1)
        metrics["mae"] = float(total_abs_error) / target_count
        return metrics

    if task_type == BINARY_CLASSIFICATION:
        return metrics

    raise ValueError(f"Unsupported task_type '{task_type}'.")
