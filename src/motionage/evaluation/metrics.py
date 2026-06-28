"""Task-aware evaluation metrics for MotionAge experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import somersd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from lifelines.utils import concordance_index as lifelines_concordance_index
except Exception:  # pragma: no cover - optional in lightweight environments.
    lifelines_concordance_index = None


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities with clipping for numerical stability."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    logits_arr = np.clip(logits_arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits_arr))


def binary_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Return threshold-free binary-classification metrics."""
    y_true_arr, y_prob_arr = _prepare_binary_inputs(y_true, y_prob)
    positive_rate = float(np.mean(y_true_arr))

    if np.unique(y_true_arr).size >= 2:
        auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
        auprc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        auroc = 0.5
        auprc = positive_rate

    return {
        "auroc": auroc,
        "auprc": auprc,
        "logloss": float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "positive_rate": positive_rate,
    }


def binary_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Return threshold-dependent binary-classification metrics."""
    y_true_arr, y_prob_arr = _prepare_binary_inputs(y_true, y_prob)
    y_pred = (y_prob_arr >= float(threshold)).astype(np.int64)

    if np.unique(y_true_arr).size >= 2:
        balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred))
    else:
        balanced_accuracy = 0.5

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def select_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
) -> tuple[float, float]:
    """Choose the threshold that maximizes a validation metric."""
    y_true_arr, y_prob_arr = _prepare_binary_inputs(y_true, y_prob)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob_arr)))

    best_threshold = 0.5
    best_score = float("-inf")
    best_distance = 0.0

    for threshold in candidates:
        score = float(
            binary_threshold_metrics(y_true_arr, y_prob_arr, float(threshold)).get(
                metric,
                float("-inf"),
            )
        )
        distance = abs(float(threshold) - 0.5)
        if score > best_score or (score == best_score and distance < best_distance):
            best_threshold = float(threshold)
            best_score = score
            best_distance = distance

    return best_threshold, best_score


def binary_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, float]]:
    """Return one metrics row per candidate threshold."""
    y_true_arr, y_prob_arr = _prepare_binary_inputs(y_true, y_prob)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob_arr)))
    rows = [binary_threshold_metrics(y_true_arr, y_prob_arr, float(threshold)) for threshold in candidates]
    rows.sort(key=lambda row: row["threshold"])
    return rows


def binary_precision_recall_curve_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, float]]:
    """Return precision-recall curve rows with a stable schema."""
    y_true_arr, y_prob_arr = _prepare_binary_inputs(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    rows: list[dict[str, float]] = []
    for idx, precision_value in enumerate(precision):
        rows.append(
            {
                "precision": float(precision_value),
                "recall": float(recall[idx]),
                "threshold": float(thresholds[idx]) if idx < len(thresholds) else float("nan"),
            }
        )
    return rows


def binary_roc_curve_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, float]]:
    """Return ROC curve rows with a stable schema."""
    y_true_arr, y_prob_arr = _prepare_binary_inputs(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)

    rows: list[dict[str, float]] = []
    for idx, fpr_value in enumerate(fpr):
        rows.append(
            {
                "fpr": float(fpr_value),
                "tpr": float(tpr[idx]),
                "threshold": float(thresholds[idx]),
            }
        )
    return rows


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_type: str = "regression",
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute metrics for a regression or binary-classification task."""
    if task_type == "binary_classification":
        metrics = binary_probability_metrics(y_true, y_pred)
        if threshold is not None:
            metrics.update(binary_threshold_metrics(y_true, y_pred, threshold))
        return metrics
    if task_type != "regression":
        raise ValueError(f"Unsupported task_type: {task_type}")

    y_true_arr, y_pred_arr = _prepare_regression_inputs(y_true, y_pred)
    return {
        "mae": _mae(y_true_arr, y_pred_arr),
        "rmse": _rmse(y_true_arr, y_pred_arr),
        "r2": _r2(y_true_arr, y_pred_arr),
        "c_index": _c_index(y_true_arr, y_pred_arr),
        "median_ae": _median_ae(y_true_arr, y_pred_arr),
    }


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def _c_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.all(y_true == y_true[0]):
        return 0.5

    if lifelines_concordance_index is not None:
        c_stat = float(lifelines_concordance_index(y_true, y_pred))
        return float(np.clip(c_stat, 0.0, 1.0))

    d_stat = float(somersd(y_true, y_pred).statistic)
    if not np.isfinite(d_stat):
        return 0.5
    return float(np.clip((d_stat + 1.0) / 2.0, 0.0, 1.0))


def _prepare_binary_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob).reshape(-1)
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError(
            "y_true and y_prob must have the same length, "
            f"got {len(y_true_arr)} and {len(y_prob_arr)}."
        )

    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    if not np.any(valid_mask):
        raise ValueError("At least one finite binary target/probability pair is required.")

    y_true_arr = y_true_arr[valid_mask].astype(np.int64)
    y_prob_arr = np.clip(y_prob_arr[valid_mask].astype(np.float64), 1.0e-7, 1.0 - 1.0e-7)
    return y_true_arr, y_prob_arr


def _prepare_regression_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length, "
            f"got {len(y_true_arr)} and {len(y_pred_arr)}."
        )

    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(valid_mask):
        raise ValueError("At least one finite regression target/prediction pair is required.")
    return y_true_arr[valid_mask], y_pred_arr[valid_mask]
