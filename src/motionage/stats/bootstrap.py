"""Bootstrap utilities for MotionAge evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_binary_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_resamples: int,
    ci_level: float,
    random_seed: int,
) -> dict[str, float | int]:
    """Bootstrap AUROC on a fixed prediction table without refitting a model."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_prob).reshape(-1)
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError(
            "y_true and y_prob must have the same length for bootstrap AUROC "
            f"(got {y_true_arr.shape[0]} vs {y_prob_arr.shape[0]})."
        )
    if y_true_arr.size == 0:
        raise ValueError("bootstrap AUROC requires at least one prediction.")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}.")

    observed = _safe_observed_auroc(y_true_arr, y_prob_arr)
    rng = np.random.default_rng(int(random_seed))
    sample_count = int(y_true_arr.shape[0])

    resampled_scores: list[float] = []
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, sample_count, size=sample_count)
        sample_true = y_true_arr[indices]
        if np.unique(sample_true).size < 2:
            continue
        sample_prob = y_prob_arr[indices]
        resampled_scores.append(float(roc_auc_score(sample_true, sample_prob)))

    if not resampled_scores:
        lower = observed
        upper = observed
        mean_score = observed
    else:
        alpha = 1.0 - float(ci_level)
        lower = float(np.quantile(resampled_scores, alpha / 2.0))
        upper = float(np.quantile(resampled_scores, 1.0 - alpha / 2.0))
        mean_score = float(np.mean(resampled_scores))

    return {
        "observed_auroc": observed,
        "bootstrap_mean_auroc": mean_score,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": float(ci_level),
        "n_resamples_requested": int(n_resamples),
        "n_resamples_valid": int(len(resampled_scores)),
    }


def _safe_observed_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique_targets = np.unique(y_true)
    if unique_targets.size < 2:
        return 0.5
    if not set(unique_targets.tolist()).issubset({0, 1}):
        raise ValueError("bootstrap AUROC targets must be binary values 0/1.")
    return float(roc_auc_score(y_true, y_prob))
