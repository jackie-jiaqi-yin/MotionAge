"""Bootstrap utilities for MotionAge evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

_ESTIMATE_KEYS = (
    "observed_auroc",
    "observed_auc_delta",
    "auc_delta",
    "bootstrap_mean_auroc",
    "bootstrap_mean_delta",
    "mean",
)
_BOOTSTRAP_MEAN_KEYS = ("bootstrap_mean_auroc", "bootstrap_mean_delta", "mean")
_CI_LOWER_KEYS = ("ci_lower", "ci95_lower")
_CI_UPPER_KEYS = ("ci_upper", "ci95_upper")
_VALID_RESAMPLE_KEYS = ("n_resamples_valid", "valid_resamples")


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


def public_bootstrap_interval_row(
    summary: Mapping[str, object],
    *,
    metric: str,
    comparison: str | None = None,
    resampling_unit: str | None = None,
) -> dict[str, str | float | int | bool]:
    """Return an allowlisted bootstrap interval row for public reports."""
    row: dict[str, str | float | int | bool] = {"metric": str(metric)}
    if comparison is not None:
        row["comparison"] = str(comparison)

    row["estimate"] = _first_numeric(summary, _ESTIMATE_KEYS, required=True)
    row["bootstrap_mean"] = _first_numeric(summary, _BOOTSTRAP_MEAN_KEYS, required=True)
    row["ci_lower"] = _first_numeric(summary, _CI_LOWER_KEYS, required=True)
    row["ci_upper"] = _first_numeric(summary, _CI_UPPER_KEYS, required=True)
    row["ci_level"] = _numeric(summary, "ci_level")
    row["n_resamples_requested"] = _integer(summary, "n_resamples_requested")
    row["valid_resamples"] = _integer_from_first(summary, _VALID_RESAMPLE_KEYS)

    unit = resampling_unit if resampling_unit is not None else summary.get("resampling_unit")
    if unit not in (None, ""):
        row["resampling_unit"] = str(unit)
    if "stratified" in summary:
        row["stratified"] = bool(summary["stratified"])
    if "valid_folds" in summary:
        row["valid_folds"] = _integer(summary, "valid_folds")
    if "p_value" in summary:
        row["p_value"] = _numeric(summary, "p_value")
    return row


def public_bootstrap_interval_table(
    summaries: Sequence[Mapping[str, object]],
    *,
    resampling_unit: str | None = None,
) -> list[dict[str, str | float | int | bool]]:
    """Return allowlisted bootstrap interval rows for public report tables."""
    rows: list[dict[str, str | float | int | bool]] = []
    for index, summary in enumerate(summaries):
        metric = summary.get("metric")
        if metric in (None, ""):
            raise ValueError(f"Bootstrap summary at index {index} must include metric.")
        comparison_value = summary.get("comparison")
        comparison = None if comparison_value in (None, "") else str(comparison_value)
        rows.append(
            public_bootstrap_interval_row(
                summary,
                metric=str(metric),
                comparison=comparison,
                resampling_unit=resampling_unit,
            )
        )
    return rows


def _safe_observed_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique_targets = np.unique(y_true)
    if unique_targets.size < 2:
        return 0.5
    if not set(unique_targets.tolist()).issubset({0, 1}):
        raise ValueError("bootstrap AUROC targets must be binary values 0/1.")
    return float(roc_auc_score(y_true, y_prob))


def _first_numeric(
    summary: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    required: bool,
) -> float:
    for key in keys:
        if key in summary:
            return _numeric(summary, key)
    if required:
        raise KeyError(f"Bootstrap summary missing required key from: {keys}")
    return float("nan")


def _integer_from_first(summary: Mapping[str, object], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in summary:
            return _integer(summary, key)
    raise KeyError(f"Bootstrap summary missing required key from: {keys}")


def _numeric(summary: Mapping[str, object], key: str) -> float:
    value = summary[key]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Bootstrap summary field {key!r} must be numeric.")
    return float(value)


def _integer(summary: Mapping[str, object], key: str) -> int:
    numeric = _numeric(summary, key)
    if not numeric.is_integer():
        raise ValueError(f"Bootstrap summary field {key!r} must be an integer.")
    return int(numeric)
