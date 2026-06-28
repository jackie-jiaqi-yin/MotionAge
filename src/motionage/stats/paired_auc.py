"""Paired AUROC comparison and bootstrap utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


PAIRED_COLUMNS = ("SEQN", "fold", "target", "RIDAGEYR", "score_left", "score_right")


def safe_auc(y_true: Iterable[int] | np.ndarray | pd.Series, y_score: Iterable[float] | np.ndarray | pd.Series) -> float:
    """Compute AUROC using the rank definition, returning NaN for single-class targets."""
    y_arr = np.asarray(list(y_true) if not isinstance(y_true, (np.ndarray, pd.Series)) else y_true, dtype=int)
    score_arr = np.asarray(list(y_score) if not isinstance(y_score, (np.ndarray, pd.Series)) else y_score, dtype=float)
    _validate_equal_length(y_arr, score_arr)

    unique = np.unique(y_arr)
    if unique.size < 2:
        return float("nan")
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError("AUROC targets must be binary values 0/1.")

    pos = score_arr[y_arr == 1]
    neg = score_arr[y_arr == 0]
    comparison = (pos[:, None] > neg[None, :]).astype(float)
    comparison += 0.5 * (pos[:, None] == neg[None, :])
    return float(comparison.mean())


def make_paired_score_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    id_column: str = "SEQN",
    target_column: str = "target",
    age_column: str = "RIDAGEYR",
    fold_column: str = "fold",
    score_column: str = "score",
    min_age: int | float | None = None,
) -> pd.DataFrame:
    """Pair two participant score tables on shared IDs and validate target alignment."""
    left_required = [id_column, target_column, age_column, fold_column, score_column]
    right_required = [id_column, target_column, age_column, fold_column, score_column]
    _validate_columns(left, left_required, label="left")
    _validate_columns(right, right_required, label="right")

    paired = left.merge(
        right[[id_column, target_column, age_column, fold_column, score_column]],
        on=id_column,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    target_mismatch = paired[f"{target_column}_left"].astype(int) != paired[f"{target_column}_right"].astype(int)
    if target_mismatch.any():
        raise ValueError(f"Found {int(target_mismatch.sum())} target mismatches after pairing.")
    if min_age is not None:
        paired = paired[paired[f"{age_column}_left"] >= float(min_age)].copy()

    result = pd.DataFrame(
        {
            "SEQN": paired[id_column],
            "fold": paired[f"{fold_column}_left"].astype(int),
            "target": paired[f"{target_column}_left"].astype(int),
            "RIDAGEYR": paired[f"{age_column}_left"].astype(float),
            "score_left": paired[f"{score_column}_left"].astype(float),
            "score_right": paired[f"{score_column}_right"].astype(float),
        }
    )
    return result.dropna(subset=list(PAIRED_COLUMNS)).reset_index(drop=True)


def paired_auc_delta(paired: pd.DataFrame) -> dict[str, float | int]:
    """Compute observed paired AUROC difference from a paired score frame."""
    _validate_paired_frame(paired)
    y = paired["target"].to_numpy(dtype=int)
    left = paired["score_left"].to_numpy(dtype=float)
    right = paired["score_right"].to_numpy(dtype=float)
    left_auc = safe_auc(y, left)
    right_auc = safe_auc(y, right)
    return {
        "n": int(len(paired)),
        "events": int((y == 1).sum()),
        "non_events": int((y == 0).sum()),
        "auc_left": float(left_auc),
        "auc_right": float(right_auc),
        "auc_delta": float(left_auc - right_auc) if np.isfinite(left_auc) and np.isfinite(right_auc) else float("nan"),
    }


def paired_bootstrap_auc_delta(
    paired: pd.DataFrame,
    *,
    n_resamples: int,
    random_seed: int,
    stratified: bool,
) -> dict[str, float | int]:
    """Bootstrap the paired AUROC delta over participants."""
    _validate_paired_frame(paired)
    _validate_resamples(n_resamples)
    y = paired["target"].to_numpy(dtype=int)
    left = paired["score_left"].to_numpy(dtype=float)
    right = paired["score_right"].to_numpy(dtype=float)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {
            **_empty_bootstrap_summary(),
            "n_resamples_requested": int(n_resamples),
            "ci_level": 0.95,
            "stratified": bool(stratified),
        }

    rng = np.random.default_rng(random_seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        if stratified:
            sample_idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=pos_idx.size, replace=True),
                    rng.choice(neg_idx, size=neg_idx.size, replace=True),
                ]
            )
        else:
            sample_idx = rng.integers(0, y.size, size=y.size)
            if np.unique(y[sample_idx]).size < 2:
                continue
        delta = safe_auc(y[sample_idx], left[sample_idx]) - safe_auc(y[sample_idx], right[sample_idx])
        if np.isfinite(delta):
            deltas.append(float(delta))

    return {
        **_summarize_bootstrap(np.asarray(deltas, dtype=float)),
        "n_resamples_requested": int(n_resamples),
        "ci_level": 0.95,
        "stratified": bool(stratified),
    }


def fold_structured_paired_bootstrap_auc_delta(
    paired: pd.DataFrame,
    *,
    n_resamples: int,
    random_seed: int,
) -> dict[str, float | int]:
    """Bootstrap paired AUROC deltas within each fold, then average fold deltas."""
    _validate_paired_frame(paired)
    _validate_resamples(n_resamples)
    rng = np.random.default_rng(random_seed)

    fold_payload: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    left_aucs: list[float] = []
    right_aucs: list[float] = []
    for _, fold_df in paired.groupby("fold", sort=True):
        y = fold_df["target"].to_numpy(dtype=int)
        left = fold_df["score_left"].to_numpy(dtype=float)
        right = fold_df["score_right"].to_numpy(dtype=float)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue
        fold_payload.append((y, left, right, pos_idx, neg_idx))
        left_aucs.append(safe_auc(y, left))
        right_aucs.append(safe_auc(y, right))

    if not fold_payload:
        return {
            **_empty_bootstrap_summary(),
            "observed_auc_left": float("nan"),
            "observed_auc_right": float("nan"),
            "observed_auc_delta": float("nan"),
            "n_resamples_requested": int(n_resamples),
            "ci_level": 0.95,
            "resampling_unit": "fold_stratified",
            "valid_folds": 0,
        }

    deltas = np.empty(n_resamples, dtype=float)
    for sample_number in range(n_resamples):
        fold_deltas: list[float] = []
        for y, left, right, pos_idx, neg_idx in fold_payload:
            sample_idx = np.concatenate(
                [
                    rng.choice(pos_idx, size=pos_idx.size, replace=True),
                    rng.choice(neg_idx, size=neg_idx.size, replace=True),
                ]
            )
            fold_deltas.append(safe_auc(y[sample_idx], left[sample_idx]) - safe_auc(y[sample_idx], right[sample_idx]))
        deltas[sample_number] = float(np.mean(fold_deltas))

    summary = _summarize_bootstrap(deltas)
    observed_left = float(np.mean(left_aucs))
    observed_right = float(np.mean(right_aucs))
    return {
        "observed_auc_left": observed_left,
        "observed_auc_right": observed_right,
        "observed_auc_delta": observed_left - observed_right,
        "bootstrap_mean_delta": summary["mean"],
        "bootstrap_sd_delta": summary["sd"],
        "ci95_lower": summary["ci95_lower"],
        "ci95_upper": summary["ci95_upper"],
        "ci95_width": summary["ci95_width"],
        "p_value": summary["p_value"],
        "valid_resamples": summary["valid_resamples"],
        "n_resamples_requested": int(n_resamples),
        "ci_level": 0.95,
        "resampling_unit": "fold_stratified",
        "valid_folds": int(len(fold_payload)),
    }


def _summarize_bootstrap(samples: np.ndarray) -> dict[str, float | int]:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return _empty_bootstrap_summary()
    lower, upper = np.quantile(samples, [0.025, 0.975])
    p_value = 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0)))
    return {
        "mean": float(samples.mean()),
        "sd": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "ci95_width": float(upper - lower),
        "p_value": float(min(1.0, p_value)),
        "valid_resamples": int(samples.size),
    }


def _empty_bootstrap_summary() -> dict[str, float | int]:
    return {
        "mean": float("nan"),
        "sd": float("nan"),
        "ci95_lower": float("nan"),
        "ci95_upper": float("nan"),
        "ci95_width": float("nan"),
        "p_value": float("nan"),
        "valid_resamples": 0,
    }


def _validate_equal_length(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Inputs must have the same length, got {left.shape[0]} and {right.shape[0]}.")


def _validate_columns(frame: pd.DataFrame, required_columns: list[str], *, label: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} score table missing required columns: {missing}")


def _validate_paired_frame(paired: pd.DataFrame) -> None:
    _validate_columns(paired, list(PAIRED_COLUMNS), label="paired")


def _validate_resamples(n_resamples: int) -> None:
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}.")
