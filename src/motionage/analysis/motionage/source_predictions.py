"""Source-model prediction aggregation for MotionAge analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def logits_to_probabilities(logits: np.ndarray | list[float] | pd.Series) -> np.ndarray:
    """Convert logits to probabilities with a numerically stable sigmoid."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(logits_arr, dtype=np.float64)

    positive = logits_arr >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits_arr[positive]))
    exp_logits = np.exp(logits_arr[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def build_participant_source_predictions(
    windows: pd.DataFrame,
    *,
    id_column: str = "SEQN",
    split_column: str = "split",
    target_column: str | None = None,
    logit_column: str | None = None,
    probability_column: str | None = None,
    include_participant_logit: bool = True,
    participant_probability_column: str = "participant_probability",
    participant_logit_column: str = "participant_logit",
    n_windows_column: str = "n_windows",
    logit_clip_eps: float = 1.0e-7,
) -> pd.DataFrame:
    """Aggregate window-level source-model predictions to participant-level risk rows."""
    if probability_column is None and logit_column is None:
        raise ValueError("Either probability_column or logit_column must be provided.")
    if not (0.0 < float(logit_clip_eps) < 0.5):
        raise ValueError("logit_clip_eps must be between 0 and 0.5.")

    required_columns = [id_column, split_column]
    if target_column is not None:
        required_columns.append(target_column)
    if probability_column is not None:
        required_columns.append(probability_column)
    elif logit_column is not None:
        required_columns.append(logit_column)
    _validate_required_columns(windows, required_columns)

    work_columns = [id_column, split_column]
    if target_column is not None:
        work_columns.append(target_column)
    work = windows[work_columns].copy()
    if probability_column is not None:
        probability_values = pd.to_numeric(windows[probability_column], errors="coerce").to_numpy(dtype=np.float64)
        _validate_probabilities(probability_values)
    else:
        logit_values = pd.to_numeric(windows[logit_column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(logit_values).all():
            raise ValueError(f"Window logit column {logit_column!r} must contain finite numeric values.")
        probability_values = logits_to_probabilities(logit_values)
    work["_window_probability"] = probability_values

    _validate_single_value_per_participant(work, id_column=id_column, value_column=split_column, label="split")
    if target_column is not None:
        _validate_single_value_per_participant(work, id_column=id_column, value_column=target_column, label="target")

    aggregations: dict[str, tuple[str, str]] = {
        split_column: (split_column, "first"),
        participant_probability_column: ("_window_probability", "mean"),
        n_windows_column: ("_window_probability", "size"),
    }
    if target_column is not None:
        aggregations[target_column] = (target_column, "first")

    grouped = work.groupby(id_column, as_index=False).agg(**aggregations)
    if target_column is not None:
        ordered_columns = [
            id_column,
            split_column,
            target_column,
            participant_probability_column,
            n_windows_column,
        ]
    else:
        ordered_columns = [id_column, split_column, participant_probability_column, n_windows_column]

    if include_participant_logit:
        clipped = np.clip(
            grouped[participant_probability_column].to_numpy(dtype=np.float64),
            float(logit_clip_eps),
            1.0 - float(logit_clip_eps),
        )
        grouped[participant_logit_column] = np.log(clipped / (1.0 - clipped))
        ordered_columns.insert(-1, participant_logit_column)

    return grouped[ordered_columns].copy()


def summarize_source_predictions(
    participants: pd.DataFrame,
    *,
    split_column: str = "split",
    target_column: str | None = None,
    participant_probability_column: str = "participant_probability",
    n_windows_column: str = "n_windows",
) -> list[dict[str, float | int | str]]:
    """Summarize participant-level source predictions into public aggregate rows."""
    required_columns = [split_column, participant_probability_column, n_windows_column]
    if target_column is not None:
        required_columns.append(target_column)
    _validate_required_columns(participants, required_columns)

    rows: list[dict[str, float | int | str]] = []
    for split_name, split_df in participants.groupby(split_column, sort=True):
        probabilities = pd.to_numeric(
            split_df[participant_probability_column],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        _validate_probabilities(probabilities)
        n_windows = pd.to_numeric(split_df[n_windows_column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(n_windows).all() or (n_windows < 0.0).any():
            raise ValueError(f"{n_windows_column!r} must contain finite non-negative values.")

        row: dict[str, float | int | str] = {
            split_column: str(split_name),
            "n": int(split_df.shape[0]),
        }
        if target_column is not None:
            targets = pd.to_numeric(split_df[target_column], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(targets).all() or not np.isin(targets, [0.0, 1.0]).all():
                raise ValueError(f"{target_column!r} must contain binary 0/1 values.")
            events = int(np.sum(targets == 1.0))
            row.update(
                {
                    "events": events,
                    "non_events": int(split_df.shape[0] - events),
                    "event_rate": _public_float(events / split_df.shape[0]),
                }
            )
        row.update(
            {
                "n_windows": int(np.sum(n_windows)),
                "mean_windows_per_participant": _public_float(np.mean(n_windows)),
                f"{participant_probability_column}_mean": _public_float(np.mean(probabilities)),
                f"{participant_probability_column}_std": _public_float(np.std(probabilities, ddof=0)),
                f"{participant_probability_column}_min": _public_float(np.min(probabilities)),
                f"{participant_probability_column}_max": _public_float(np.max(probabilities)),
            }
        )
        rows.append(row)
    return rows


def _validate_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required source prediction columns: {missing}.")


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("Window probability column must contain finite numeric values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Window probability column must contain values between 0 and 1.")


def _validate_single_value_per_participant(
    df: pd.DataFrame,
    *,
    id_column: str,
    value_column: str,
    label: str,
) -> None:
    unique_counts = df.groupby(id_column)[value_column].nunique(dropna=False)
    invalid_ids = unique_counts[unique_counts != 1]
    if invalid_ids.empty:
        return
    examples = [str(value) for value in invalid_ids.index[:5]]
    raise ValueError(f"Each participant must map to exactly one {label}; examples: {examples}.")


def _public_float(value: float | np.floating) -> float:
    return round(float(value), 12)
