"""Window creation utilities for activity time-series modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from motionage.preprocessing.constants import ActivityColumns


def create_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
    stride: int = 1,
    id_column: str = ActivityColumns.ID,
    time_columns: list[str] | None = None,
    target_mode: str = "participant",
    attention_column: str | None = ActivityColumns.ATTENTION_FLAG,
    min_attention_ratio: float = 0.0,
) -> dict[str, Any]:
    """Create sliding windows from a participant-level time-series dataframe."""
    _validate_inputs(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        seq_len=seq_len,
        stride=stride,
        id_column=id_column,
        target_mode=target_mode,
        attention_column=attention_column,
        min_attention_ratio=min_attention_ratio,
    )

    sort_cols = _resolve_time_columns(df=df, id_column=id_column, time_columns=time_columns)
    df_sorted = df.sort_values(sort_cols).copy()

    x_windows: list[np.ndarray] = []
    y_windows: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    for participant_id, group in df_sorted.groupby(id_column, sort=False):
        group = group.reset_index(drop=True)
        n_rows = len(group)
        if n_rows < seq_len:
            continue

        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_column].to_numpy(dtype=np.float32)
        attention = (
            group[attention_column].to_numpy(dtype=np.float32)
            if attention_column is not None
            else None
        )

        participant_target = None
        if target_mode == "participant":
            participant_target = _resolve_participant_target(
                group=group,
                target_column=target_column,
                participant_id=participant_id,
            )
            if participant_target is None:
                continue

        for start_idx in range(0, n_rows - seq_len + 1, stride):
            end_idx = start_idx + seq_len

            if attention is not None:
                attention_ratio = float(attention[start_idx:end_idx].mean())
                if attention_ratio < min_attention_ratio:
                    continue

            x_windows.append(features[start_idx:end_idx])
            if target_mode == "participant":
                y_windows.append(float(participant_target))
            else:
                y_windows.append(float(targets[end_idx - 1]))

            meta_rows.append(
                {
                    "id": participant_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

    n_features = len(feature_columns)
    if len(x_windows) == 0:
        x = np.empty((0, seq_len, n_features), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        meta = pd.DataFrame(columns=["id", "start_idx", "end_idx"])
    else:
        x = np.stack(x_windows).astype(np.float32)
        y = np.asarray(y_windows, dtype=np.float32)
        meta = pd.DataFrame(meta_rows)

    return {"X": x, "y": y, "meta": meta}


def create_windows_for_ids(
    df: pd.DataFrame,
    participant_ids: np.ndarray | list[Any],
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
    stride: int = 1,
    id_column: str = ActivityColumns.ID,
    time_columns: list[str] | None = None,
    target_mode: str = "participant",
    attention_column: str | None = ActivityColumns.ATTENTION_FLAG,
    min_attention_ratio: float = 0.0,
) -> dict[str, Any]:
    """Create windows from a participant subset."""
    id_set = set(np.asarray(participant_ids).tolist())
    subset = df[df[id_column].isin(id_set)].copy()
    return create_windows(
        df=subset,
        feature_columns=feature_columns,
        target_column=target_column,
        seq_len=seq_len,
        stride=stride,
        id_column=id_column,
        time_columns=time_columns,
        target_mode=target_mode,
        attention_column=attention_column,
        min_attention_ratio=min_attention_ratio,
    )


def _resolve_time_columns(
    df: pd.DataFrame,
    id_column: str,
    time_columns: list[str] | None,
) -> list[str]:
    if time_columns is not None:
        required = [id_column, *time_columns]
        _require_columns(df=df, columns=required)
        return required

    if ActivityColumns.MINUTE_INDEX in df.columns:
        return [id_column, ActivityColumns.MINUTE_INDEX]

    fallback = [ActivityColumns.DAY, ActivityColumns.HOUR, ActivityColumns.MINUTE]
    _require_columns(df=df, columns=[id_column, *fallback])
    return [id_column, *fallback]


def _resolve_participant_target(
    group: pd.DataFrame,
    target_column: str,
    participant_id: Any,
) -> float | None:
    values = group[target_column].dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(
            f"Participant {participant_id} has multiple values in '{target_column}'. "
            "Use target_mode='last_timestep' or fix target alignment."
        )
    return float(values[0])


def _validate_inputs(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
    stride: int,
    id_column: str,
    target_mode: str,
    attention_column: str | None,
    min_attention_ratio: float,
) -> None:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}.")
    if len(feature_columns) == 0:
        raise ValueError("feature_columns must not be empty.")
    if target_mode not in {"participant", "last_timestep"}:
        raise ValueError(
            f"target_mode must be 'participant' or 'last_timestep', got {target_mode}."
        )
    if not (0.0 <= min_attention_ratio <= 1.0):
        raise ValueError(
            "min_attention_ratio must be between 0 and 1 inclusive, "
            f"got {min_attention_ratio}."
        )

    required = [id_column, target_column, *feature_columns]
    if attention_column is not None:
        required.append(attention_column)
    _require_columns(df=df, columns=required)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
