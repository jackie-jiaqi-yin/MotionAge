"""Wear-aware preprocessing helpers for accelerometer activity data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def detect_nonwear_choi(
    intensity: Sequence[float] | np.ndarray,
    *,
    min_period_len: int = 90,
    spike_tolerance: int = 2,
    spike_stim_min: float = 0.0,
    spike_stim_max: float = 100.0,
    spike_surround_min: int = 30,
) -> np.ndarray:
    """Detect non-wear intervals using Choi-style zero-count rules."""
    if min_period_len <= 0:
        raise ValueError("min_period_len must be positive.")
    if spike_tolerance < 0:
        raise ValueError("spike_tolerance must be non-negative.")
    if spike_surround_min < 0:
        raise ValueError("spike_surround_min must be non-negative.")

    values = np.asarray(intensity, dtype=float).reshape(-1)
    n = values.size
    nonwear = np.zeros(n, dtype=bool)
    if n == 0:
        return nonwear

    is_zero = values == 0
    is_spike = (values > spike_stim_min) & (values <= spike_stim_max)
    is_high = values > spike_stim_max

    high_indices = np.where(is_high)[0]
    segments: list[tuple[int, int]] = []
    start = 0
    for high_idx in high_indices:
        if high_idx > start:
            segments.append((start, int(high_idx)))
        start = int(high_idx) + 1
    if start < n:
        segments.append((start, n))

    for seg_start, seg_end in segments:
        if seg_end - seg_start < min_period_len:
            continue
        seg_zero = is_zero[seg_start:seg_end]
        seg_spike_positions = np.where(is_spike[seg_start:seg_end])[0]
        if seg_spike_positions.size == 0:
            nonwear[seg_start:seg_end] = True
            continue
        if _valid_nonwear_segment(
            seg_zero,
            seg_spike_positions,
            spike_tolerance=spike_tolerance,
            spike_surround_min=spike_surround_min,
        ):
            nonwear[seg_start:seg_end] = True
            continue
        _mark_valid_subsegments(
            nonwear,
            seg_start=seg_start,
            seg_end=seg_end,
            seg_zero=seg_zero,
            spike_positions=seg_spike_positions,
            min_period_len=min_period_len,
            spike_tolerance=spike_tolerance,
            spike_surround_min=spike_surround_min,
        )

    return nonwear


def downsample_wear_epochs(
    frame: pd.DataFrame,
    *,
    interval: int = 5,
    tau: float = 0.2,
    id_column: str = "SEQN",
    minute_index_column: str = "PAXN",
    intensity_column: str = "PAXINTEN",
    wear_column: str = "wear_flag",
    day_column: str = "PAXDAY",
    hour_column: str = "PAXHOUR",
    minute_column: str = "PAXMINUT",
    interval_column: str = "interval_group",
    attention_column: str = "attention_flag",
    wear_time_column: str = "wear_time",
    intensity_mean_column: str = "intensity_mean",
) -> pd.DataFrame:
    """Aggregate minute-level activity into wear-aware fixed-length epochs."""
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if not (0.0 <= tau <= 1.0):
        raise ValueError("tau must be between 0 and 1 inclusive.")
    _require_columns(frame, [id_column, minute_index_column, intensity_column, wear_column])

    working = frame.copy()
    working[minute_index_column] = pd.to_numeric(working[minute_index_column], errors="coerce")
    working[intensity_column] = pd.to_numeric(working[intensity_column], errors="coerce").fillna(0.0)
    working[wear_column] = pd.to_numeric(working[wear_column], errors="coerce").fillna(0.0)
    min_index = working[minute_index_column].min()
    if pd.isna(min_index):
        raise ValueError("minute index column contains no valid values.")
    offset = 0 if int(min_index) == 0 else 1
    working[interval_column] = ((working[minute_index_column] - offset) // int(interval)).astype("int64")
    working["_wear_intensity_sum"] = working[intensity_column] * working[wear_column]

    aggregations: dict[str, str | list[str]] = {
        minute_index_column: "first",
        intensity_column: "sum",
        wear_column: ["mean", "sum"],
        "_wear_intensity_sum": "sum",
    }
    for optional_column in (day_column, hour_column, minute_column):
        if optional_column in working.columns:
            aggregations[optional_column] = "first"

    grouped = working.groupby([id_column, interval_column], sort=False).agg(aggregations).reset_index()
    grouped.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    rename_map = {
        f"{minute_index_column}_first": minute_index_column,
        f"{intensity_column}_sum": intensity_column,
        f"{wear_column}_mean": "_wear_proportion",
        f"{wear_column}_sum": wear_time_column,
        "_wear_intensity_sum_sum": "_wear_intensity_sum",
    }
    for optional_column in (day_column, hour_column, minute_column):
        rename_map[f"{optional_column}_first"] = optional_column
    grouped = grouped.rename(columns=rename_map)

    grouped[attention_column] = (grouped["_wear_proportion"] >= float(tau)).astype(int)
    grouped[intensity_mean_column] = np.where(
        grouped[wear_time_column] > 0,
        grouped["_wear_intensity_sum"] / grouped[wear_time_column],
        0.0,
    )
    return grouped.drop(columns=["_wear_proportion", "_wear_intensity_sum"])


def retained_window_count(
    attention: Sequence[float] | np.ndarray,
    *,
    seq_len: int,
    stride: int,
    coverage_cutoff: float,
) -> int:
    """Count sliding windows whose mean attention meets a coverage cutoff."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if not (0.0 <= coverage_cutoff <= 1.0):
        raise ValueError("coverage_cutoff must be between 0 and 1 inclusive.")
    values = np.asarray(attention, dtype=float).reshape(-1)
    if values.size < seq_len:
        return 0
    starts = np.arange(0, values.size - int(seq_len) + 1, int(stride))
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    sums = cumsum[starts + int(seq_len)] - cumsum[starts]
    return int(np.count_nonzero((sums / float(seq_len)) >= float(coverage_cutoff)))


def summarize_window_retention(
    frame: pd.DataFrame,
    *,
    seq_lens: Iterable[int],
    coverage_cutoffs: Iterable[float],
    stride_ratio: float = 1.0,
    id_column: str = "SEQN",
    attention_column: str = "attention_flag",
    target_column: str | None = None,
    time_column: str = "PAXN",
) -> pd.DataFrame:
    """Summarize retained windows, participants, and events across settings."""
    if stride_ratio <= 0:
        raise ValueError("stride_ratio must be positive.")
    required = [id_column, attention_column]
    if target_column is not None:
        required.append(target_column)
    _require_columns(frame, required)

    grouped = []
    for participant_id, group in frame.groupby(id_column, sort=False):
        if time_column in group.columns:
            group = group.sort_values(time_column)
        attention = group[attention_column].to_numpy(dtype=float)
        target = _participant_target(group, target_column) if target_column is not None else None
        grouped.append((participant_id, attention, target))

    rows: list[dict[str, int | float]] = []
    for seq_len in seq_lens:
        seq_len = int(seq_len)
        stride = max(1, int(seq_len * float(stride_ratio)))
        for cutoff in coverage_cutoffs:
            retained_windows = 0
            retained_participants = 0
            retained_events = 0
            for _, attention, target in grouped:
                count = retained_window_count(
                    attention,
                    seq_len=seq_len,
                    stride=stride,
                    coverage_cutoff=float(cutoff),
                )
                retained_windows += count
                if count > 0:
                    retained_participants += 1
                    if target == 1:
                        retained_events += 1
            rows.append(
                {
                    "seq_len": int(seq_len),
                    "coverage_cutoff": float(cutoff),
                    "stride": int(stride),
                    "retained_windows": int(retained_windows),
                    "retained_participants": int(retained_participants),
                    "retained_events": int(retained_events),
                }
            )
    return pd.DataFrame(rows)


def _valid_nonwear_segment(
    seg_zero: np.ndarray,
    spike_positions: np.ndarray,
    *,
    spike_tolerance: int,
    spike_surround_min: int,
) -> bool:
    if spike_positions.size > spike_tolerance:
        return False
    if spike_positions.size > 1 and np.any(np.diff(spike_positions) == 1):
        return False
    for spike_pos in spike_positions:
        left_start = max(0, int(spike_pos) - int(spike_surround_min))
        right_end = min(seg_zero.size, int(spike_pos) + 1 + int(spike_surround_min))
        if int(spike_pos) - left_start < spike_surround_min:
            return False
        if right_end - (int(spike_pos) + 1) < spike_surround_min:
            return False
        if not np.all(seg_zero[left_start:int(spike_pos)]):
            return False
        if not np.all(seg_zero[int(spike_pos) + 1:right_end]):
            return False
    return True


def _mark_valid_subsegments(
    nonwear: np.ndarray,
    *,
    seg_start: int,
    seg_end: int,
    seg_zero: np.ndarray,
    spike_positions: np.ndarray,
    min_period_len: int,
    spike_tolerance: int,
    spike_surround_min: int,
) -> None:
    seg_len = seg_end - seg_start
    spike_position_set = set(spike_positions.tolist())
    i = 0
    while i < seg_len:
        j = i
        local_spikes: list[int] = []
        while j < seg_len:
            if j in spike_position_set:
                if len(local_spikes) >= spike_tolerance:
                    break
                if local_spikes and j == local_spikes[-1] + 1:
                    break
                local_spikes.append(j)
            j += 1
        if j - i >= min_period_len and _valid_nonwear_segment(
            seg_zero[i:j],
            np.asarray(local_spikes, dtype=int) - i,
            spike_tolerance=spike_tolerance,
            spike_surround_min=spike_surround_min,
        ):
            nonwear[seg_start + i : seg_start + j] = True
            i = j
        else:
            i += 1


def _participant_target(group: pd.DataFrame, target_column: str | None) -> int | None:
    if target_column is None:
        return None
    values = pd.to_numeric(group[target_column], errors="coerce").dropna().unique()
    if values.size == 0:
        return None
    if values.size > 1:
        raise ValueError(f"Participant has multiple target values in {target_column}.")
    return int(values[0])


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
