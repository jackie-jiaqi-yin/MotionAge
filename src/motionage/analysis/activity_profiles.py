"""Activity-profile summaries for MotionAge interpretability reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

PROFILE_METRICS: tuple[str, ...] = (
    "mean_intensity",
    "weekly_amplitude",
    "active_hours",
    "daytime_mean",
    "nighttime_mean",
    "day_night_ratio",
)


def summarize_activity_profiles(
    frame: pd.DataFrame,
    *,
    id_column: str = "SEQN",
    age_band_column: str = "age_band",
    stratum_column: str = "accel_stratum",
    hour_index_column: str = "hour_index",
    hour_column: str = "PAXHOUR",
    intensity_column: str = "intensity_mean",
    daytime_start_hour: int = 6,
    daytime_end_hour: int = 22,
) -> pd.DataFrame:
    """Summarize participant-level activity profiles by age band and acceleration stratum."""
    required = [id_column, age_band_column, stratum_column, hour_column, intensity_column]
    _require_columns(frame, required)

    group_columns = [id_column, age_band_column, stratum_column, hour_column]
    if hour_index_column in frame.columns:
        group_columns.insert(3, hour_index_column)

    working = frame.copy()
    working[hour_column] = pd.to_numeric(working[hour_column], errors="coerce")
    working[intensity_column] = pd.to_numeric(working[intensity_column], errors="coerce")
    working = working.dropna(subset=[hour_column, intensity_column]).copy()
    hourly = (
        working.groupby(group_columns, sort=False, dropna=False, observed=True)[intensity_column]
        .mean()
        .reset_index()
    )
    hourly["is_daytime"] = hourly[hour_column].between(
        int(daytime_start_hour),
        int(daytime_end_hour) - 1,
        inclusive="both",
    )

    participant_rows = []
    participant_group_columns = [id_column, age_band_column, stratum_column]
    for keys, group in hourly.groupby(participant_group_columns, sort=False, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(participant_group_columns, keys, strict=True))
        row.update(_summarize_participant_profile(group, intensity_column=intensity_column))
        participant_rows.append(row)

    participant_profiles = pd.DataFrame(participant_rows)
    if participant_profiles.empty:
        return _empty_activity_summary(age_band_column, stratum_column)

    summary = (
        participant_profiles.groupby([age_band_column, stratum_column], sort=False, dropna=False, observed=True)
        .agg(
            n_participants=(id_column, "nunique"),
            mean_intensity=("mean_intensity", "mean"),
            weekly_amplitude=("weekly_amplitude", "mean"),
            active_hours=("active_hours", "mean"),
            daytime_mean=("daytime_mean", "mean"),
            nighttime_mean=("nighttime_mean", "mean"),
        )
        .reset_index()
    )
    summary["day_night_ratio"] = summary.apply(
        lambda row: _safe_ratio(float(row["daytime_mean"]), float(row["nighttime_mean"])),
        axis=1,
    )
    return summary


def compare_activity_profile_strata(
    summary: pd.DataFrame,
    *,
    age_band_column: str = "age_band",
    stratum_column: str = "accel_stratum",
    low_stratum: str = "low accel",
    high_stratum: str = "high accel",
    metrics: Sequence[str] = PROFILE_METRICS,
) -> pd.DataFrame:
    """Compare high versus low acceleration strata within each age band."""
    required = [age_band_column, stratum_column, "n_participants", *metrics]
    _require_columns(summary, required)

    rows: list[dict[str, object]] = []
    for age_band in summary[age_band_column].drop_duplicates().tolist():
        band = summary[summary[age_band_column] == age_band]
        low = _single_stratum_row(band, stratum_column=stratum_column, stratum=low_stratum)
        high = _single_stratum_row(band, stratum_column=stratum_column, stratum=high_stratum)
        if low is None or high is None:
            continue

        row: dict[str, object] = {
            age_band_column: age_band,
            "low_n": int(low["n_participants"]),
            "high_n": int(high["n_participants"]),
        }
        for metric in metrics:
            low_value = float(low[metric])
            high_value = float(high[metric])
            row[f"{metric}_low"] = low_value
            row[f"{metric}_high"] = high_value
            row[f"{metric}_delta_high_minus_low"] = high_value - low_value
            row[f"{metric}_percent_delta"] = _percent_delta(high_value, low_value)
        rows.append(row)

    if not rows:
        columns = [age_band_column, "low_n", "high_n"]
        for metric in metrics:
            columns.extend(
                [
                    f"{metric}_low",
                    f"{metric}_high",
                    f"{metric}_delta_high_minus_low",
                    f"{metric}_percent_delta",
                ]
            )
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _summarize_participant_profile(group: pd.DataFrame, *, intensity_column: str) -> dict[str, float]:
    values = group[intensity_column].to_numpy(dtype=float)
    daytime = group.loc[group["is_daytime"], intensity_column].to_numpy(dtype=float)
    nighttime = group.loc[~group["is_daytime"], intensity_column].to_numpy(dtype=float)
    daytime_mean = _nanmean(daytime)
    nighttime_mean = _nanmean(nighttime)
    return {
        "mean_intensity": float(np.mean(values)),
        "weekly_amplitude": float(np.max(values) - np.min(values)),
        "active_hours": float(np.sum(values > 0.0)),
        "daytime_mean": daytime_mean,
        "nighttime_mean": nighttime_mean,
        "day_night_ratio": _safe_ratio(daytime_mean, nighttime_mean),
    }


def _single_stratum_row(band: pd.DataFrame, *, stratum_column: str, stratum: str) -> pd.Series | None:
    rows = band[band[stratum_column].astype(str) == stratum]
    if rows.empty:
        return None
    if len(rows) > 1:
        raise ValueError(f"Expected one row for stratum {stratum}, found {len(rows)}.")
    return rows.iloc[0]


def _percent_delta(high_value: float, low_value: float) -> float:
    if not np.isfinite(low_value) or abs(low_value) < 1.0e-12:
        return float("nan")
    return float((high_value - low_value) / low_value * 100.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1.0e-12:
        return float("nan")
    return float(numerator / denominator)


def _nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _empty_activity_summary(age_band_column: str, stratum_column: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[age_band_column, stratum_column, "n_participants", *PROFILE_METRICS])


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
