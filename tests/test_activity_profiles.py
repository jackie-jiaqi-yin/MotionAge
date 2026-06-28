from __future__ import annotations

import pandas as pd
import pytest

from motionage.analysis.activity_profiles import (
    compare_activity_profile_strata,
    summarize_activity_profiles,
)


def test_summarize_activity_profiles_aggregates_participant_then_group_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"SEQN": 1, "age_band": "45-64", "accel_stratum": "low accel", "hour_index": 0, "PAXHOUR": 8, "intensity_mean": 10.0},
            {"SEQN": 1, "age_band": "45-64", "accel_stratum": "low accel", "hour_index": 1, "PAXHOUR": 20, "intensity_mean": 20.0},
            {"SEQN": 1, "age_band": "45-64", "accel_stratum": "low accel", "hour_index": 2, "PAXHOUR": 2, "intensity_mean": 5.0},
            {"SEQN": 2, "age_band": "45-64", "accel_stratum": "low accel", "hour_index": 0, "PAXHOUR": 8, "intensity_mean": 14.0},
            {"SEQN": 2, "age_band": "45-64", "accel_stratum": "low accel", "hour_index": 1, "PAXHOUR": 20, "intensity_mean": 26.0},
            {"SEQN": 2, "age_band": "45-64", "accel_stratum": "low accel", "hour_index": 2, "PAXHOUR": 2, "intensity_mean": 8.0},
            {"SEQN": 3, "age_band": "45-64", "accel_stratum": "high accel", "hour_index": 0, "PAXHOUR": 8, "intensity_mean": 8.0},
            {"SEQN": 3, "age_band": "45-64", "accel_stratum": "high accel", "hour_index": 1, "PAXHOUR": 20, "intensity_mean": 12.0},
            {"SEQN": 3, "age_band": "45-64", "accel_stratum": "high accel", "hour_index": 2, "PAXHOUR": 2, "intensity_mean": 4.0},
        ]
    )

    summary = summarize_activity_profiles(frame)

    low = summary.loc[summary["accel_stratum"] == "low accel"].iloc[0]
    high = summary.loc[summary["accel_stratum"] == "high accel"].iloc[0]

    assert low["n_participants"] == 2
    assert low["mean_intensity"] == pytest.approx((35 / 3 + 48 / 3) / 2)
    assert low["weekly_amplitude"] == pytest.approx((15 + 18) / 2)
    assert low["active_hours"] == pytest.approx(3.0)
    assert low["daytime_mean"] == pytest.approx(((10 + 20) / 2 + (14 + 26) / 2) / 2)
    assert low["nighttime_mean"] == pytest.approx((5 + 8) / 2)
    assert low["day_night_ratio"] == pytest.approx(low["daytime_mean"] / low["nighttime_mean"])
    assert high["n_participants"] == 1
    assert high["mean_intensity"] == pytest.approx(8.0)


def test_summarize_activity_profiles_collapses_duplicate_hour_bins_first() -> None:
    frame = pd.DataFrame(
        [
            {"SEQN": 1, "age_band": "65+", "accel_stratum": "low accel", "hour_index": 0, "PAXHOUR": 8, "intensity_mean": 10.0},
            {"SEQN": 1, "age_band": "65+", "accel_stratum": "low accel", "hour_index": 0, "PAXHOUR": 8, "intensity_mean": 14.0},
            {"SEQN": 1, "age_band": "65+", "accel_stratum": "low accel", "hour_index": 1, "PAXHOUR": 2, "intensity_mean": 6.0},
        ]
    )

    summary = summarize_activity_profiles(frame)

    row = summary.iloc[0]
    assert row["mean_intensity"] == pytest.approx(9.0)
    assert row["weekly_amplitude"] == pytest.approx(6.0)


def test_compare_activity_profile_strata_reports_high_minus_low_deltas() -> None:
    summary = pd.DataFrame(
        [
            {
                "age_band": "45-64",
                "accel_stratum": "low accel",
                "n_participants": 10,
                "mean_intensity": 100.0,
                "weekly_amplitude": 40.0,
                "active_hours": 80.0,
                "daytime_mean": 120.0,
                "nighttime_mean": 60.0,
                "day_night_ratio": 2.0,
            },
            {
                "age_band": "45-64",
                "accel_stratum": "high accel",
                "n_participants": 8,
                "mean_intensity": 75.0,
                "weekly_amplitude": 30.0,
                "active_hours": 70.0,
                "daytime_mean": 90.0,
                "nighttime_mean": 45.0,
                "day_night_ratio": 2.0,
            },
        ]
    )

    comparison = compare_activity_profile_strata(summary)

    row = comparison.iloc[0]
    assert row["low_n"] == 10
    assert row["high_n"] == 8
    assert row["mean_intensity_delta_high_minus_low"] == pytest.approx(-25.0)
    assert row["mean_intensity_percent_delta"] == pytest.approx(-25.0)
    assert row["weekly_amplitude_delta_high_minus_low"] == pytest.approx(-10.0)
    assert row["active_hours_delta_high_minus_low"] == pytest.approx(-10.0)


def test_compare_activity_profile_strata_skips_incomplete_age_bands() -> None:
    summary = pd.DataFrame(
        [
            {
                "age_band": "18-29",
                "accel_stratum": "low accel",
                "n_participants": 5,
                "mean_intensity": 100.0,
                "weekly_amplitude": 10.0,
                "active_hours": 20.0,
                "daytime_mean": 110.0,
                "nighttime_mean": 55.0,
                "day_night_ratio": 2.0,
            }
        ]
    )

    comparison = compare_activity_profile_strata(summary)

    assert comparison.empty
