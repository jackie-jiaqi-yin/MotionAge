from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from motionage.preprocessing.wear import (
    detect_nonwear_choi,
    downsample_wear_epochs,
    retained_window_count,
    summarize_window_retention,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reports_docs_describe_public_wear_sensitivity_rates() -> None:
    reports_doc = (REPO_ROOT / "docs" / "reports" / "README.md").read_text(encoding="utf-8")

    assert "participant_retention_rate" in reports_doc
    assert "event_retention_rate" in reports_doc
    assert "participant identifiers" in reports_doc


def test_detect_nonwear_choi_marks_long_zero_run() -> None:
    intensity = np.array([150, 0, 0, 0, 0, 0, 150], dtype=float)

    nonwear = detect_nonwear_choi(intensity, min_period_len=5)

    assert nonwear.tolist() == [False, True, True, True, True, True, False]


def test_detect_nonwear_choi_allows_isolated_spike_with_zero_surround() -> None:
    intensity = np.array([0, 0, 0, 50, 0, 0, 0], dtype=float)

    nonwear = detect_nonwear_choi(
        intensity,
        min_period_len=7,
        spike_tolerance=1,
        spike_stim_max=100,
        spike_surround_min=3,
    )

    assert nonwear.all()


def test_detect_nonwear_choi_keeps_high_activity_boundary_as_wear() -> None:
    intensity = np.array([0, 0, 0, 150, 0, 0, 0], dtype=float)

    nonwear = detect_nonwear_choi(intensity, min_period_len=3, spike_stim_max=100)

    assert nonwear.tolist() == [True, True, True, False, True, True, True]


def test_downsample_wear_epochs_uses_tau_on_wear_proportion_and_wear_mean_intensity() -> None:
    frame = pd.DataFrame(
        {
            "SEQN": [1, 1, 1, 1, 1],
            "PAXN": [1, 2, 3, 4, 5],
            "PAXDAY": [1, 1, 1, 1, 1],
            "PAXHOUR": [0, 0, 0, 0, 0],
            "PAXMINUT": [0, 1, 2, 3, 4],
            "PAXINTEN": [10, 20, 30, 40, 50],
            "wear_flag": [1, 0, 0, 0, 0],
        }
    )

    tau02 = downsample_wear_epochs(frame, interval=5, tau=0.2)
    tau03 = downsample_wear_epochs(frame, interval=5, tau=0.3)

    assert tau02.loc[0, "attention_flag"] == 1
    assert tau03.loc[0, "attention_flag"] == 0
    assert tau02.loc[0, "wear_time"] == 1
    assert tau02.loc[0, "PAXINTEN"] == 150
    assert tau02.loc[0, "intensity_mean"] == 10


def test_tau_point_one_and_point_two_are_equivalent_for_one_of_five_wear_minutes() -> None:
    frame = pd.DataFrame(
        {
            "SEQN": [1, 1, 1, 1, 1],
            "PAXN": [0, 1, 2, 3, 4],
            "PAXINTEN": [1, 2, 3, 4, 5],
            "wear_flag": [1, 0, 0, 0, 0],
        }
    )

    tau01 = downsample_wear_epochs(frame, interval=5, tau=0.1)
    tau02 = downsample_wear_epochs(frame, interval=5, tau=0.2)

    assert tau01["attention_flag"].tolist() == tau02["attention_flag"].tolist() == [1]


def test_retained_window_count_counts_windows_meeting_coverage_cutoff() -> None:
    attention = np.array([1, 1, 0, 1, 1], dtype=float)

    assert retained_window_count(attention, seq_len=3, stride=1, coverage_cutoff=2 / 3) == 3
    assert retained_window_count(attention, seq_len=7, stride=1, coverage_cutoff=0.3) == 0


def test_summarize_window_retention_counts_windows_participants_and_events() -> None:
    frame = pd.DataFrame(
        {
            "SEQN": [1, 1, 1, 1, 2, 2, 2, 2],
            "PAXN": [1, 2, 3, 4, 1, 2, 3, 4],
            "attention_flag": [1, 1, 0, 0, 1, 0, 0, 0],
            "mortstat": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )

    summary = summarize_window_retention(
        frame,
        seq_lens=[2, 4],
        coverage_cutoffs=[0.5],
        stride_ratio=1.0,
        target_column="mortstat",
    )

    row_len2 = summary.loc[summary["seq_len"] == 2].iloc[0]
    row_len4 = summary.loc[summary["seq_len"] == 4].iloc[0]

    assert row_len2["stride"] == 2
    assert row_len2["eligible_windows"] == 4
    assert row_len2["retained_windows"] == 2
    assert row_len2["window_retention_rate"] == 0.5
    assert row_len2["retained_participants"] == 2
    assert row_len2["retained_events"] == 1
    assert row_len2["eligible_participants"] == 2
    assert row_len2["eligible_events"] == 1
    assert row_len2["participant_retention_rate"] == 1.0
    assert row_len2["event_retention_rate"] == 1.0
    assert row_len4["eligible_windows"] == 2
    assert row_len4["retained_windows"] == 1
    assert row_len4["window_retention_rate"] == 0.5
    assert row_len4["retained_participants"] == 1
    assert row_len4["retained_events"] == 1
    assert row_len4["eligible_participants"] == 2
    assert row_len4["eligible_events"] == 1
    assert row_len4["participant_retention_rate"] == 0.5
    assert row_len4["event_retention_rate"] == 1.0
