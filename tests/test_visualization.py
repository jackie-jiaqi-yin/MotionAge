from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from motionage.visualization.plots import plot_intensity_timeseries_by_hour


def _make_hourly_plot_frame() -> pd.DataFrame:
    rows = []
    ages = [8, 14, 24, 36, 52, 70]
    for idx, age in enumerate(ages, start=1):
        for day in (1, 2):
            for hour in (8, 20):
                rows.append(
                    {
                        "SEQN": idx,
                        "PAXDAY": day,
                        "PAXHOUR": hour,
                        "intensity_mean": idx * 10 + hour,
                        "RIDAGEYR": age,
                    }
                )
    return pd.DataFrame(rows)


def test_plot_intensity_timeseries_by_hour_uses_custom_age_groups() -> None:
    frame = _make_hourly_plot_frame()
    age_bins = [6, 12, 18, 30, 45, 65, float("inf")]
    age_labels = ["6-11", "12-17", "18-29", "30-44", "45-64", "65+"]

    ax = plot_intensity_timeseries_by_hour(
        data=frame,
        age_bins=age_bins,
        age_labels=age_labels,
        show_std=False,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == age_labels
    assert len(ax.lines) >= len(age_labels)
    assert len(ax.collections) == 0


def test_plot_intensity_timeseries_by_hour_validates_custom_age_groups() -> None:
    frame = _make_hourly_plot_frame()

    with pytest.raises(ValueError, match="age_labels"):
        plot_intensity_timeseries_by_hour(
            data=frame,
            age_bins=[6, 12, 18],
            age_labels=["6-11"],
        )


def test_plot_intensity_timeseries_by_hour_supports_publication_axis_layout() -> None:
    frame = _make_hourly_plot_frame()
    age_bins = [6, 12, 18, 30, 45, 65, float("inf")]
    age_labels = ["6-11", "12-17", "18-29", "30-44", "45-64", "65+"]

    baseline_ax = plot_intensity_timeseries_by_hour(
        data=frame,
        age_bins=age_bins,
        age_labels=age_labels,
        show_std=False,
        week_start_day=2,
        title=None,
        show_stats=False,
        legend_title="AgeGroup",
        tick_interval_hours=12,
    )
    baseline_top = baseline_ax.get_ylim()[1]

    ax = plot_intensity_timeseries_by_hour(
        data=frame,
        age_bins=age_bins,
        age_labels=age_labels,
        show_std=False,
        figsize=(6.75, 3.0),
        week_start_day=2,
        title=None,
        show_stats=False,
        legend_title="AgeGroup",
        legend_ncol=6,
        legend_loc="upper center",
        legend_bbox_to_anchor=(0.5, 0.98),
        legend_frameon=False,
        y_headroom=0.16,
        axis_label_fontsize=9,
        tick_label_fontsize=7,
        legend_fontsize=7,
        legend_title_fontsize=8,
        line_width=1.6,
        tick_interval_hours=12,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "AgeGroup"
    assert legend._ncols == 6
    assert legend.get_frame_on() is False
    assert tuple(round(v, 2) for v in ax.figure.get_size_inches()) == (6.75, 3.0)
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "Time of Week"
    assert ax.get_ylim()[1] > baseline_top
    assert ax.xaxis.label.get_fontsize() == 9
    assert ax.yaxis.label.get_fontsize() == 9
    assert ax.get_xticklabels()[0].get_fontsize() == 7
    assert legend.get_texts()[0].get_fontsize() == 7
    assert legend.get_title().get_fontsize() == 8
    assert ax.lines[0].get_linewidth() == 1.6
    assert list(ax.lines[0].get_xdata()) == [8, 20, 152, 164]
    assert [tick.get_text() for tick in ax.get_xticklabels()[:4]] == [
        "Mon 00:00",
        "Mon 12:00",
        "Tue 00:00",
        "Tue 12:00",
    ]
    assert len(ax.texts) == 0
