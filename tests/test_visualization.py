from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from motionage.visualization.plots import (
    build_public_lower_triangle_ci_heatmap_frame,
    build_public_motionage_mapping_frame,
    build_public_metric_interval_frame,
    plot_intensity_timeseries_by_hour,
    plot_motionage_mapping_diagnostics,
    plot_metric_interval_forest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reports_docs_describe_public_metric_interval_frame() -> None:
    reports_doc = (REPO_ROOT / "docs" / "reports" / "README.md").read_text(encoding="utf-8")

    assert "build_public_metric_interval_frame" in reports_doc
    assert "resample draws" in reports_doc


def test_reports_docs_describe_motionage_mapping_diagnostics_boundary() -> None:
    reports_doc = (REPO_ROOT / "docs" / "reports" / "README.md").read_text(encoding="utf-8")

    assert "build_public_motionage_mapping_frame" in reports_doc
    assert "age-bin mapping diagnostics" in reports_doc


def test_reports_docs_describe_lower_triangle_ci_heatmap_frame() -> None:
    reports_doc = (REPO_ROOT / "docs" / "reports" / "README.md").read_text(encoding="utf-8")

    assert "build_public_lower_triangle_ci_heatmap_frame" in reports_doc
    assert "lower-triangle paired confidence-interval heatmaps" in reports_doc


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


def test_plot_metric_interval_forest_uses_aggregate_confidence_intervals() -> None:
    frame = pd.DataFrame(
        [
            {"model_label": "Age", "auroc": 0.781, "ci_lower": 0.752, "ci_upper": 0.808},
            {"model_label": "PhenoAge", "auroc": 0.804, "ci_lower": 0.779, "ci_upper": 0.829},
            {"model_label": "MotionAge-FRC", "auroc": 0.836, "ci_lower": 0.812, "ci_upper": 0.858},
        ]
    )

    ax = plot_metric_interval_forest(
        frame,
        label_col="model_label",
        point_col="auroc",
        lower_col="ci_lower",
        upper_col="ci_upper",
        xlabel="AUROC",
        title=None,
        reference_value=0.8,
        marker_color="#1f77b4",
    )

    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "Age",
        "PhenoAge",
        "MotionAge-FRC",
    ]
    assert ax.get_xlabel() == "AUROC"
    assert ax.get_title() == ""
    assert len(ax.collections) == 1
    assert len(ax.lines) >= 1
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    assert ax.collections[0].get_offsets().shape[0] == 3


def test_build_public_metric_interval_frame_keeps_only_aggregate_plot_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_label": "Age",
                "estimate": 0.781,
                "ci_lower": 0.752,
                "ci_upper": 0.808,
                "metrics_path": "internal-run/metrics.csv",
                "resample_draws": [0.75, 0.80],
            },
            {
                "model_label": "MotionAge-FRC",
                "estimate": 0.836,
                "ci_lower": 0.812,
                "ci_upper": 0.858,
                "metrics_path": "internal-run/metrics.csv",
                "resample_draws": [0.81, 0.86],
            },
        ]
    )

    public_frame = build_public_metric_interval_frame(
        frame,
        label_col="model_label",
        point_col="estimate",
        lower_col="ci_lower",
        upper_col="ci_upper",
        metric_label="AUROC",
    )

    assert public_frame.to_dict("records") == [
        {
            "label": "Age",
            "estimate": 0.781,
            "ci_lower": 0.752,
            "ci_upper": 0.808,
            "metric": "AUROC",
        },
        {
            "label": "MotionAge-FRC",
            "estimate": 0.836,
            "ci_lower": 0.812,
            "ci_upper": 0.858,
            "metric": "AUROC",
        },
    ]
    assert "metrics_path" not in public_frame.columns
    assert "resample_draws" not in public_frame.columns


def test_build_public_lower_triangle_ci_heatmap_frame_keeps_only_aggregate_cells() -> None:
    paired = pd.DataFrame(
        [
            {
                "left_label": "MotionAge-FRC",
                "right_label": "PhenoAge",
                "observed_auc_delta": 0.018131,
                "ci95_lower": 0.004671,
                "ci95_upper": 0.031983,
                "source_path": "internal-run/pairs.csv",
                "resample_draws": [0.01, 0.02],
            },
            {
                "left_label": "Age",
                "right_label": "MotionAge-FRC",
                "observed_auc_delta": -0.052155,
                "ci95_lower": -0.066401,
                "ci95_upper": -0.038438,
                "source_path": "internal-run/pairs.csv",
                "resample_draws": [-0.06, -0.04],
            },
        ]
    )

    public_frame = build_public_lower_triangle_ci_heatmap_frame(
        paired,
        labels=["Age", "PhenoAge", "MotionAge-FRC"],
    )

    assert public_frame.to_dict("records") == [
        {
            "row_label": "MotionAge-FRC",
            "column_label": "Age",
            "delta": 0.052155,
            "ci_lower": 0.038438,
            "ci_upper": 0.066401,
            "significant": True,
            "annotation": "0.0522 (0.0384, 0.0664)*",
        },
        {
            "row_label": "MotionAge-FRC",
            "column_label": "PhenoAge",
            "delta": 0.018131,
            "ci_lower": 0.004671,
            "ci_upper": 0.031983,
            "significant": True,
            "annotation": "0.0181 (0.0047, 0.0320)*",
        },
    ]
    assert "source_path" not in public_frame.columns
    assert "resample_draws" not in public_frame.columns
    assert not ((public_frame["row_label"] == "Age") & (public_frame["column_label"] == "MotionAge-FRC")).any()


def test_build_public_motionage_mapping_frame_keeps_only_aggregate_plot_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "sex_value": 1,
                "age_bin": 65,
                "representative_probability": 0.042,
                "fitted_probability": 0.045,
                "logit_probability": -3.13,
                "fitted_logit_probability": -3.05,
                "n_participants": 128,
                "source_path": "internal-run/mapping.csv",
                "row_ids": ["a", "b"],
            },
            {
                "sex_value": 2,
                "age_bin": 70,
                "representative_probability": 0.057,
                "fitted_probability": 0.061,
                "logit_probability": -2.81,
                "fitted_logit_probability": -2.73,
                "n_participants": 119,
                "source_path": "internal-run/mapping.csv",
                "row_ids": ["c", "d"],
            },
        ]
    )

    public_frame = build_public_motionage_mapping_frame(frame)

    assert public_frame.to_dict("records") == [
        {
            "sex_value": 1,
            "sex_label": "Male",
            "age_bin": 65,
            "representative_probability": 0.042,
            "fitted_probability": 0.045,
            "logit_probability": -3.13,
            "fitted_logit_probability": -3.05,
            "n_participants": 128,
        },
        {
            "sex_value": 2,
            "sex_label": "Female",
            "age_bin": 70,
            "representative_probability": 0.057,
            "fitted_probability": 0.061,
            "logit_probability": -2.81,
            "fitted_logit_probability": -2.73,
            "n_participants": 119,
        },
    ]
    assert "source_path" not in public_frame.columns
    assert "row_ids" not in public_frame.columns


def test_build_public_motionage_mapping_frame_requires_known_sex_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "sex_value": 9,
                "age_bin": 65,
                "representative_probability": 0.042,
                "fitted_probability": 0.045,
                "logit_probability": -3.13,
                "fitted_logit_probability": -3.05,
            }
        ]
    )

    with pytest.raises(ValueError, match="Missing public sex labels"):
        build_public_motionage_mapping_frame(frame)


def test_build_public_motionage_mapping_frame_allows_explicit_public_sex_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "sex_value": 9,
                "age_bin": 65,
                "representative_probability": 0.042,
                "fitted_probability": 0.045,
                "logit_probability": -3.13,
                "fitted_logit_probability": -3.05,
            }
        ]
    )

    public_frame = build_public_motionage_mapping_frame(
        frame,
        sex_labels={9: "Other"},
    )

    assert public_frame[["sex_value", "sex_label"]].to_dict("records") == [
        {"sex_value": 9, "sex_label": "Other"}
    ]


def test_plot_motionage_mapping_diagnostics_uses_age_bin_summary_inputs() -> None:
    public_frame = build_public_motionage_mapping_frame(
        pd.DataFrame(
            [
                {
                    "sex_value": 1,
                    "age_bin": 60,
                    "representative_probability": 0.031,
                    "fitted_probability": 0.034,
                    "logit_probability": -3.44,
                    "fitted_logit_probability": -3.35,
                },
                {
                    "sex_value": 1,
                    "age_bin": 70,
                    "representative_probability": 0.053,
                    "fitted_probability": 0.057,
                    "logit_probability": -2.91,
                    "fitted_logit_probability": -2.81,
                },
                {
                    "sex_value": 2,
                    "age_bin": 60,
                    "representative_probability": 0.024,
                    "fitted_probability": 0.027,
                    "logit_probability": -3.71,
                    "fitted_logit_probability": -3.58,
                },
                {
                    "sex_value": 2,
                    "age_bin": 70,
                    "representative_probability": 0.046,
                    "fitted_probability": 0.049,
                    "logit_probability": -3.03,
                    "fitted_logit_probability": -2.97,
                },
            ]
        )
    )

    ax = plot_motionage_mapping_diagnostics(public_frame, title=None)

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Female observed",
        "Female fitted",
        "Male observed",
        "Male fitted",
    ]
    assert ax.get_xlabel() == "Chronological age"
    assert ax.get_ylabel() == "Representative mortality probability"
    assert ax.get_title() == ""
    assert len(ax.collections) == 2
    assert len(ax.lines) == 2
