from __future__ import annotations

import pandas as pd
import pytest

from motionage.reporting.tables import (
    build_lower_triangle_ci_matrix,
    format_mean_sd,
    render_markdown_table,
    summarize_fold_metrics,
)


def test_summarize_fold_metrics_reports_mean_sd_and_count() -> None:
    frame = pd.DataFrame(
        [
            {"model_label": "Age", "population": "age_ge_40", "fold": "fold_0", "auroc": 0.77, "cindex": 0.71},
            {"model_label": "Age", "population": "age_ge_40", "fold": "fold_1", "auroc": 0.83, "cindex": 0.75},
            {"model_label": "MotionAge-FRC", "population": "age_ge_40", "fold": "fold_0", "auroc": 0.84, "cindex": 0.78},
            {"model_label": "MotionAge-FRC", "population": "age_ge_40", "fold": "fold_1", "auroc": 0.86, "cindex": 0.80},
        ]
    )

    summary = summarize_fold_metrics(
        frame,
        group_columns=["model_label", "population"],
        metric_columns=["auroc", "cindex"],
    )

    age = summary.loc[summary["model_label"] == "Age"].iloc[0]
    motionage = summary.loc[summary["model_label"] == "MotionAge-FRC"].iloc[0]

    assert age["fold_count"] == 2
    assert age["auroc_mean"] == pytest.approx(0.80)
    assert age["auroc_sd"] == pytest.approx(0.0424264)
    assert age["cindex_mean"] == pytest.approx(0.73)
    assert motionage["auroc_mean"] == pytest.approx(0.85)
    assert motionage["cindex_sd"] == pytest.approx(0.01414214)


def test_format_mean_sd_uses_ascii_report_format() -> None:
    assert format_mean_sd(0.851616, 0.01523, digits=3) == "0.852 +/- 0.015"
    assert format_mean_sd(0.851616, 0.01523, digits=4) == "0.8516 +/- 0.0152"
    assert format_mean_sd(float("nan"), 0.01523, digits=3) == ""


def test_build_lower_triangle_ci_matrix_formats_requested_order() -> None:
    paired = pd.DataFrame(
        [
            {
                "left_label": "MotionAge-FRC",
                "right_label": "PhenoAge",
                "observed_auc_delta": 0.018131,
                "ci95_lower": 0.004671,
                "ci95_upper": 0.031983,
            },
            {
                "left_label": "MotionAge-FRC",
                "right_label": "Age",
                "observed_auc_delta": 0.052155,
                "ci95_lower": 0.038438,
                "ci95_upper": 0.066401,
            },
            {
                "left_label": "LLM-Age",
                "right_label": "Age",
                "observed_auc_delta": 0.0034,
                "ci95_lower": -0.0025,
                "ci95_upper": 0.0093,
            },
        ]
    )

    matrix = build_lower_triangle_ci_matrix(
        paired,
        labels=["Age", "LLM-Age", "PhenoAge", "MotionAge-FRC"],
    )

    assert matrix.loc["Age", "Age"] == "-"
    assert matrix.loc["Age", "LLM-Age"] == ""
    assert matrix.loc["LLM-Age", "Age"] == "0.0034 (-0.0025, 0.0093)"
    assert matrix.loc["MotionAge-FRC", "PhenoAge"] == "0.0181 (0.0047, 0.0320)*"
    assert matrix.loc["MotionAge-FRC", "Age"] == "0.0522 (0.0384, 0.0664)*"
    assert matrix.loc["PhenoAge", "LLM-Age"] == ""


def test_lower_triangle_ci_matrix_can_use_reverse_input_orientation() -> None:
    paired = pd.DataFrame(
        [
            {
                "left_label": "Age",
                "right_label": "MotionAge-FRC",
                "observed_auc_delta": -0.052155,
                "ci95_lower": -0.066401,
                "ci95_upper": -0.038438,
            }
        ]
    )

    matrix = build_lower_triangle_ci_matrix(paired, labels=["Age", "MotionAge-FRC"])

    assert matrix.loc["MotionAge-FRC", "Age"] == "0.0522 (0.0384, 0.0664)*"


def test_render_markdown_table_keeps_blank_upper_triangle_cells() -> None:
    matrix = pd.DataFrame(
        [["-", ""], ["0.0522 (0.0384, 0.0664)*", "-"]],
        index=["Age", "MotionAge-FRC"],
        columns=["Age", "MotionAge-FRC"],
    )

    markdown = render_markdown_table(matrix, index_label="Row - Column")

    assert "| Row - Column  | Age                      | MotionAge-FRC |" in markdown
    assert "| Age           | -                        |               |" in markdown
    assert "| MotionAge-FRC | 0.0522 (0.0384, 0.0664)* | -             |" in markdown
