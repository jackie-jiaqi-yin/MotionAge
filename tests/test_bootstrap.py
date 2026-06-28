from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motionage.stats.bootstrap import (
    bootstrap_binary_auroc,
    public_bootstrap_interval_row,
    public_bootstrap_interval_table,
    public_paired_auc_interval_table,
)


def test_bootstrap_binary_auroc_returns_reproducible_ci() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])

    first = bootstrap_binary_auroc(
        y_true,
        y_prob,
        n_resamples=200,
        ci_level=0.95,
        random_seed=7,
    )
    second = bootstrap_binary_auroc(
        y_true,
        y_prob,
        n_resamples=200,
        ci_level=0.95,
        random_seed=7,
    )

    assert first == second
    assert first["observed_auroc"] == 1.0
    assert first["ci_lower"] <= first["observed_auroc"] <= first["ci_upper"]
    assert first["ci_level"] == 0.95
    assert 0 < first["n_resamples_valid"] <= first["n_resamples_requested"]


def test_public_bootstrap_interval_row_normalizes_binary_auroc_summary() -> None:
    summary = {
        "observed_auroc": 0.81,
        "bootstrap_mean_auroc": 0.80,
        "ci_lower": 0.72,
        "ci_upper": 0.88,
        "ci_level": 0.95,
        "n_resamples_requested": 1000,
        "n_resamples_valid": 997,
        "private_resample_values": [0.7, 0.8],
    }

    row = public_bootstrap_interval_row(summary, metric="AUROC", resampling_unit="participant")

    assert row == {
        "metric": "AUROC",
        "estimate": 0.81,
        "bootstrap_mean": 0.80,
        "ci_lower": 0.72,
        "ci_upper": 0.88,
        "ci_level": 0.95,
        "n_resamples_requested": 1000,
        "valid_resamples": 997,
        "resampling_unit": "participant",
    }
    assert "private_resample_values" not in row


def test_public_bootstrap_interval_row_normalizes_fold_structured_delta_summary() -> None:
    summary = {
        "observed_auc_delta": 0.06,
        "bootstrap_mean_delta": 0.05,
        "ci95_lower": 0.01,
        "ci95_upper": 0.09,
        "ci_level": 0.95,
        "n_resamples_requested": 2000,
        "valid_resamples": 2000,
        "p_value": 0.04,
        "resampling_unit": "fold_stratified",
        "valid_folds": 5,
    }

    row = public_bootstrap_interval_row(summary, metric="paired AUROC delta", comparison="MotionAge - PhenoAge")

    assert row == {
        "metric": "paired AUROC delta",
        "comparison": "MotionAge - PhenoAge",
        "estimate": 0.06,
        "bootstrap_mean": 0.05,
        "ci_lower": 0.01,
        "ci_upper": 0.09,
        "ci_level": 0.95,
        "n_resamples_requested": 2000,
        "valid_resamples": 2000,
        "resampling_unit": "fold_stratified",
        "valid_folds": 5,
        "p_value": 0.04,
    }


def test_public_bootstrap_interval_table_normalizes_multiple_summaries() -> None:
    summaries = [
        {
            "metric": "AUROC",
            "comparison": "MotionAge",
            "observed_auroc": 0.81,
            "bootstrap_mean_auroc": 0.80,
            "ci_lower": 0.72,
            "ci_upper": 0.88,
            "ci_level": 0.95,
            "n_resamples_requested": 1000,
            "n_resamples_valid": 997,
            "raw_resamples": [0.72, 0.88],
        },
        {
            "metric": "paired AUROC delta",
            "comparison": "MotionAge - PhenoAge",
            "observed_auc_delta": 0.06,
            "bootstrap_mean_delta": 0.05,
            "ci95_lower": 0.01,
            "ci95_upper": 0.09,
            "ci_level": 0.95,
            "n_resamples_requested": 2000,
            "valid_resamples": 2000,
            "resampling_unit": "fold_stratified",
            "local_resample_path": "local-bootstrap/fold0.csv",
        },
    ]

    table = public_bootstrap_interval_table(summaries, resampling_unit="participant")

    assert table == [
        {
            "metric": "AUROC",
            "comparison": "MotionAge",
            "estimate": 0.81,
            "bootstrap_mean": 0.80,
            "ci_lower": 0.72,
            "ci_upper": 0.88,
            "ci_level": 0.95,
            "n_resamples_requested": 1000,
            "valid_resamples": 997,
            "resampling_unit": "participant",
        },
        {
            "metric": "paired AUROC delta",
            "comparison": "MotionAge - PhenoAge",
            "estimate": 0.06,
            "bootstrap_mean": 0.05,
            "ci_lower": 0.01,
            "ci_upper": 0.09,
            "ci_level": 0.95,
            "n_resamples_requested": 2000,
            "valid_resamples": 2000,
            "resampling_unit": "participant",
        },
    ]
    assert "raw_resamples" not in table[0]
    assert "local_resample_path" not in table[1]


def test_public_bootstrap_interval_table_preserves_public_context_labels_only() -> None:
    summaries = [
        {
            "analysis": "complete-case sensitivity",
            "population": "age_ge_40",
            "metric": "paired AUROC delta",
            "comparison": "MotionAge - PhenoAge",
            "observed_auc_delta": 0.018,
            "bootstrap_mean_delta": 0.017,
            "ci95_lower": 0.004,
            "ci95_upper": 0.032,
            "ci_level": 0.95,
            "n_resamples_requested": 2000,
            "valid_resamples": 1998,
            "private_note": "do not publish",
            "local_artifact_path": "generated/bootstrap.csv",
            "raw_resamples": [0.004, 0.032],
        }
    ]

    table = public_bootstrap_interval_table(summaries, resampling_unit="participant")

    assert table == [
        {
            "analysis": "complete-case sensitivity",
            "population": "age_ge_40",
            "metric": "paired AUROC delta",
            "comparison": "MotionAge - PhenoAge",
            "estimate": 0.018,
            "bootstrap_mean": 0.017,
            "ci_lower": 0.004,
            "ci_upper": 0.032,
            "ci_level": 0.95,
            "n_resamples_requested": 2000,
            "valid_resamples": 1998,
            "resampling_unit": "participant",
        }
    ]
    assert "private_note" not in table[0]
    assert "local_artifact_path" not in table[0]
    assert "raw_resamples" not in table[0]


def test_public_paired_auc_interval_table_summarizes_public_comparison_rows() -> None:
    paired = pd.DataFrame(
        {
            "SEQN": [1, 2, 3, 4, 5, 6],
            "fold": [0, 0, 0, 1, 1, 1],
            "target": [0, 0, 1, 1, 0, 1],
            "RIDAGEYR": [35, 45, 55, 65, 75, 85],
            "score_left": [0.10, 0.20, 0.85, 0.75, 0.30, 0.90],
            "score_right": [0.20, 0.70, 0.60, 0.65, 0.50, 0.80],
        }
    )

    rows = public_paired_auc_interval_table(
        paired,
        left_label="MotionAge-FRC",
        right_label="PhenoAge",
        n_resamples=100,
        random_seed=7,
    )

    assert [row["resampling_unit"] for row in rows] == [
        "participant",
        "participant_stratified",
        "fold_stratified",
    ]
    assert rows[0]["estimate"] == pytest.approx(2 / 9)
    assert rows[1]["estimate"] == pytest.approx(2 / 9)
    assert rows[2]["estimate"] == pytest.approx(0.25)
    for row in rows:
        assert row["metric"] == "paired AUROC delta"
        assert row["comparison"] == "MotionAge-FRC - PhenoAge"
        assert row["left_label"] == "MotionAge-FRC"
        assert row["right_label"] == "PhenoAge"
        assert row["n"] == 6
        assert row["events"] == 3
        assert row["non_events"] == 3
        assert row["ci_level"] == 0.95
        assert row["n_resamples_requested"] == 100
        assert 0 <= row["valid_resamples"] <= 100
        assert "SEQN" not in row
        assert "score_left" not in row
        assert "score_right" not in row


def test_bootstrap_binary_auroc_returns_observed_for_single_class_targets() -> None:
    summary = bootstrap_binary_auroc(
        np.array([0, 0, 0]),
        np.array([0.1, 0.2, 0.3]),
        n_resamples=50,
        ci_level=0.95,
        random_seed=11,
    )

    assert summary == {
        "observed_auroc": 0.5,
        "bootstrap_mean_auroc": 0.5,
        "ci_lower": 0.5,
        "ci_upper": 0.5,
        "ci_level": 0.95,
        "n_resamples_requested": 50,
        "n_resamples_valid": 0,
    }


def test_bootstrap_binary_auroc_validates_inputs_and_parameters() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_binary_auroc(
            np.array([0, 1]),
            np.array([0.2]),
            n_resamples=50,
            ci_level=0.95,
            random_seed=1,
        )

    with pytest.raises(ValueError, match="positive"):
        bootstrap_binary_auroc(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            n_resamples=0,
            ci_level=0.95,
            random_seed=1,
        )

    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_binary_auroc(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            n_resamples=50,
            ci_level=1.5,
            random_seed=1,
        )
