from __future__ import annotations

import numpy as np
import pytest

from motionage.stats.bootstrap import bootstrap_binary_auroc, public_bootstrap_interval_row


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
