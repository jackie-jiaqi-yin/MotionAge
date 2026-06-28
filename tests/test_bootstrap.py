from __future__ import annotations

import numpy as np
import pytest

from motionage.stats.bootstrap import bootstrap_binary_auroc


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
    assert 0 < first["n_resamples_valid"] <= first["n_resamples_requested"]


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
