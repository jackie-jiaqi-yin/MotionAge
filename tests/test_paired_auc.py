from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motionage.stats.paired_auc import (
    fold_structured_paired_bootstrap_auc_delta,
    make_paired_score_frame,
    paired_auc_delta,
    paired_bootstrap_auc_delta,
    safe_auc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _score_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame(
        {
            "SEQN": [1, 2, 3, 4, 5, 6],
            "target": [0, 0, 1, 1, 0, 1],
            "RIDAGEYR": [35, 45, 55, 65, 75, 85],
            "fold": [0, 0, 0, 1, 1, 1],
            "score": [0.10, 0.20, 0.85, 0.75, 0.30, 0.90],
        }
    )
    right = pd.DataFrame(
        {
            "SEQN": [1, 2, 3, 4, 5, 6],
            "target": [0, 0, 1, 1, 0, 1],
            "RIDAGEYR": [35, 45, 55, 65, 75, 85],
            "fold": [0, 0, 0, 1, 1, 1],
            "score": [0.20, 0.70, 0.60, 0.65, 0.50, 0.80],
        }
    )
    return left, right


def test_method_docs_describe_bootstrap_metadata_fields() -> None:
    method_doc = (REPO_ROOT / "docs" / "method.md").read_text(encoding="utf-8")

    for term in ("n_resamples_requested", "ci_level", "stratified", "fold_stratified"):
        assert term in method_doc


def test_safe_auc_matches_rank_definition_with_ties() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.2, 0.5, 0.5, 0.9])

    assert safe_auc(y, score) == pytest.approx(0.875)


def test_safe_auc_returns_nan_for_single_class_targets() -> None:
    assert math.isnan(safe_auc([1, 1, 1], [0.2, 0.4, 0.8]))


def test_make_paired_score_frame_pairs_shared_participants_and_filters_age() -> None:
    left, right = _score_tables()

    paired = make_paired_score_frame(left, right, min_age=40)

    assert paired["SEQN"].tolist() == [2, 3, 4, 5, 6]
    assert paired["target"].tolist() == [0, 1, 1, 0, 1]
    assert paired["score_left"].tolist() == [0.20, 0.85, 0.75, 0.30, 0.90]
    assert paired["score_right"].tolist() == [0.70, 0.60, 0.65, 0.50, 0.80]


def test_make_paired_score_frame_rejects_target_mismatch() -> None:
    left, right = _score_tables()
    right.loc[right["SEQN"] == 3, "target"] = 0

    with pytest.raises(ValueError, match="target mismatches"):
        make_paired_score_frame(left, right)


def test_paired_auc_delta_reports_observed_left_right_and_delta() -> None:
    left, right = _score_tables()
    paired = make_paired_score_frame(left, right)

    result = paired_auc_delta(paired)

    assert result["n"] == 6
    assert result["events"] == 3
    assert result["non_events"] == 3
    assert result["auc_left"] == pytest.approx(1.0)
    assert result["auc_right"] == pytest.approx(7 / 9)
    assert result["auc_delta"] == pytest.approx(2 / 9)


def test_stratified_paired_bootstrap_is_reproducible_and_keeps_resample_count() -> None:
    left, right = _score_tables()
    paired = make_paired_score_frame(left, right)

    first = paired_bootstrap_auc_delta(
        paired,
        n_resamples=200,
        random_seed=11,
        stratified=True,
    )
    second = paired_bootstrap_auc_delta(
        paired,
        n_resamples=200,
        random_seed=11,
        stratified=True,
    )

    assert first == second
    assert first["n_resamples_requested"] == 200
    assert first["valid_resamples"] == 200
    assert first["ci_level"] == 0.95
    assert first["stratified"] is True
    assert first["ci95_lower"] <= first["mean"] <= first["ci95_upper"]
    assert 0.0 <= first["p_value"] <= 1.0


def test_fold_structured_bootstrap_averages_fold_level_paired_deltas() -> None:
    left, right = _score_tables()
    paired = make_paired_score_frame(left, right)

    result = fold_structured_paired_bootstrap_auc_delta(
        paired,
        n_resamples=100,
        random_seed=7,
    )

    assert result["valid_folds"] == 2
    assert result["n_resamples_requested"] == 100
    assert result["valid_resamples"] == 100
    assert result["ci_level"] == 0.95
    assert result["resampling_unit"] == "fold_stratified"
    assert result["observed_auc_left"] == pytest.approx(1.0)
    assert result["observed_auc_right"] == pytest.approx(0.75)
    assert result["observed_auc_delta"] == pytest.approx(0.25)
    assert result["ci95_lower"] <= result["bootstrap_mean_delta"] <= result["ci95_upper"]


def test_bootstrap_summaries_return_nan_when_auc_is_not_defined() -> None:
    paired = pd.DataFrame(
        {
            "SEQN": [1, 2, 3],
            "fold": [0, 0, 0],
            "target": [1, 1, 1],
            "RIDAGEYR": [50, 60, 70],
            "score_left": [0.2, 0.3, 0.4],
            "score_right": [0.1, 0.2, 0.3],
        }
    )

    result = paired_bootstrap_auc_delta(paired, n_resamples=10, random_seed=1, stratified=True)

    assert result["valid_resamples"] == 0
    assert result["n_resamples_requested"] == 10
    assert result["ci_level"] == 0.95
    assert result["stratified"] is True
    assert math.isnan(result["mean"])
