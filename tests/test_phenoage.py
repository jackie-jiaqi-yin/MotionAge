from __future__ import annotations

import math

import pandas as pd
import pytest

from motionage.benchmarks.phenoage import (
    PHENOAGE_BIOMARKER_COLUMNS,
    PHENOAGE_INPUT_COLUMNS,
    compute_phenoage,
    fit_phenoage_imputer,
    select_complete_phenoage_cases,
    summarize_phenoage_missingness,
)


def _phenoage_frame(rows: list[dict[str, float | int | None]]) -> pd.DataFrame:
    defaults: dict[str, float | int] = {
        "SEQN": 1,
        "RIDAGEYR": 50,
        "RIAGENDR": 1,
        "LBDSALSI": 43.0,
        "LBDSCRSI": 80.0,
        "LBDSGLSI": 5.2,
        "log_crp": math.log(0.3),
        "LBXLYPCT": 30.0,
        "MCV": 90.0,
        "LBXRDW": 13.0,
        "LBXSAPSI": 65.0,
        "LBXWBCSI": 6.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_compute_phenoage_matches_published_formula() -> None:
    frame = _phenoage_frame([{"SEQN": 10, "RIDAGEYR": 60}])

    phenoage = compute_phenoage(frame)

    xb = (
        -19.9067
        + -0.0336 * 43.0
        + 0.0095 * 80.0
        + 0.1953 * 5.2
        + 0.0954 * math.log(0.3)
        + -0.0120 * 30.0
        + 0.0268 * 90.0
        + 0.3306 * 13.0
        + 0.0019 * 65.0
        + 0.0554 * 6.0
        + 0.0804 * 60.0
    )
    mortality_score = 1 - math.exp(-math.exp(xb) * (math.exp(0.0076927 * 120) - 1) / 0.0076927)
    expected = 141.50225 + math.log(-0.00553 * math.log(1 - mortality_score)) / 0.090165

    assert phenoage.name == "PhenoAge"
    assert phenoage.iloc[0] == pytest.approx(expected)


def test_summarize_phenoage_missingness_counts_overall_and_age_threshold() -> None:
    frame = _phenoage_frame(
        [
            {"SEQN": 1, "RIDAGEYR": 39, "LBDSALSI": None},
            {"SEQN": 2, "RIDAGEYR": 45, "LBDSALSI": None},
            {"SEQN": 3, "RIDAGEYR": 60, "MCV": None},
        ]
    )

    summary = summarize_phenoage_missingness(frame, age_threshold=40)

    albumin = summary.loc[summary["column"] == "LBDSALSI"].iloc[0]
    age = summary.loc[summary["column"] == "RIDAGEYR"].iloc[0]
    mcv = summary.loc[summary["column"] == "MCV"].iloc[0]

    assert albumin["input_role"] == "biomarker"
    assert albumin["n_total"] == 3
    assert albumin["n_missing"] == 2
    assert albumin["missing_rate"] == pytest.approx(2 / 3)
    assert albumin["n_age_ge_threshold"] == 2
    assert albumin["n_missing_age_ge_threshold"] == 1
    assert albumin["missing_rate_age_ge_threshold"] == pytest.approx(0.5)
    assert age["input_role"] == "chronological_age"
    assert mcv["n_missing_age_ge_threshold"] == 1


def test_train_median_imputer_uses_training_values_for_held_out_rows() -> None:
    train = _phenoage_frame(
        [
            {"SEQN": 1, "LBDSALSI": 40.0, "LBDSGLSI": 4.0},
            {"SEQN": 2, "LBDSALSI": 50.0, "LBDSGLSI": 8.0},
        ]
    )
    held_out = _phenoage_frame(
        [
            {"SEQN": 3, "LBDSALSI": None, "LBDSGLSI": None},
            {"SEQN": 4, "LBDSALSI": 47.0, "LBDSGLSI": None},
        ]
    )

    imputer = fit_phenoage_imputer(train, strategy="median")
    filled = imputer.transform(held_out)

    assert filled.loc[filled["SEQN"] == 3, "LBDSALSI"].iloc[0] == pytest.approx(45.0)
    assert filled.loc[filled["SEQN"] == 4, "LBDSALSI"].iloc[0] == pytest.approx(47.0)
    assert filled["LBDSGLSI"].tolist() == pytest.approx([6.0, 6.0])


def test_age_sex_median_imputer_uses_stratum_then_sex_then_global_fallback() -> None:
    train = _phenoage_frame(
        [
            {"SEQN": 1, "RIAGENDR": 1, "RIDAGEYR": 42, "LBDSALSI": 40.0},
            {"SEQN": 2, "RIAGENDR": 1, "RIDAGEYR": 48, "LBDSALSI": 60.0},
            {"SEQN": 3, "RIAGENDR": 1, "RIDAGEYR": 76, "LBDSALSI": 100.0},
            {"SEQN": 4, "RIAGENDR": 2, "RIDAGEYR": 44, "LBDSALSI": 80.0},
        ]
    )
    held_out = _phenoage_frame(
        [
            {"SEQN": 5, "RIAGENDR": 1, "RIDAGEYR": 44, "LBDSALSI": None},
            {"SEQN": 6, "RIAGENDR": 1, "RIDAGEYR": 64, "LBDSALSI": None},
            {"SEQN": 7, "RIAGENDR": 9, "RIDAGEYR": 84, "LBDSALSI": None},
        ]
    )

    imputer = fit_phenoage_imputer(train, strategy="age_sex_median")
    filled = imputer.transform(held_out)

    assert filled["LBDSALSI"].tolist() == pytest.approx([50.0, 60.0, 70.0])


def test_complete_case_selector_removes_rows_with_any_missing_phenoage_input() -> None:
    frame = _phenoage_frame(
        [
            {"SEQN": 1},
            {"SEQN": 2, "LBDSALSI": None},
            {"SEQN": 3, "MCV": None},
        ]
    )

    complete = select_complete_phenoage_cases(frame)

    assert tuple(complete["SEQN"]) == (1,)
    assert set(PHENOAGE_BIOMARKER_COLUMNS).issubset(PHENOAGE_INPUT_COLUMNS)
