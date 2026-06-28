from __future__ import annotations

from pathlib import Path

import pandas as pd

from motionage import benchmarks
from motionage.benchmarks.llm_age import (
    LLM_AGE_ACCEL_COLUMN,
    LLM_AGE_COLUMN,
    LLM_AGE_FEATURE_SETS,
    discover_cv_folds,
    load_llm_age_participants,
    run_fold_benchmark,
    run_llm_age_benchmark_cv,
)


def test_llm_age_helpers_are_exported_from_benchmarks_namespace() -> None:
    assert benchmarks.LLM_AGE_COLUMN == LLM_AGE_COLUMN
    assert benchmarks.LLM_AGE_FEATURE_SETS == LLM_AGE_FEATURE_SETS
    assert benchmarks.load_llm_age_participants is load_llm_age_participants
    assert benchmarks.run_llm_age_benchmark_cv is run_llm_age_benchmark_cv


def test_load_llm_age_participants_filters_failed_rows_and_builds_60m_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "llm_age.csv"
    pd.DataFrame(
        {
            "SEQN": ["1", "2", "3", "4"],
            "mortstat": ["0", "1", "1", "1"],
            "permth_int": [90, 24, 72, 18],
            "RIDAGEYR": ["45", "50", "55", "57"],
            "RIAGENDR": [1, 2, 1, 2],
            "overall_age": ["47.5", "52.0", "59.0", "60.0"],
            "llm_ok": [True, False, "true", "true"],
            "ignored": ["a", "b", "c", "d"],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_llm_age_participants(csv_path)

    assert loaded.columns.tolist() == ["SEQN", "mortstat", "RIDAGEYR", "RIAGENDR", LLM_AGE_COLUMN]
    assert loaded["SEQN"].tolist() == [1, 3, 4]
    assert loaded["mortstat"].tolist() == [0, 0, 1]
    assert loaded[LLM_AGE_COLUMN].tolist() == [47.5, 59.0, 60.0]


def test_run_fold_benchmark_reports_age_acceleration_and_population_metrics(tmp_path: Path) -> None:
    participants = load_llm_age_participants(_write_participants(tmp_path, _base_participants()))

    predictions, metrics = run_fold_benchmark(
        participants,
        train_ids=pd.Series([1, 2, 3, 4], name="SEQN"),
        test_ids=pd.Series([5, 6], name="SEQN"),
        age_threshold=40,
    )

    assert set(predictions["feature_set"]) == set(LLM_AGE_FEATURE_SETS)
    assert set(predictions["split"]) == {"train", "test"}
    assert set(metrics["population"]) == {"overall", "age_ge_40"}
    assert set(metrics["feature_set"]) == set(LLM_AGE_FEATURE_SETS)

    returned_test = (
        predictions[(predictions["SEQN"] == 5) & (predictions["split"] == "test")]
        .drop_duplicates(subset=["SEQN"])[["SEQN", LLM_AGE_COLUMN, LLM_AGE_ACCEL_COLUMN]]
        .reset_index(drop=True)
    )
    assert returned_test.shape[0] == 1
    assert returned_test.loc[0, LLM_AGE_COLUMN] == 64.0
    assert returned_test.loc[0, LLM_AGE_ACCEL_COLUMN] == -1.0


def test_run_llm_age_benchmark_cv_writes_local_fold_outputs_and_summary(tmp_path: Path) -> None:
    participants = load_llm_age_participants(_write_participants(tmp_path, _base_participants()))
    fold_root = _write_folds(tmp_path)

    output_root = tmp_path / "benchmark"
    summary = run_llm_age_benchmark_cv(
        participants,
        fold_root=fold_root,
        output_root=output_root,
        age_threshold=40,
        fold_limit=1,
    )

    assert [path.name for path in discover_cv_folds(fold_root, fold_limit=1)] == ["fold_0"]
    assert (output_root / "fold_0" / "participant_predictions.csv").exists()
    assert (output_root / "fold_0" / "metrics.csv").exists()
    assert (output_root / "summary.csv").exists()
    assert set(summary["feature_set"]) == set(LLM_AGE_FEATURE_SETS)
    assert set(summary["population"]) == {"overall", "age_ge_40"}


def _base_participants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SEQN": [1, 2, 3, 4, 5, 6],
            "mortstat": [0, 0, 1, 1, 0, 1],
            "permth_int": [80, 75, 30, 120, 72, 18],
            "RIDAGEYR": [45, 50, 55, 60, 65, 70],
            "RIAGENDR": [1, 2, 1, 2, 1, 2],
            "overall_age": [44.0, 51.0, 58.0, 63.0, 64.0, 76.0],
            "llm_ok": [True, True, True, True, True, True],
        }
    )


def _write_participants(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "llm_age.csv"
    frame.to_csv(path, index=False)
    return path


def _write_folds(tmp_path: Path) -> Path:
    fold_root = tmp_path / "folds"
    (fold_root / "fold_0").mkdir(parents=True)
    (fold_root / "fold_1").mkdir(parents=True)
    pd.DataFrame({"SEQN": [1, 2, 3, 4]}).to_csv(fold_root / "fold_0" / "train_ids.csv", index=False)
    pd.DataFrame({"SEQN": [5, 6]}).to_csv(fold_root / "fold_0" / "test_ids.csv", index=False)
    pd.DataFrame({"SEQN": [3, 4, 5, 6]}).to_csv(fold_root / "fold_1" / "train_ids.csv", index=False)
    pd.DataFrame({"SEQN": [1, 2]}).to_csv(fold_root / "fold_1" / "test_ids.csv", index=False)
    return fold_root
