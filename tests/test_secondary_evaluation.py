from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from motionage.evaluation.secondary import evaluate_secondary_feature_sets


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_secondary_feature_sets_returns_aggregate_public_rows() -> None:
    participants = _secondary_participants()

    rows = evaluate_secondary_feature_sets(
        participants,
        feature_sets=[
            {"name": "chronological_age_baseline", "columns": ["RIDAGEYR", "sex_code"]},
            {"name": "motionage_raw", "columns": ["MotionAge", "sex_code"]},
            {"name": "disabled", "columns": ["MotionAge"], "enabled": False},
        ],
        fit_partitions=["train", "validation"],
        target_column="mortstat_60m",
        baseline_feature_set="chronological_age_baseline",
    )

    assert [row["feature_set"] for row in rows] == ["chronological_age_baseline", "motionage_raw"]
    assert rows[0]["feature_columns"] == "RIDAGEYR,sex_code"
    assert rows[0]["train_n"] == 6
    assert rows[0]["test_n"] == 4
    assert rows[0]["train_events"] == 3
    assert rows[0]["test_events"] == 2
    assert rows[0]["delta_vs_chronological_age_baseline"] == pytest.approx(0.0)
    assert rows[1]["delta_vs_chronological_age_baseline"] == pytest.approx(
        rows[1]["test_auroc"] - rows[0]["test_auroc"]
    )
    for row in rows:
        assert "participant_predictions" not in row
        assert "source_path" not in row


def test_secondary_evaluation_docs_describe_public_aggregate_boundary() -> None:
    method_doc = (REPO_ROOT / "docs" / "method.md").read_text(encoding="utf-8")
    reports_doc = (REPO_ROOT / "docs" / "reports" / "README.md").read_text(encoding="utf-8")

    assert "evaluate_secondary_feature_sets" in method_doc
    assert "aggregate secondary-evaluation rows" in reports_doc
    assert "participant-level prediction tables" in reports_doc


def _secondary_participants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": [
                "train",
                "train",
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
                "test",
                "test",
            ],
            "mortstat_60m": [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            "RIDAGEYR": [42, 47, 61, 68, 50, 66, 44, 52, 63, 72],
            "sex_code": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "MotionAge": [44, 48, 66, 73, 52, 70, 46, 53, 68, 78],
        }
    )
