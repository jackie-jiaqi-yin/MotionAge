from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from motionage.reporting.mortality_cv import (
    build_mortality_cv_summary_table,
    build_public_mortality_cv_rank_table,
    build_public_mortality_cv_summary_table,
    collect_fold_metrics,
    summarize_mortality_cv_fold_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reports_docs_describe_public_mortality_cv_table_helper() -> None:
    reports_doc = (REPO_ROOT / "docs" / "reports" / "README.md").read_text(encoding="utf-8")

    assert "build_public_mortality_cv_summary_table" in reports_doc
    assert "reader-facing labels" in reports_doc
    assert "private experiment paths" in reports_doc


def _write_fold_summary(
    fold_dir: Path,
    *,
    model_id: str,
    fold: str,
    official_feature_set: str,
    metrics: list[dict[str, object]],
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_dir / "secondary_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "fold": fold,
                "official_feature_set": official_feature_set,
                "stage2_metrics_path": str(metrics_path),
            }
        ),
        encoding="utf-8",
    )


def test_collect_fold_metrics_filters_to_allowed_models(tmp_path: Path) -> None:
    cv_root = tmp_path / "mortality_cv_primary_60m"
    _write_fold_summary(
        cv_root / "gru_fitbit_only" / "fold_0",
        model_id="gru_fitbit_only",
        fold="fold_0",
        official_feature_set="motionage_accel",
        metrics=[
            {
                "feature_set": "motionage_accel",
                "test_auroc": 0.81,
                "test_auprc": 0.31,
                "test_logloss": 0.49,
                "test_brier": 0.18,
            }
        ],
    )
    _write_fold_summary(
        cv_root / "transformer_level1_latefusion" / "fold_0",
        model_id="transformer_level1_latefusion",
        fold="fold_0",
        official_feature_set="motionage_accel",
        metrics=[
            {
                "feature_set": "motionage_accel",
                "test_auroc": 0.75,
                "test_auprc": 0.25,
                "test_logloss": 0.55,
                "test_brier": 0.22,
            }
        ],
    )

    rows = collect_fold_metrics(cv_root, allowed_model_ids={"gru_fitbit_only"})

    assert rows["model_id"].tolist() == ["gru_fitbit_only"]
    assert rows["fold"].tolist() == ["fold_0"]
    assert rows["stage2_experiment_id"].tolist() == ["motionage_accel"]
    assert rows["is_official"].tolist() == [True]
    assert rows["test_auroc"].tolist() == [0.81]


def test_summarize_mortality_cv_fold_metrics_orders_official_rows_first() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["gru_fitbit_only", "gru_fitbit_only", "gru_fitbit_only", "gru_fitbit_only"],
            "stage2_experiment_id": [
                "motionage_accel",
                "motionage_accel",
                "stage1_probability",
                "stage1_probability",
            ],
            "fold": ["fold_0", "fold_1", "fold_0", "fold_1"],
            "test_auroc": [0.80, 0.82, 0.78, 0.79],
            "test_auprc": [0.30, 0.31, 0.28, 0.29],
            "test_logloss": [0.50, 0.49, 0.53, 0.52],
            "test_brier": [0.19, 0.18, 0.20, 0.20],
            "is_official": [True, True, False, False],
        }
    )

    summary = summarize_mortality_cv_fold_metrics(frame)

    assert summary[["model_id", "stage2_experiment_id"]].values.tolist() == [
        ["gru_fitbit_only", "motionage_accel"],
        ["gru_fitbit_only", "stage1_probability"],
    ]
    official_row = summary[summary["stage2_experiment_id"] == "motionage_accel"].iloc[0]
    assert bool(official_row["is_official"]) is True
    assert official_row["fold_count"] == 2
    assert round(official_row["test_auroc_mean"], 4) == 0.8100
    assert round(official_row["test_auprc_mean"], 4) == 0.3050
    assert round(official_row["test_logloss_mean"], 4) == 0.4950


def test_build_mortality_cv_summary_table_formats_aggregate_metrics() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_id": "gru_fitbit_only",
                "stage2_experiment_id": "motionage_accel",
                "fold_count": 5,
                "is_official": True,
                "test_auroc_mean": 0.81234,
                "test_auroc_std": 0.01567,
                "test_auprc_mean": 0.31234,
                "test_auprc_std": 0.02567,
                "test_logloss_mean": 0.48765,
                "test_logloss_std": 0.01012,
                "test_brier_mean": 0.18123,
                "test_brier_std": 0.00567,
            },
            {
                "model_id": "transformer_level1_latefusion",
                "stage2_experiment_id": "stage1_probability",
                "fold_count": 5,
                "is_official": False,
                "test_auroc_mean": 0.79876,
                "test_auroc_std": 0.01432,
                "test_auprc_mean": 0.29876,
                "test_auprc_std": 0.02432,
                "test_logloss_mean": 0.50123,
                "test_logloss_std": 0.01234,
                "test_brier_mean": 0.19123,
                "test_brier_std": 0.00678,
            },
        ]
    )

    table = build_mortality_cv_summary_table(summary, digits=3)

    assert table.columns.tolist() == [
        "model_id",
        "stage2_experiment_id",
        "fold_count",
        "is_official",
        "test_auroc",
        "test_auprc",
        "test_logloss",
        "test_brier",
    ]
    assert table.iloc[0].to_dict() == {
        "model_id": "gru_fitbit_only",
        "stage2_experiment_id": "motionage_accel",
        "fold_count": 5,
        "is_official": True,
        "test_auroc": "0.812 +/- 0.016",
        "test_auprc": "0.312 +/- 0.026",
        "test_logloss": "0.488 +/- 0.010",
        "test_brier": "0.181 +/- 0.006",
    }


def test_build_public_mortality_cv_summary_table_uses_reader_labels_and_omits_private_paths() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_id": "gru_fitbit_only",
                "stage2_experiment_id": "motionage_accel",
                "fold_count": 5,
                "is_official": True,
                "test_auroc_mean": 0.81234,
                "test_auroc_std": 0.01567,
                "test_auprc_mean": 0.31234,
                "test_auprc_std": 0.02567,
                "test_logloss_mean": 0.48765,
                "test_logloss_std": 0.01012,
                "test_brier_mean": 0.18123,
                "test_brier_std": 0.00567,
                "metrics_path": "internal-run/secondary_metrics.csv",
            },
            {
                "model_id": "gru_fitbit_only",
                "stage2_experiment_id": "stage1_probability",
                "fold_count": 5,
                "is_official": False,
                "test_auroc_mean": 0.79876,
                "test_auroc_std": 0.01432,
                "test_auprc_mean": 0.29876,
                "test_auprc_std": 0.02432,
                "test_logloss_mean": 0.50123,
                "test_logloss_std": 0.01234,
                "test_brier_mean": 0.19123,
                "test_brier_std": 0.00678,
                "metrics_path": "internal-run/secondary_metrics.csv",
            },
        ]
    )

    table = build_public_mortality_cv_summary_table(
        summary,
        model_labels={"gru_fitbit_only": "GRU wearable"},
        feature_labels={"motionage_accel": "MotionAge acceleration"},
        official_only=True,
        digits=3,
    )

    assert table.columns.tolist() == [
        "model",
        "feature_set",
        "fold_count",
        "test_auroc",
        "test_auprc",
        "test_logloss",
        "test_brier",
    ]
    assert table.to_dict("records") == [
        {
            "model": "GRU wearable",
            "feature_set": "MotionAge acceleration",
            "fold_count": 5,
            "test_auroc": "0.812 +/- 0.016",
            "test_auprc": "0.312 +/- 0.026",
            "test_logloss": "0.488 +/- 0.010",
            "test_brier": "0.181 +/- 0.006",
        }
    ]
    assert "metrics_path" not in table.columns
    assert "model_id" not in table.columns
    assert "stage2_experiment_id" not in table.columns


def test_build_public_mortality_cv_rank_table_orders_aggregate_model_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_id": "gru_level1_latefusion",
                "stage2_experiment_id": "motionage_accel",
                "fold_count": 5,
                "is_official": True,
                "test_auroc_mean": 0.836,
                "test_auroc_std": 0.014,
                "test_auprc_mean": 0.331,
                "test_auprc_std": 0.021,
                "metrics_path": "internal-run/gru.csv",
            },
            {
                "model_id": "lstm_level1_latefusion",
                "stage2_experiment_id": "motionage_accel",
                "fold_count": 5,
                "is_official": True,
                "test_auroc_mean": 0.824,
                "test_auroc_std": 0.016,
                "test_auprc_mean": 0.318,
                "test_auprc_std": 0.024,
                "metrics_path": "internal-run/lstm.csv",
            },
            {
                "model_id": "transformer_level1_latefusion",
                "stage2_experiment_id": "stage1_probability",
                "fold_count": 5,
                "is_official": False,
                "test_auroc_mean": 0.818,
                "test_auroc_std": 0.018,
                "test_auprc_mean": 0.311,
                "test_auprc_std": 0.026,
                "metrics_path": "internal-run/transformer.csv",
            },
        ]
    )

    table = build_public_mortality_cv_rank_table(
        summary,
        model_labels={
            "gru_level1_latefusion": "GRU MotionAge-FRC",
            "lstm_level1_latefusion": "LSTM MotionAge-FRC",
            "transformer_level1_latefusion": "Transformer MotionAge-FRC",
        },
        feature_labels={"motionage_accel": "MotionAge acceleration"},
        metric="test_auroc",
        official_only=True,
        digits=3,
    )

    assert table.to_dict("records") == [
        {
            "rank": 1,
            "model": "GRU MotionAge-FRC",
            "feature_set": "MotionAge acceleration",
            "fold_count": 5,
            "metric": "test_auroc",
            "mean": 0.836,
            "sd": 0.014,
            "mean_sd": "0.836 +/- 0.014",
        },
        {
            "rank": 2,
            "model": "LSTM MotionAge-FRC",
            "feature_set": "MotionAge acceleration",
            "fold_count": 5,
            "metric": "test_auroc",
            "mean": 0.824,
            "sd": 0.016,
            "mean_sd": "0.824 +/- 0.016",
        },
    ]
    assert "model_id" not in table.columns
    assert "stage2_experiment_id" not in table.columns
    assert "metrics_path" not in table.columns
