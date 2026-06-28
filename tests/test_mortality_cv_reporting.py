from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from motionage.reporting.mortality_cv import (
    collect_fold_metrics,
    summarize_mortality_cv_fold_metrics,
)


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
