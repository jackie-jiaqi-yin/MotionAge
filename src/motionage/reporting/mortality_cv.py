"""Reporting helpers for fixed-parameter mortality CV studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


MORTALITY_CV_METRIC_COLUMNS = (
    "test_auroc",
    "test_auprc",
    "test_logloss",
    "test_brier",
)


def collect_fold_metrics(
    cv_root: str | Path,
    *,
    allowed_model_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Collect secondary model metrics from mortality-CV fold summaries."""
    root = Path(cv_root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/fold_*/fold_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_id = str(summary["model_id"])
        if allowed_model_ids is not None and model_id not in allowed_model_ids:
            continue

        metrics_path = Path(str(summary["stage2_metrics_path"]))
        metrics_frame = pd.read_csv(metrics_path)
        _require_columns(metrics_frame, ["feature_set", *MORTALITY_CV_METRIC_COLUMNS])
        for row in metrics_frame.to_dict(orient="records"):
            feature_set = str(row["feature_set"])
            rows.append(
                {
                    "model_id": model_id,
                    "fold": str(summary["fold"]),
                    "stage2_experiment_id": feature_set,
                    "is_official": feature_set == str(summary["official_feature_set"]),
                    "test_auroc": float(row["test_auroc"]),
                    "test_auprc": float(row["test_auprc"]),
                    "test_logloss": float(row["test_logloss"]),
                    "test_brier": float(row["test_brier"]),
                    "metrics_path": str(metrics_path),
                    "fold_summary_path": str(summary_path),
                }
            )
    return pd.DataFrame(rows)


def summarize_mortality_cv_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize mortality-CV fold metrics as mean and fold standard deviation."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "stage2_experiment_id",
                "fold_count",
                "is_official",
                *[
                    f"{metric}_{stat}"
                    for metric in MORTALITY_CV_METRIC_COLUMNS
                    for stat in ("mean", "std")
                ],
            ]
        )

    _require_columns(
        frame,
        [
            "model_id",
            "stage2_experiment_id",
            "fold",
            "is_official",
            *MORTALITY_CV_METRIC_COLUMNS,
        ],
    )
    grouped = frame.groupby(["model_id", "stage2_experiment_id"], as_index=False).agg(
        fold_count=("fold", "nunique"),
        is_official=("is_official", "max"),
        test_auroc_mean=("test_auroc", "mean"),
        test_auroc_std=("test_auroc", "std"),
        test_auprc_mean=("test_auprc", "mean"),
        test_auprc_std=("test_auprc", "std"),
        test_logloss_mean=("test_logloss", "mean"),
        test_logloss_std=("test_logloss", "std"),
        test_brier_mean=("test_brier", "mean"),
        test_brier_std=("test_brier", "std"),
    )
    return grouped.sort_values(
        ["is_official", "test_auprc_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
