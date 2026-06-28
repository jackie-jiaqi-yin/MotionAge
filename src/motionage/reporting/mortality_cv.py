"""Reporting helpers for fixed-parameter mortality CV studies."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from motionage.reporting.tables import format_mean_sd


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


def build_mortality_cv_summary_table(
    summary: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = MORTALITY_CV_METRIC_COLUMNS,
    digits: int = 3,
    missing: str = "",
) -> pd.DataFrame:
    """Format aggregate mortality-CV summaries for publication tables."""
    metric_columns = list(metric_columns)
    base_columns = ["model_id", "stage2_experiment_id", "fold_count", "is_official"]
    output_columns = [*base_columns, *metric_columns]
    if summary.empty:
        return pd.DataFrame(columns=output_columns)

    required_metric_columns = [
        column
        for metric in metric_columns
        for column in (f"{metric}_mean", f"{metric}_std")
    ]
    _require_columns(summary, [*base_columns, *required_metric_columns])

    rows: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        table_row: dict[str, object] = {column: row[column] for column in base_columns}
        for metric in metric_columns:
            table_row[metric] = format_mean_sd(
                row[f"{metric}_mean"],
                row[f"{metric}_std"],
                digits=digits,
                missing=missing,
            )
        rows.append(table_row)
    return pd.DataFrame(rows, columns=output_columns)


def build_public_mortality_cv_summary_table(
    summary: pd.DataFrame,
    *,
    model_labels: dict[str, str] | None = None,
    feature_labels: dict[str, str] | None = None,
    official_only: bool = True,
    metric_columns: Sequence[str] = MORTALITY_CV_METRIC_COLUMNS,
    digits: int = 3,
    missing: str = "",
) -> pd.DataFrame:
    """Format mortality-CV summaries with public labels and no internal paths."""
    metric_columns = list(metric_columns)
    output_columns = ["model", "feature_set", "fold_count", *metric_columns]
    if summary.empty:
        return pd.DataFrame(columns=output_columns)

    _require_columns(summary, ["model_id", "stage2_experiment_id", "fold_count", "is_official"])
    working = summary.copy()
    if official_only:
        working = working[working["is_official"].astype(bool)].copy()

    formatted = build_mortality_cv_summary_table(
        working,
        metric_columns=metric_columns,
        digits=digits,
        missing=missing,
    )
    model_labels = model_labels or {}
    feature_labels = feature_labels or {}

    rows: list[dict[str, object]] = []
    for row in formatted.to_dict(orient="records"):
        model_id = str(row["model_id"])
        feature_id = str(row["stage2_experiment_id"])
        rows.append(
            {
                "model": model_labels.get(model_id, model_id),
                "feature_set": feature_labels.get(feature_id, feature_id),
                "fold_count": row["fold_count"],
                **{metric: row[metric] for metric in metric_columns},
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def build_public_mortality_cv_rank_table(
    summary: pd.DataFrame,
    *,
    model_labels: dict[str, str] | None = None,
    feature_labels: dict[str, str] | None = None,
    metric: str = "test_auroc",
    higher_is_better: bool = True,
    official_only: bool = True,
    digits: int = 3,
    missing: str = "",
) -> pd.DataFrame:
    """Build a public aggregate ranking table from mortality-CV summaries."""
    output_columns = [
        "rank",
        "model",
        "feature_set",
        "fold_count",
        "metric",
        "mean",
        "sd",
        "mean_sd",
    ]
    if summary.empty:
        return pd.DataFrame(columns=output_columns)

    mean_column = f"{metric}_mean"
    sd_column = f"{metric}_std"
    _require_columns(
        summary,
        [
            "model_id",
            "stage2_experiment_id",
            "fold_count",
            "is_official",
            mean_column,
            sd_column,
        ],
    )

    working = summary.copy()
    if official_only:
        working = working[working["is_official"].astype(bool)].copy()
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    working[mean_column] = pd.to_numeric(working[mean_column], errors="coerce")
    working[sd_column] = pd.to_numeric(working[sd_column], errors="coerce")
    working = working.sort_values(
        [mean_column, "model_id", "stage2_experiment_id"],
        ascending=[not higher_is_better, True, True],
        na_position="last",
    ).reset_index(drop=True)

    model_labels = model_labels or {}
    feature_labels = feature_labels or {}
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(working.to_dict(orient="records"), start=1):
        model_id = str(row["model_id"])
        feature_id = str(row["stage2_experiment_id"])
        mean = float(row[mean_column])
        sd = float(row[sd_column])
        rows.append(
            {
                "rank": int(rank),
                "model": model_labels.get(model_id, model_id),
                "feature_set": feature_labels.get(feature_id, feature_id),
                "fold_count": int(row["fold_count"]),
                "metric": str(metric),
                "mean": mean,
                "sd": sd,
                "mean_sd": format_mean_sd(mean, sd, digits=digits, missing=missing),
            }
        )

    return pd.DataFrame(rows, columns=output_columns)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
