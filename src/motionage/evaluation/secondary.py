"""Aggregate second-stage evaluation helpers for MotionAge representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression

from motionage.evaluation.metrics import binary_probability_metrics


def evaluate_secondary_feature_sets(
    participants: pd.DataFrame,
    *,
    feature_sets: Sequence[Mapping[str, object]],
    fit_partitions: Sequence[str],
    target_column: str,
    split_column: str = "split",
    baseline_feature_set: str | None = None,
) -> list[dict[str, float | int | str]]:
    """Fit public second-stage logistic models and return aggregate metric rows."""
    _validate_required_columns(participants, [split_column, target_column])
    normalized_fit_partitions = {str(partition) for partition in fit_partitions}
    if not normalized_fit_partitions:
        raise ValueError("fit_partitions must include at least one split label.")

    train_df = participants[participants[split_column].astype(str).isin(normalized_fit_partitions)].copy()
    test_df = participants[participants[split_column].astype(str) == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Secondary evaluation requires non-empty fit and test partitions.")

    rows: list[dict[str, float | int | str]] = []
    for feature_set in feature_sets:
        if not bool(feature_set.get("enabled", True)):
            continue
        feature_name = str(feature_set["name"])
        feature_columns = [str(column) for column in feature_set.get("columns", [])]
        if not feature_columns:
            raise ValueError(f"Secondary feature set {feature_name!r} must include at least one column.")
        ordered_feature_columns = list(dict.fromkeys(feature_columns))
        required_columns = [target_column, *ordered_feature_columns]
        _validate_required_columns(train_df, required_columns)
        _validate_required_columns(test_df, required_columns)

        train_subset = train_df[required_columns].dropna().copy()
        test_subset = test_df[required_columns].dropna().copy()
        if train_subset.empty or test_subset.empty:
            raise ValueError(f"Secondary feature set {feature_name!r} produced empty train or test rows.")
        if train_subset[target_column].nunique() < 2:
            raise ValueError(f"Secondary feature set {feature_name!r} needs both classes in fit partitions.")

        estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
        estimator.fit(train_subset[ordered_feature_columns], train_subset[target_column])
        train_probability = estimator.predict_proba(train_subset[ordered_feature_columns])[:, 1]
        test_probability = estimator.predict_proba(test_subset[ordered_feature_columns])[:, 1]

        train_metrics = binary_probability_metrics(train_subset[target_column].to_numpy(), train_probability)
        test_metrics = binary_probability_metrics(test_subset[target_column].to_numpy(), test_probability)
        rows.append(
            {
                "feature_set": feature_name,
                "feature_columns": ",".join(ordered_feature_columns),
                "train_n": int(train_metrics["n"]),
                "train_events": int(train_metrics["events"]),
                "test_n": int(test_metrics["n"]),
                "test_events": int(test_metrics["events"]),
                "train_auroc": float(train_metrics["auroc"]),
                "train_auprc": float(train_metrics["auprc"]),
                "test_auroc": float(test_metrics["auroc"]),
                "test_auprc": float(test_metrics["auprc"]),
                "test_logloss": float(test_metrics["logloss"]),
                "test_brier": float(test_metrics["brier"]),
                "test_event_rate": float(test_metrics["event_rate"]),
            }
        )

    if not rows:
        raise ValueError("At least one enabled secondary feature set is required.")
    _add_baseline_delta(rows, baseline_feature_set)
    return rows


def _add_baseline_delta(
    rows: list[dict[str, float | int | str]],
    baseline_feature_set: str | None,
) -> None:
    if baseline_feature_set is None:
        return
    baseline_rows = [row for row in rows if row["feature_set"] == baseline_feature_set]
    if not baseline_rows:
        raise ValueError(f"Baseline feature set {baseline_feature_set!r} was not evaluated.")
    baseline_auroc = float(baseline_rows[0]["test_auroc"])
    delta_field = f"delta_vs_{baseline_feature_set}"
    for row in rows:
        row[delta_field] = float(row["test_auroc"]) - baseline_auroc


def _validate_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required secondary evaluation columns: {missing}.")
