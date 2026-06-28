"""LLM-age benchmark utilities for optional external prediction artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from motionage.evaluation.metrics import binary_probability_metrics

LLM_AGE_COLUMN = "LLMOverallAge"
LLM_AGE_ACCEL_COLUMN = "LLMOverallAgeAccel"
FOLLOWUP_MONTH_COLUMN = "permth_int"
FOLLOWUP_MONTHS = 60

LLM_AGE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "chronological_age_baseline": ("RIDAGEYR", "sex_code"),
    "llm_age_raw": (LLM_AGE_COLUMN, "sex_code"),
    "llm_age_accel": (LLM_AGE_ACCEL_COLUMN, "RIDAGEYR", "sex_code"),
}


def load_llm_age_participants(csv_path: Path | str) -> pd.DataFrame:
    """Load an external LLM-age prediction table into benchmark-ready columns."""
    participants = pd.read_csv(csv_path, low_memory=False)
    required = {"SEQN", "mortstat", FOLLOWUP_MONTH_COLUMN, "RIDAGEYR", "RIAGENDR", "overall_age"}
    missing = sorted(required.difference(participants.columns))
    if missing:
        raise KeyError(f"LLM-age CSV missing required columns: {missing}")

    if "llm_ok" in participants.columns:
        ok_mask = _normalize_bool_series(participants["llm_ok"])
        participants = participants.loc[ok_mask].copy()
    else:
        participants = participants.copy()

    participants["SEQN"] = _normalize_seqn(participants["SEQN"])
    participants = _coerce_numeric_columns(
        participants,
        ["mortstat", FOLLOWUP_MONTH_COLUMN, "RIDAGEYR", "RIAGENDR", "overall_age"],
    )
    participants["mortstat"] = (
        (participants["mortstat"] == 1) & (participants[FOLLOWUP_MONTH_COLUMN] <= FOLLOWUP_MONTHS)
    ).astype("int64")
    participants = participants.rename(columns={"overall_age": LLM_AGE_COLUMN})
    return (
        participants.loc[:, ["SEQN", "mortstat", "RIDAGEYR", "RIAGENDR", LLM_AGE_COLUMN]]
        .dropna()
        .drop_duplicates(subset=["SEQN"], keep="first")
        .reset_index(drop=True)
    )


def run_fold_benchmark(
    participants: pd.DataFrame,
    *,
    train_ids: pd.Series | pd.DataFrame | Iterable[int],
    test_ids: pd.Series | pd.DataFrame | Iterable[int],
    age_threshold: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one fold of benchmark logistic models and return predictions plus metrics."""
    required = {"SEQN", "mortstat", "RIDAGEYR", "RIAGENDR", LLM_AGE_COLUMN}
    missing = sorted(required.difference(participants.columns))
    if missing:
        raise KeyError(f"Participants missing required columns: {missing}")

    participant_table = participants.copy()
    participant_table["SEQN"] = _normalize_seqn(participant_table["SEQN"])
    participant_table = _coerce_numeric_columns(
        participant_table,
        ["mortstat", "RIDAGEYR", "RIAGENDR", LLM_AGE_COLUMN],
    )
    participant_table[LLM_AGE_ACCEL_COLUMN] = participant_table[LLM_AGE_COLUMN] - participant_table["RIDAGEYR"]
    participant_table["sex_code"] = _encode_sex(participant_table["RIAGENDR"])

    train_series = _normalize_id_series(train_ids)
    test_series = _normalize_id_series(test_ids)
    model_columns = ["mortstat", "RIDAGEYR", "RIAGENDR", LLM_AGE_COLUMN, LLM_AGE_ACCEL_COLUMN, "sex_code"]
    train_df = (
        participant_table[participant_table["SEQN"].isin(train_series.tolist())]
        .dropna(subset=model_columns)
        .copy()
    )
    test_df = (
        participant_table[participant_table["SEQN"].isin(test_series.tolist())]
        .dropna(subset=model_columns)
        .copy()
    )
    if train_df.empty or test_df.empty:
        raise ValueError("Fold benchmark requires non-empty train and test subsets.")
    if train_df["mortstat"].nunique() < 2:
        raise ValueError("Fold benchmark needs both mortality classes in training data.")

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    for feature_set, columns in LLM_AGE_FEATURE_SETS.items():
        estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
        estimator.fit(train_df[list(columns)], train_df["mortstat"])
        train_probability = estimator.predict_proba(train_df[list(columns)])[:, 1]
        test_probability = estimator.predict_proba(test_df[list(columns)])[:, 1]

        feature_predictions = pd.concat(
            [
                _prediction_frame(
                    train_df,
                    split="train",
                    feature_set=feature_set,
                    probability=train_probability,
                ),
                _prediction_frame(
                    test_df,
                    split="test",
                    feature_set=feature_set,
                    probability=test_probability,
                ),
            ],
            ignore_index=True,
        )
        prediction_rows.append(feature_predictions)

        for population, minimum_age in (("overall", None), ("age_ge_40", int(age_threshold))):
            subset = feature_predictions[feature_predictions["split"] == "test"].copy()
            if minimum_age is not None:
                subset = subset[subset["RIDAGEYR"] >= minimum_age].copy()
            metrics = _population_metrics(subset)
            metric_rows.append(
                {
                    "feature_set": feature_set,
                    "population": population,
                    "train_participants": int(train_df.shape[0]),
                    "test_participants": int(subset.shape[0]),
                    "test_auroc": float(metrics["auroc"]),
                    "test_auprc": float(metrics["auprc"]),
                    "test_logloss": float(metrics["logloss"]),
                    "test_brier": float(metrics["brier"]),
                    "test_positive_rate": float(metrics["positive_rate"]),
                }
            )

    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)


def discover_cv_folds(fold_root: Path | str, *, fold_limit: int | None = None) -> list[Path]:
    """Return `fold_<index>` directories that contain train/test ID CSVs."""
    root = Path(fold_root)
    fold_dirs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("fold_"):
            continue
        if not (path / "train_ids.csv").exists() or not (path / "test_ids.csv").exists():
            continue
        try:
            int(path.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        fold_dirs.append(path)

    fold_dirs.sort(key=lambda path: int(path.name.split("_", maxsplit=1)[1]))
    if fold_limit is not None:
        return fold_dirs[: int(fold_limit)]
    return fold_dirs


def run_llm_age_benchmark_cv(
    participants: pd.DataFrame,
    *,
    fold_root: Path | str,
    output_root: Path | str,
    age_threshold: int = 40,
    fold_limit: int | None = None,
) -> pd.DataFrame:
    """Run LLM-age benchmark folds and write local replay artifacts."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    for fold_dir in discover_cv_folds(fold_root, fold_limit=fold_limit):
        train_ids = pd.read_csv(fold_dir / "train_ids.csv")["SEQN"]
        test_ids = pd.read_csv(fold_dir / "test_ids.csv")["SEQN"]
        predictions, metrics = run_fold_benchmark(
            participants,
            train_ids=train_ids,
            test_ids=test_ids,
            age_threshold=age_threshold,
        )

        fold_output_dir = output_dir / fold_dir.name
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(fold_output_dir / "participant_predictions.csv", index=False)
        metrics.to_csv(fold_output_dir / "metrics.csv", index=False)
        rows.append(metrics.assign(fold=fold_dir.name))

    if not rows:
        raise ValueError(f"No fold directories discovered under {fold_root}.")

    all_metrics = pd.concat(rows, ignore_index=True)
    summary = (
        all_metrics.groupby(["feature_set", "population"], as_index=False)
        .agg(
            fold_count=("fold", "nunique"),
            test_auroc_mean=("test_auroc", "mean"),
            test_auprc_mean=("test_auprc", "mean"),
            test_logloss_mean=("test_logloss", "mean"),
            test_brier_mean=("test_brier", "mean"),
            test_positive_rate_mean=("test_positive_rate", "mean"),
            test_participants_mean=("test_participants", "mean"),
        )
        .sort_values(["population", "feature_set"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _population_metrics(subset: pd.DataFrame) -> dict[str, float]:
    if subset.empty:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "positive_rate": float("nan"),
        }
    return binary_probability_metrics(subset["target"].to_numpy(), subset["probability"].to_numpy())


def _normalize_seqn(series: pd.Series) -> pd.Series:
    normalized = pd.to_numeric(series, errors="coerce")
    if normalized.isna().any():
        raise ValueError("SEQN contains non-numeric values.")
    return normalized.astype("int64")


def _normalize_id_series(ids: pd.Series | pd.DataFrame | Iterable[int]) -> pd.Series:
    if isinstance(ids, pd.DataFrame):
        if "SEQN" not in ids.columns:
            raise KeyError("ID DataFrame must include SEQN.")
        return _normalize_seqn(ids["SEQN"])
    if isinstance(ids, pd.Series):
        return _normalize_seqn(ids)
    return _normalize_seqn(pd.Series(list(ids), name="SEQN"))


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    coerced = frame.copy()
    for column in columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def _encode_sex(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").round()
    labels = numeric.map({1.0: "male", 2.0: "female"}).fillna("missing")
    encoded = pd.Categorical(labels, categories=["male", "female", "missing"]).codes
    return pd.Series(encoded.astype("int64"), index=series.index, name="sex_code")


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    truthy = text.isin({"true", "t", "yes", "y"})
    falsy = text.isin({"false", "f", "no", "n"})
    if numeric.notna().any():
        truthy = truthy | (numeric == 1)
        falsy = falsy | (numeric == 0)

    resolved = pd.Series(np.where(truthy, True, np.where(falsy, False, np.nan)), index=series.index)
    if resolved.isna().any():
        raise ValueError("llm_ok contains unsupported values.")
    return resolved.astype(bool)


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    feature_set: str,
    probability: np.ndarray,
) -> pd.DataFrame:
    columns = ["SEQN", "RIDAGEYR", "RIAGENDR", LLM_AGE_COLUMN, LLM_AGE_ACCEL_COLUMN, "mortstat"]
    return frame.loc[:, columns].assign(
        split=split,
        target=frame["mortstat"].astype(int).to_numpy(),
        feature_set=feature_set,
        probability=probability,
    )[
        [
            "SEQN",
            "split",
            "target",
            "feature_set",
            "probability",
            "RIDAGEYR",
            "RIAGENDR",
            LLM_AGE_COLUMN,
            LLM_AGE_ACCEL_COLUMN,
        ]
    ]
