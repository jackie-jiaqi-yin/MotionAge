"""Utilities for participant-level static covariate preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StaticCovariatePreprocessor:
    """Train-split-fitted preprocessing state for static covariates."""

    id_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_fill_values: dict[str, float]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_maps: dict[str, dict[str, int]]


def build_static_covariate_table(
    df: pd.DataFrame,
    *,
    id_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Return a one-row-per-participant static covariate table."""
    requested = [id_column, *numeric_columns, *categorical_columns]
    missing = [column for column in requested if column not in df.columns]
    if missing:
        raise KeyError(f"Missing covariate columns: {missing}")

    table = df[requested].copy()
    duplicated = table[id_column].duplicated(keep=False)
    if duplicated.any():
        dup_count = int(duplicated.sum())
        raise ValueError(
            f"Static covariate table must be one row per participant; found {dup_count} duplicate rows "
            f"for id column '{id_column}'"
        )

    return table.reset_index(drop=True)


def summarize_covariate_missingness(
    df: pd.DataFrame,
    *,
    columns: list[str],
) -> list[dict[str, str | int | float]]:
    """Return aggregate missingness diagnostics for selected covariates."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing covariate columns: {missing_columns}")

    n = int(len(df))
    rows: list[dict[str, str | int | float]] = []
    for column in columns:
        missing = int(df[column].isna().sum())
        rows.append(
            {
                "column": column,
                "n": n,
                "observed": int(n - missing),
                "missing": missing,
                "missing_rate": float(missing / n) if n else 0.0,
            }
        )
    return rows


def fit_static_covariate_preprocessor(
    train_df: pd.DataFrame,
    *,
    id_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> StaticCovariatePreprocessor:
    """Fit train-split preprocessing statistics for static covariates."""
    table = build_static_covariate_table(
        train_df,
        id_column=id_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    numeric_fill_values: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for column in numeric_columns:
        series = pd.to_numeric(table[column], errors="coerce")
        median = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median)
        mean = float(filled.mean()) if len(filled) else 0.0
        std = float(filled.std(ddof=0)) if len(filled) else 1.0
        if std < 1e-8:
            std = 1.0

        numeric_fill_values[column] = median
        numeric_means[column] = mean
        numeric_stds[column] = std

    categorical_maps: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        series = table[column]
        keys = sorted({_categorical_key(value) for value in series.dropna().tolist()})
        categorical_maps[column] = {key: idx + 1 for idx, key in enumerate(keys)}

    return StaticCovariatePreprocessor(
        id_column=id_column,
        numeric_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns),
        numeric_fill_values=numeric_fill_values,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_maps=categorical_maps,
    )


def transform_static_covariates(
    df: pd.DataFrame,
    preprocessor: StaticCovariatePreprocessor,
) -> dict[str, np.ndarray]:
    """Transform a participant-level covariate table into model-ready arrays."""
    table = build_static_covariate_table(
        df,
        id_column=preprocessor.id_column,
        numeric_columns=preprocessor.numeric_columns,
        categorical_columns=preprocessor.categorical_columns,
    )

    numeric_parts: list[np.ndarray] = []
    missing_parts: list[np.ndarray] = []
    for column in preprocessor.numeric_columns:
        series = pd.to_numeric(table[column], errors="coerce")
        missing = series.isna().to_numpy(dtype=np.float32)
        filled = series.fillna(preprocessor.numeric_fill_values[column]).to_numpy(dtype=np.float32)
        standardized = (filled - preprocessor.numeric_means[column]) / preprocessor.numeric_stds[column]
        numeric_parts.append(standardized.astype(np.float32))
        missing_parts.append(missing)

    categorical_parts: list[np.ndarray] = []
    for column in preprocessor.categorical_columns:
        mapping = preprocessor.categorical_maps[column]
        encoded = np.asarray(
            [
                0 if pd.isna(value) else mapping.get(_categorical_key(value), 0)
                for value in table[column].tolist()
            ],
            dtype=np.int64,
        )
        categorical_parts.append(encoded)

    sample_count = int(len(table))
    static_num = _stack_or_empty(numeric_parts, sample_count=sample_count, dtype=np.float32)
    static_num_missing = _stack_or_empty(missing_parts, sample_count=sample_count, dtype=np.float32)
    static_cat = _stack_or_empty(categorical_parts, sample_count=sample_count, dtype=np.int64)

    return {
        preprocessor.id_column: table[preprocessor.id_column].to_numpy(),
        "static_num": static_num,
        "static_num_missing": static_num_missing,
        "static_cat": static_cat,
    }


def _categorical_key(value: Any) -> str:
    """Normalize raw categorical values to stable string keys."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _stack_or_empty(
    values: list[np.ndarray],
    *,
    sample_count: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Stack column vectors into [N, F], returning an empty feature matrix if needed."""
    if not values:
        return np.empty((sample_count, 0), dtype=dtype)

    matrix = np.column_stack(values)
    return matrix.astype(dtype, copy=False)
