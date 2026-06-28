"""Probability-to-age mappings for MotionAge analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class LogisticInverseGroupFit:
    """Fitted logistic-inverse parameters for one sex group."""

    sex_value: Any
    alpha: float
    beta: float
    age_min: float
    age_max: float
    n_participants: int
    n_age_bins: int


@dataclass(frozen=True)
class MotionAgeMapping:
    """Reusable sex-specific logistic-inverse MotionAge mapping."""

    method: str
    fit_partitions: tuple[str, ...]
    representative_probability: str
    clip_eps: float
    weighted_fit: bool
    clamp_output_to_fit_age_range: bool
    require_positive_beta: bool
    age_column: str
    sex_column: str
    probability_column: str
    groups: tuple[LogisticInverseGroupFit, ...]


def fit_motionage_mapping(
    participants: pd.DataFrame,
    *,
    fit_partitions: list[str] | tuple[str, ...],
    representative_probability: str,
    clip_eps: float,
    weighted_fit: bool,
    clamp_output_to_fit_age_range: bool,
    require_positive_beta: bool,
    probability_column: str = "participant_probability",
    age_column: str = "RIDAGEYR",
    sex_column: str = "RIAGENDR",
) -> tuple[MotionAgeMapping, pd.DataFrame]:
    """Fit a sex-specific logistic-inverse mapping from risk probability to age."""
    _validate_required_columns(
        participants,
        required_columns=["split", probability_column, age_column, sex_column],
    )
    normalized_partitions = _normalize_partitions(fit_partitions)
    _validate_clip_eps(clip_eps)

    representative = str(representative_probability).strip().lower()
    if representative not in {"median", "mean"}:
        raise ValueError(f"Unsupported representative_probability '{representative_probability}'.")

    fit_df = participants[participants["split"].isin(normalized_partitions)].copy()
    if fit_df.empty:
        raise ValueError(f"No participants found in fit partitions {normalized_partitions}.")

    fit_df = fit_df.dropna(subset=[probability_column, age_column, sex_column]).copy()
    if fit_df.empty:
        raise ValueError("No complete participants remain after dropping missing mapping inputs.")
    fit_df["age_bin"] = np.rint(fit_df[age_column].to_numpy(dtype=np.float64)).astype(np.int64)

    group_fits: list[LogisticInverseGroupFit] = []
    diagnostics: list[pd.DataFrame] = []
    for sex_value, sex_df in fit_df.groupby(sex_column, sort=True):
        age_diag = _fit_one_group(
            sex_df=sex_df,
            sex_value=sex_value,
            representative_probability=representative,
            clip_eps=clip_eps,
            weighted_fit=weighted_fit,
            require_positive_beta=require_positive_beta,
            probability_column=probability_column,
        )
        group_fits.append(
            LogisticInverseGroupFit(
                sex_value=sex_value,
                alpha=float(age_diag.attrs["alpha"]),
                beta=float(age_diag.attrs["beta"]),
                age_min=float(age_diag["age_bin"].min()),
                age_max=float(age_diag["age_bin"].max()),
                n_participants=int(sex_df.shape[0]),
                n_age_bins=int(age_diag.shape[0]),
            )
        )
        diagnostics.append(age_diag)

    mapping = MotionAgeMapping(
        method="sex_specific_logistic_inverse",
        fit_partitions=tuple(normalized_partitions),
        representative_probability=representative,
        clip_eps=float(clip_eps),
        weighted_fit=bool(weighted_fit),
        clamp_output_to_fit_age_range=bool(clamp_output_to_fit_age_range),
        require_positive_beta=bool(require_positive_beta),
        age_column=age_column,
        sex_column=sex_column,
        probability_column=probability_column,
        groups=tuple(group_fits),
    )
    diagnostic_df = pd.concat(diagnostics, ignore_index=True)
    return mapping, diagnostic_df


def apply_motionage_mapping(
    participants: pd.DataFrame,
    mapping: MotionAgeMapping,
    *,
    motionage_column: str = "MotionAge",
    accel_column: str = "MotionAgeAccel",
) -> pd.DataFrame:
    """Apply a fitted mapping to participant-level probabilities."""
    _validate_required_columns(
        participants,
        required_columns=[mapping.probability_column, mapping.age_column, mapping.sex_column],
    )
    _validate_clip_eps(mapping.clip_eps)

    result = participants.copy()
    result[motionage_column] = np.nan

    group_lookup = {fit.sex_value: fit for fit in mapping.groups}
    clipped = np.clip(
        result[mapping.probability_column].to_numpy(dtype=np.float64),
        mapping.clip_eps,
        1.0 - mapping.clip_eps,
    )
    participant_logits = np.log(clipped / (1.0 - clipped))

    motionage_idx = result.columns.get_loc(motionage_column)
    for row_idx, sex_value in enumerate(result[mapping.sex_column].tolist()):
        group_fit = group_lookup.get(sex_value)
        if group_fit is None:
            continue
        motion_age = (participant_logits[row_idx] - group_fit.alpha) / group_fit.beta
        if mapping.clamp_output_to_fit_age_range:
            motion_age = float(np.clip(motion_age, group_fit.age_min, group_fit.age_max))
        result.iat[row_idx, motionage_idx] = motion_age

    result[accel_column] = result[motionage_column] - result[mapping.age_column].astype(float)
    return result


def mapping_to_jsonable(mapping: MotionAgeMapping) -> dict[str, Any]:
    """Convert a fitted mapping to a JSON-friendly payload."""
    return {
        "method": mapping.method,
        "fit_partitions": list(mapping.fit_partitions),
        "representative_probability": mapping.representative_probability,
        "clip_eps": float(mapping.clip_eps),
        "weighted_fit": bool(mapping.weighted_fit),
        "clamp_output_to_fit_age_range": bool(mapping.clamp_output_to_fit_age_range),
        "require_positive_beta": bool(mapping.require_positive_beta),
        "age_column": mapping.age_column,
        "sex_column": mapping.sex_column,
        "probability_column": mapping.probability_column,
        "groups": [
            {
                "sex_value": _jsonify_value(group.sex_value),
                "alpha": float(group.alpha),
                "beta": float(group.beta),
                "age_min": float(group.age_min),
                "age_max": float(group.age_max),
                "n_participants": int(group.n_participants),
                "n_age_bins": int(group.n_age_bins),
            }
            for group in mapping.groups
        ],
    }


def _fit_one_group(
    *,
    sex_df: pd.DataFrame,
    sex_value: Any,
    representative_probability: str,
    clip_eps: float,
    weighted_fit: bool,
    require_positive_beta: bool,
    probability_column: str,
) -> pd.DataFrame:
    grouped = (
        sex_df.groupby("age_bin", as_index=False)
        .agg(
            n_participants=(probability_column, "size"),
            representative_probability=(
                probability_column,
                "median" if representative_probability == "median" else "mean",
            ),
        )
        .sort_values("age_bin")
        .reset_index(drop=True)
    )
    if grouped.shape[0] < 2:
        raise ValueError(f"Need at least two age bins to fit MotionAge mapping for sex={sex_value}.")

    grouped["split"] = np.nan
    grouped["sex_value"] = sex_value
    grouped["clipped_probability"] = np.clip(
        grouped["representative_probability"].to_numpy(dtype=np.float64),
        clip_eps,
        1.0 - clip_eps,
    )
    grouped["logit_probability"] = np.log(
        grouped["clipped_probability"] / (1.0 - grouped["clipped_probability"])
    )

    x = grouped[["age_bin"]].to_numpy(dtype=np.float64)
    y = grouped["logit_probability"].to_numpy(dtype=np.float64)
    weights = grouped["n_participants"].to_numpy(dtype=np.float64) if weighted_fit else None
    model = LinearRegression()
    if weights is None:
        model.fit(x, y)
    else:
        model.fit(x, y, sample_weight=weights)

    alpha = float(model.intercept_)
    beta = float(model.coef_[0])
    if require_positive_beta and beta <= 0.0:
        raise ValueError(
            f"MotionAge mapping requires a positive beta, but fitted beta={beta:.6f} for sex={sex_value}."
        )

    fitted_logits = alpha + beta * grouped["age_bin"].to_numpy(dtype=np.float64)
    grouped["fitted_logit_probability"] = fitted_logits
    grouped["fitted_probability"] = 1.0 / (1.0 + np.exp(-fitted_logits))
    grouped.attrs["alpha"] = alpha
    grouped.attrs["beta"] = beta
    return grouped


def _normalize_partitions(partitions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(partition).strip().lower() for partition in partitions if str(partition).strip())
    if not normalized:
        raise ValueError("fit_partitions must contain at least one partition name.")
    return normalized


def _validate_required_columns(df: pd.DataFrame, *, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _validate_clip_eps(clip_eps: float) -> None:
    if not (0.0 < float(clip_eps) < 0.5):
        raise ValueError(f"clip_eps must be in (0, 0.5), got {clip_eps}.")


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
