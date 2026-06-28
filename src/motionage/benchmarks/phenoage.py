"""PhenoAge benchmark formula and missing-data helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

PHENOAGE_INPUT_COLUMNS: tuple[str, ...] = (
    "RIDAGEYR",
    "LBDSALSI",
    "LBDSCRSI",
    "LBDSGLSI",
    "log_crp",
    "LBXLYPCT",
    "MCV",
    "LBXRDW",
    "LBXSAPSI",
    "LBXWBCSI",
)
PHENOAGE_BIOMARKER_COLUMNS: tuple[str, ...] = PHENOAGE_INPUT_COLUMNS[1:]
PHENOAGE_IMPUTATION_STRATEGIES: tuple[str, ...] = ("median", "age_sex_median")

_INTERCEPT = -19.9067
_GAMMA = 0.0076927
_W_ALBUMIN = -0.0336
_W_CREATININE = 0.0095
_W_GLUCOSE = 0.1953
_W_LOG_CRP = 0.0954
_W_LYMPH_PCT = -0.0120
_W_MCV = 0.0268
_W_RDW = 0.3306
_W_ALP = 0.0019
_W_WBC = 0.0554
_W_AGE = 0.0804


@dataclass(frozen=True)
class PhenoAgeImputer:
    """Train-fitted imputer for PhenoAge formula inputs."""

    strategy: str
    global_medians: pd.Series
    sex_medians: pd.DataFrame | None = None
    strata_medians: pd.DataFrame | None = None
    input_columns: tuple[str, ...] = PHENOAGE_INPUT_COLUMNS
    biomarker_columns: tuple[str, ...] = PHENOAGE_BIOMARKER_COLUMNS
    age_column: str = "RIDAGEYR"
    sex_column: str = "RIAGENDR"
    age_band_width: int = 10

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply train-fitted imputations to a new frame."""
        _require_columns(frame, self.input_columns)
        filled = _coerce_numeric_columns(frame.copy(), self.input_columns)

        if self.strategy == "median":
            filled.loc[:, self.input_columns] = filled.loc[:, self.input_columns].fillna(self.global_medians)
            return filled

        if self.strategy != "age_sex_median":
            raise ValueError(f"Unknown PhenoAge imputation strategy: {self.strategy}")
        _require_columns(filled, [self.sex_column, self.age_column])
        if self.sex_medians is None or self.strata_medians is None:
            raise ValueError("Age/sex median imputation requires fitted sex and age-sex median tables.")

        keyed = _add_age_sex_imputation_keys(
            filled,
            sex_column=self.sex_column,
            age_column=self.age_column,
            age_band_width=self.age_band_width,
        )
        keyed.loc[:, self.age_column] = keyed[self.age_column].fillna(self.global_medians.get(self.age_column, np.nan))
        for idx, row in keyed.iterrows():
            sex = row["_impute_sex"]
            age_band = row["_impute_age_band"]
            for column in self.biomarker_columns:
                if pd.notna(row[column]):
                    continue
                keyed.at[idx, column] = _lookup_imputation_value(
                    column,
                    sex=sex,
                    age_band=age_band,
                    global_medians=self.global_medians,
                    sex_medians=self.sex_medians,
                    strata_medians=self.strata_medians,
                )
        return keyed.drop(columns=["_impute_sex", "_impute_age_band"])


def compute_phenoage(frame: pd.DataFrame) -> pd.Series:
    """Compute PhenoAge from chronological age and biomarker inputs."""
    _require_columns(frame, PHENOAGE_INPUT_COLUMNS)
    values = _coerce_numeric_columns(frame.copy(), PHENOAGE_INPUT_COLUMNS)
    xb = (
        _INTERCEPT
        + _W_ALBUMIN * values["LBDSALSI"]
        + _W_CREATININE * values["LBDSCRSI"]
        + _W_GLUCOSE * values["LBDSGLSI"]
        + _W_LOG_CRP * values["log_crp"]
        + _W_LYMPH_PCT * values["LBXLYPCT"]
        + _W_MCV * values["MCV"]
        + _W_RDW * values["LBXRDW"]
        + _W_ALP * values["LBXSAPSI"]
        + _W_WBC * values["LBXWBCSI"]
        + _W_AGE * values["RIDAGEYR"]
    )
    xb_clipped = xb.clip(upper=700)
    mortality_score = 1 - np.exp(-np.exp(xb_clipped) * (np.exp(_GAMMA * 120) - 1) / _GAMMA)
    mortality_score = mortality_score.clip(1e-10, 1 - 1e-10)
    inner = (-0.00553 * np.log(1 - mortality_score)).clip(lower=1e-300)
    return pd.Series(141.50225 + np.log(inner) / 0.090165, index=frame.index, name="PhenoAge")


def summarize_phenoage_missingness(
    participants: pd.DataFrame,
    *,
    age_threshold: int = 40,
    input_columns: Sequence[str] = PHENOAGE_INPUT_COLUMNS,
) -> pd.DataFrame:
    """Summarize pre-imputation missingness for PhenoAge inputs."""
    input_columns = tuple(input_columns)
    _require_columns(participants, input_columns)
    if "RIDAGEYR" not in input_columns:
        raise ValueError("input_columns must include RIDAGEYR for age-threshold summaries.")

    frame = _coerce_numeric_columns(participants.copy(), input_columns)
    age_mask = frame["RIDAGEYR"] >= int(age_threshold)
    rows: list[dict[str, float | int | str]] = []
    for column in input_columns:
        overall = frame[column]
        age_subset = frame.loc[age_mask, column]
        rows.append(
            {
                "column": column,
                "input_role": "chronological_age" if column == "RIDAGEYR" else "biomarker",
                "n_total": int(overall.shape[0]),
                "n_missing": int(overall.isna().sum()),
                "missing_rate": _missing_rate(overall),
                "n_age_ge_threshold": int(age_subset.shape[0]),
                "n_missing_age_ge_threshold": int(age_subset.isna().sum()),
                "missing_rate_age_ge_threshold": _missing_rate(age_subset),
            }
        )
    return pd.DataFrame(rows)


def fit_phenoage_imputer(
    train_frame: pd.DataFrame,
    *,
    strategy: str = "median",
    input_columns: Sequence[str] = PHENOAGE_INPUT_COLUMNS,
    biomarker_columns: Sequence[str] = PHENOAGE_BIOMARKER_COLUMNS,
    age_column: str = "RIDAGEYR",
    sex_column: str = "RIAGENDR",
    age_band_width: int = 10,
) -> PhenoAgeImputer:
    """Fit a PhenoAge input imputer using training data only."""
    if strategy not in PHENOAGE_IMPUTATION_STRATEGIES:
        raise ValueError(
            f"Unknown PhenoAge imputation strategy '{strategy}'. "
            f"Expected one of {PHENOAGE_IMPUTATION_STRATEGIES}."
        )
    input_columns = tuple(input_columns)
    biomarker_columns = tuple(biomarker_columns)
    _require_columns(train_frame, input_columns)

    train = _coerce_numeric_columns(train_frame.copy(), input_columns)
    global_medians = train.loc[:, input_columns].median(numeric_only=True)
    if strategy == "median":
        return PhenoAgeImputer(
            strategy=strategy,
            global_medians=global_medians,
            input_columns=input_columns,
            biomarker_columns=biomarker_columns,
            age_column=age_column,
            sex_column=sex_column,
            age_band_width=int(age_band_width),
        )

    _require_columns(train_frame, [sex_column, age_column])
    keyed = _add_age_sex_imputation_keys(
        train,
        sex_column=sex_column,
        age_column=age_column,
        age_band_width=int(age_band_width),
    )
    sex_medians = keyed.groupby("_impute_sex", dropna=False)[list(biomarker_columns)].median()
    strata_medians = keyed.groupby(["_impute_sex", "_impute_age_band"], dropna=False)[list(biomarker_columns)].median()
    return PhenoAgeImputer(
        strategy=strategy,
        global_medians=global_medians,
        sex_medians=sex_medians,
        strata_medians=strata_medians,
        input_columns=input_columns,
        biomarker_columns=biomarker_columns,
        age_column=age_column,
        sex_column=sex_column,
        age_band_width=int(age_band_width),
    )


def select_complete_phenoage_cases(
    frame: pd.DataFrame,
    *,
    input_columns: Sequence[str] = PHENOAGE_INPUT_COLUMNS,
) -> pd.DataFrame:
    """Return rows with complete PhenoAge formula inputs."""
    input_columns = tuple(input_columns)
    _require_columns(frame, input_columns)
    complete = _coerce_numeric_columns(frame.copy(), input_columns).dropna(subset=list(input_columns))
    return complete.reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    coerced = frame.copy()
    for column in columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def _missing_rate(series: pd.Series) -> float:
    if series.shape[0] == 0:
        return float("nan")
    return float(series.isna().mean())


def _add_age_sex_imputation_keys(
    frame: pd.DataFrame,
    *,
    sex_column: str,
    age_column: str,
    age_band_width: int,
) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["_impute_sex"] = pd.to_numeric(keyed[sex_column], errors="coerce").round()
    age_values = pd.to_numeric(keyed[age_column], errors="coerce")
    keyed["_impute_age_band"] = (age_values // int(age_band_width) * int(age_band_width)).astype("Int64")
    return keyed


def _lookup_imputation_value(
    column: str,
    *,
    sex: float,
    age_band: int,
    global_medians: pd.Series,
    sex_medians: pd.DataFrame,
    strata_medians: pd.DataFrame,
) -> float:
    if pd.notna(sex) and pd.notna(age_band):
        key = (sex, age_band)
        if key in strata_medians.index:
            value = strata_medians.at[key, column]
            if pd.notna(value):
                return float(value)
    if pd.notna(sex) and sex in sex_medians.index:
        value = sex_medians.at[sex, column]
        if pd.notna(value):
            return float(value)
    value = global_medians.get(column, np.nan)
    return float(value) if pd.notna(value) else float("nan")
