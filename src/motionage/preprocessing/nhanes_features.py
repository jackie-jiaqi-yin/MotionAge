"""Canonical NHANES covariate feature definitions for MotionAge modeling.

This module separates two concepts:

- `level`:
  the feature bundle used by the paper modeling configs.
  - level 1: low-dimensional core covariates for the first model
  - level 2: questionnaire / disease-history features
  - level 3: laboratory, CBC, and dietary features

- `kind`:
  the modeling-side encoding choice used by preprocessing and the model.
  It answers: how should this variable be represented in code?
  - `numeric`: impute + standardize + missingness indicator
  - `categorical`: integer encode + unknown/missing bucket

`level` is the public feature-bundle grouping used by this repository.
`kind` is assigned in code for preprocessing and model input construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NHANESFeatureSpec:
    """Single covariate definition sourced from NHANES variable definitions."""

    name: str
    kind: str
    level: int
    definition: str
    comment: str | None = None
    domain: str | None = None
    file_location: str | None = None


ID_COLUMN = "SEQN"
TARGET_COLUMN = "RIDAGEYR"
EXCLUDED_COLUMNS = (TARGET_COLUMN,)
FEATURE_LEVEL_DESCRIPTIONS = {
    1: "Core demographic and low-dimensional clinical covariates",
    2: "Questionnaire and disease-history covariates",
    3: "Laboratory, CBC, and dietary covariates",
}

NHANES_FEATURE_SPECS: list[NHANESFeatureSpec] = [
    NHANESFeatureSpec(
        name="INDHHINC",
        kind="categorical",
        level=1,
        definition="Annual Household Income",
        comment=None,
        domain=None,
        file_location="Demographics",
    ),
    NHANESFeatureSpec(
        name="RIAGENDR",
        kind="categorical",
        level=1,
        definition="Gender",
        comment=None,
        domain=None,
        file_location="Demographics",
    ),
    NHANESFeatureSpec(
        name="BMXBMI",
        kind="numeric",
        level=1,
        definition="BMI",
        comment=None,
        domain=None,
        file_location="Body Measures",
    ),
    NHANESFeatureSpec(
        name="BMXWAIST",
        kind="numeric",
        level=1,
        definition="Waist Circumference (cm)",
        comment=None,
        domain=None,
        file_location="Body Measures",
    ),
    NHANESFeatureSpec(
        name="BPXDI",
        kind="numeric",
        level=1,
        definition="Diastolic Blood Pressure (mmHg)",
        comment="take average of BPXDI1-4",
        domain=None,
        file_location="Blood Pressure",
    ),
    NHANESFeatureSpec(
        name="BPXSY",
        kind="numeric",
        level=1,
        definition="Systolic Blood Pressure (mmHg)",
        comment="take average of BPXSY1-4",
        domain=None,
        file_location="Blood Pressure",
    ),
    NHANESFeatureSpec(
        name="DIQ010",
        kind="categorical",
        level=2,
        definition="Doctor told you have diabetes",
        comment=None,
        domain=None,
        file_location="Diabetes",
    ),
    NHANESFeatureSpec(
        name="MCQ010",
        kind="categorical",
        level=2,
        definition="Ever been told you have asthma",
        comment=None,
        domain=None,
        file_location="Medical Conditions",
    ),
    NHANESFeatureSpec(
        name="PAD020",
        kind="categorical",
        level=2,
        definition="Walked or bicycled over past 30 days",
        comment=None,
        domain=None,
        file_location="Physical activity",
    ),
    NHANESFeatureSpec(
        name="PAD200",
        kind="categorical",
        level=2,
        definition="Vigorous activity over past 30 days",
        comment=None,
        domain=None,
        file_location="Physical activity",
    ),
    NHANESFeatureSpec(
        name="PAD320",
        kind="categorical",
        level=2,
        definition="Moderate activity over past 30 days",
        comment=None,
        domain=None,
        file_location="Physical activity",
    ),
    NHANESFeatureSpec(
        name="DRXTACAR",
        kind="numeric",
        level=3,
        definition="Alpha-carotene (mcg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTATOC",
        kind="numeric",
        level=3,
        definition="Vitamin E as alpha-tocopherol (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTBCAR",
        kind="numeric",
        level=3,
        definition="Beta-carotene (mcg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTCALC",
        kind="numeric",
        level=3,
        definition="Calcium (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTCARB",
        kind="numeric",
        level=3,
        definition="Carbohydrate (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTCHOL",
        kind="numeric",
        level=3,
        definition="Cholesterol (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTCOPP",
        kind="numeric",
        level=3,
        definition="Copper (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTFIBE",
        kind="numeric",
        level=3,
        definition="Dietary fiber (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTFOLA",
        kind="numeric",
        level=3,
        definition="Total Folate (mcg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTIRON",
        kind="numeric",
        level=3,
        definition="Iron (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTKCAL",
        kind="numeric",
        level=3,
        definition="Energy (kcal)",
        comment=None,
        domain="Lifestyles",
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTMAGN",
        kind="numeric",
        level=3,
        definition="Magnesium (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTMFAT",
        kind="numeric",
        level=3,
        definition="Total monounsaturated fatty acids (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTNIAC",
        kind="numeric",
        level=3,
        definition="Niacin (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTPFAT",
        kind="numeric",
        level=3,
        definition="Total polyunsaturated fatty acids (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTPHOS",
        kind="numeric",
        level=3,
        definition="Phosphorus (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTPOTA",
        kind="numeric",
        level=3,
        definition="Potassium (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTPROT",
        kind="numeric",
        level=3,
        definition="Protein (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTSFAT",
        kind="numeric",
        level=3,
        definition="Total saturated fatty acids (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTTFAT",
        kind="numeric",
        level=3,
        definition="Total fat (gm)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTVARA",
        kind="numeric",
        level=3,
        definition="Vitamin A as retinolactivity equivalents(mcg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTVB1",
        kind="numeric",
        level=3,
        definition="Thiamin (Vitamin B1) (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTVB12",
        kind="numeric",
        level=3,
        definition="Vitamin B12 (mcg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTVB2",
        kind="numeric",
        level=3,
        definition="Riboflavin (Vitamin B2) (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTVB6",
        kind="numeric",
        level=3,
        definition="Vitamin B6 (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTVC",
        kind="numeric",
        level=3,
        definition="Vitamin C (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="DRXTZINC",
        kind="numeric",
        level=3,
        definition="Zinc (mg)",
        comment=None,
        domain=None,
        file_location="Dietary Interview - Total Nutrient Intakes, First Day &  Second Day",
    ),
    NHANESFeatureSpec(
        name="LBDBANO",
        kind="numeric",
        level=3,
        definition="Basophils number",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBDEONO",
        kind="numeric",
        level=3,
        definition="Eosinophils number",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBDLYMNO",
        kind="numeric",
        level=3,
        definition="Lymphocyte number",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBDMONO",
        kind="numeric",
        level=3,
        definition="Monocyte number",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBDNENO",
        kind="numeric",
        level=3,
        definition="Segmented neutrophils number",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBDSALSI",
        kind="numeric",
        level=3,
        definition="Albumin (g/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSBUSI",
        kind="numeric",
        level=3,
        definition="Blood urea nitrogen (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSCASI",
        kind="numeric",
        level=3,
        definition="Calcium, total (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSCRSI",
        kind="numeric",
        level=3,
        definition="Creatinine (umol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSGBSI",
        kind="numeric",
        level=3,
        definition="Globulin (g/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSGLSI",
        kind="numeric",
        level=3,
        definition="Glucose (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSIRSI",
        kind="numeric",
        level=3,
        definition="Iron (umol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSPHSI",
        kind="numeric",
        level=3,
        definition="Phosphorus (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSTBSI",
        kind="numeric",
        level=3,
        definition="Bilirubin, total (umol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSTPSI",
        kind="numeric",
        level=3,
        definition="Protein, total (g/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSTRSI",
        kind="numeric",
        level=3,
        definition="Triglycerides (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBDSUASI",
        kind="numeric",
        level=3,
        definition="Uric acid (umol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXBAPCT",
        kind="numeric",
        level=3,
        definition="Basophils percent (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXCRP",
        kind="numeric",
        level=3,
        definition="C-reactive protein(mg/dL)",
        comment=None,
        domain=None,
        file_location="C-Reactive Protein (CRP)",
    ),
    NHANESFeatureSpec(
        name="LBXEOPCT",
        kind="numeric",
        level=3,
        definition="Eosinophils percent (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXGH",
        kind="numeric",
        level=3,
        definition="Glycohemoglobin (%)",
        comment=None,
        domain="Laboratory test",
        file_location="Glycohemoglobin",
    ),
    NHANESFeatureSpec(
        name="LBXHCT",
        kind="numeric",
        level=3,
        definition="Hematocrit (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXHGB",
        kind="numeric",
        level=3,
        definition="Hemoglobin (g/dL)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXLYPCT",
        kind="numeric",
        level=3,
        definition="Lymphocyte percent (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXMCHSI",
        kind="numeric",
        level=3,
        definition="Mean cell hemoglobin (pg)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXMOPCT",
        kind="numeric",
        level=3,
        definition="Monocyte percent (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXMPSI",
        kind="numeric",
        level=3,
        definition="Mean platelet volume (fL)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXNEPCT",
        kind="numeric",
        level=3,
        definition="Segmented neutrophils percent (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXPLTSI",
        kind="numeric",
        level=3,
        definition="Platelet count (%) SI",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXRBCSI",
        kind="numeric",
        level=3,
        definition="Red cell count SI",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXRDW",
        kind="numeric",
        level=3,
        definition="Red cell distribution width (%)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="LBXSASSI",
        kind="numeric",
        level=3,
        definition="AST (U/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXSATSI",
        kind="numeric",
        level=3,
        definition="ALT (U/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXSC3SI",
        kind="numeric",
        level=3,
        definition="Bicarbonate (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXSGTSI",
        kind="numeric",
        level=3,
        definition="GGT (U/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXSKSI",
        kind="numeric",
        level=3,
        definition="Potassium (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXSNASI",
        kind="numeric",
        level=3,
        definition="Sodium (mmol/L)",
        comment=None,
        domain=None,
        file_location="Standard Biochemistry Profile",
    ),
    NHANESFeatureSpec(
        name="LBXTC",
        kind="numeric",
        level=3,
        definition="Total cholesterol (mg/dL)",
        comment=None,
        domain=None,
        file_location="Cholesterol - Total",
    ),
    NHANESFeatureSpec(
        name="LBXWBCSI",
        kind="numeric",
        level=3,
        definition="White blood cell count (SI)",
        comment=None,
        domain=None,
        file_location="Complete Blood Count with 5-Part Differential - Whole Blood",
    ),
    NHANESFeatureSpec(
        name="URXUMA",
        kind="numeric",
        level=3,
        definition="Urine Albumin (ug/mL)",
        comment=None,
        domain=None,
        file_location="Albumin & Creatinine - Urine",
    ),
]

FEATURE_SPECS_BY_NAME = {spec.name: spec for spec in NHANES_FEATURE_SPECS}


def get_feature_specs(
    *,
    levels: Iterable[int] | None = None,
    kind: str | None = None,
) -> list[NHANESFeatureSpec]:
    """Return feature specs filtered by level and/or kind."""
    level_set = None if levels is None else set(levels)
    specs = NHANES_FEATURE_SPECS
    if level_set is not None:
        specs = [spec for spec in specs if spec.level in level_set]
    if kind is not None:
        specs = [spec for spec in specs if spec.kind == kind]
    return specs


def get_feature_names(
    *,
    levels: Iterable[int] | None = None,
    kind: str | None = None,
) -> list[str]:
    """Return feature names filtered by level and/or kind."""
    return [spec.name for spec in get_feature_specs(levels=levels, kind=kind)]


def describe_feature_bundles(
    *,
    levels: Iterable[int] | None = None,
) -> list[dict[str, int | str | list[str]]]:
    """Return public schema metadata for NHANES feature bundles."""
    if levels is None:
        requested_levels = sorted({spec.level for spec in NHANES_FEATURE_SPECS})
    else:
        requested_levels = list(dict.fromkeys(int(level) for level in levels))

    rows: list[dict[str, int | str | list[str]]] = []
    for level in requested_levels:
        numeric_features = get_feature_names(levels=[level], kind="numeric")
        categorical_features = get_feature_names(levels=[level], kind="categorical")
        rows.append(
            {
                "level": int(level),
                "description": FEATURE_LEVEL_DESCRIPTIONS.get(level, f"Level {level} covariates"),
                "feature_count": len(numeric_features) + len(categorical_features),
                "numeric_count": len(numeric_features),
                "categorical_count": len(categorical_features),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            }
        )
    return rows


LEVEL_1_NUMERIC_FEATURES = get_feature_names(levels=[1], kind="numeric")
LEVEL_1_CATEGORICAL_FEATURES = get_feature_names(levels=[1], kind="categorical")
LEVEL_2_NUMERIC_FEATURES = get_feature_names(levels=[2], kind="numeric")
LEVEL_2_CATEGORICAL_FEATURES = get_feature_names(levels=[2], kind="categorical")
LEVEL_3_NUMERIC_FEATURES = get_feature_names(levels=[3], kind="numeric")
LEVEL_3_CATEGORICAL_FEATURES = get_feature_names(levels=[3], kind="categorical")

NUMERIC_FEATURES = get_feature_names(kind="numeric")
CATEGORICAL_FEATURES = get_feature_names(kind="categorical")
