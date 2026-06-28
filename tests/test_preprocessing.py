from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import motionage.preprocessing.nhanes_features as nhanes_features
from motionage.preprocessing.covariates import (
    build_static_covariate_table,
    fit_static_covariate_preprocessor,
    transform_static_covariates,
)
from motionage.preprocessing.mortality import (
    build_fixed_horizon_mortality_table,
    fixed_horizon_target_definition,
)


def test_static_covariates_are_fit_on_training_rows_only() -> None:
    train = pd.DataFrame(
        {
            "SEQN": [1, 2, 3],
            "BMXBMI": [20.0, np.nan, 30.0],
            "RIAGENDR": [1, 2, np.nan],
        }
    )
    test = pd.DataFrame(
        {
            "SEQN": [4, 5],
            "BMXBMI": [25.0, np.nan],
            "RIAGENDR": [2, 99],
        }
    )

    preprocessor = fit_static_covariate_preprocessor(
        train,
        id_column="SEQN",
        numeric_columns=["BMXBMI"],
        categorical_columns=["RIAGENDR"],
    )
    transformed = transform_static_covariates(test, preprocessor)

    assert preprocessor.numeric_fill_values == {"BMXBMI": 25.0}
    assert transformed["SEQN"].tolist() == [4, 5]
    assert transformed["static_num"].shape == (2, 1)
    assert transformed["static_num_missing"][:, 0].tolist() == [0.0, 1.0]
    np.testing.assert_allclose(transformed["static_num"][:, 0], [0.0, 0.0])
    assert transformed["static_cat"][:, 0].tolist() == [2, 0]


def test_static_covariates_reject_duplicate_participant_rows() -> None:
    df = pd.DataFrame(
        {
            "SEQN": [1, 1],
            "BMXBMI": [20.0, 21.0],
            "RIAGENDR": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="one row per participant"):
        build_static_covariate_table(
            df,
            id_column="SEQN",
            numeric_columns=["BMXBMI"],
            categorical_columns=["RIAGENDR"],
        )


def test_fixed_horizon_mortality_table_derives_60_month_binary_label() -> None:
    mortality = pd.DataFrame(
        {
            "SEQN": [3, 1, 2, 4, 5],
            "mortstat": [1, 0, 1, 1, 2],
            "permth_int": [72, 120, 24, np.nan, 6],
            "eligibility": ["ok", "ok", "ok", "drop", "drop"],
        }
    )

    table = build_fixed_horizon_mortality_table(
        mortality,
        id_column="SEQN",
        mortality_column="mortstat",
        followup_months=60,
        passthrough_columns=["eligibility"],
    )

    assert table["SEQN"].tolist() == [1, 2, 3]
    assert table["mortstat"].tolist() == [0, 1, 0]
    assert table["permth_int"].tolist() == [120.0, 24.0, 72.0]
    assert fixed_horizon_target_definition(
        mortality_column="mortstat",
        followup_month_column="permth_int",
        followup_months=60,
    ) == "I(mortstat = 1 and permth_int <= 60)"


def test_nhanes_feature_spec_public_surface_is_neutral() -> None:
    assert nhanes_features.LEVEL_1_NUMERIC_FEATURES == [
        "BMXBMI",
        "BMXWAIST",
        "BPXDI",
        "BPXSY",
    ]
    assert nhanes_features.LEVEL_1_CATEGORICAL_FEATURES == ["INDHHINC", "RIAGENDR"]
    assert nhanes_features.LEVEL_2_CATEGORICAL_FEATURES == [
        "DIQ010",
        "MCQ010",
        "PAD020",
        "PAD200",
        "PAD320",
    ]
    assert "LBXGH" in nhanes_features.LEVEL_3_NUMERIC_FEATURES
    assert "BPXPLS" not in nhanes_features.NUMERIC_FEATURES
    private_source_terms = ("work" + "book", "data" + "/raw", "cop" + "ied")
    module_doc = (nhanes_features.__doc__ or "").lower()
    assert all(term not in module_doc for term in private_source_terms)
    assert nhanes_features.get_feature_names(levels=[1, 2], kind="categorical") == [
        "INDHHINC",
        "RIAGENDR",
        "DIQ010",
        "MCQ010",
        "PAD020",
        "PAD200",
        "PAD320",
    ]
