"""Preprocessing utilities and shared column definitions."""

from motionage.preprocessing.constants import ActivityColumns
from motionage.preprocessing.covariates import (
    StaticCovariatePreprocessor,
    build_static_covariate_table,
    fit_static_covariate_preprocessor,
    transform_static_covariates,
)
from motionage.preprocessing.mortality import (
    build_fixed_horizon_mortality_table,
    fixed_horizon_target_definition,
)
from motionage.preprocessing.wear import (
    detect_nonwear_choi,
    downsample_wear_epochs,
    retained_window_count,
    summarize_window_retention,
)

__all__ = [
    "ActivityColumns",
    "StaticCovariatePreprocessor",
    "build_fixed_horizon_mortality_table",
    "build_static_covariate_table",
    "detect_nonwear_choi",
    "downsample_wear_epochs",
    "fit_static_covariate_preprocessor",
    "fixed_horizon_target_definition",
    "retained_window_count",
    "summarize_window_retention",
    "transform_static_covariates",
]
