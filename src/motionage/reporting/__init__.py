"""Report-generation helpers for MotionAge publication artifacts."""

from __future__ import annotations

from motionage.reporting.mortality_cv import (
    build_mortality_cv_summary_table,
    build_public_mortality_cv_rank_table,
    build_public_mortality_cv_summary_table,
    collect_fold_metrics,
    summarize_mortality_cv_fold_metrics,
)
from motionage.reporting.tables import (
    build_lower_triangle_ci_matrix,
    format_mean_sd,
    render_markdown_table,
    summarize_fold_metrics,
)

__all__ = [
    "build_lower_triangle_ci_matrix",
    "build_mortality_cv_summary_table",
    "build_public_mortality_cv_rank_table",
    "build_public_mortality_cv_summary_table",
    "collect_fold_metrics",
    "format_mean_sd",
    "render_markdown_table",
    "summarize_fold_metrics",
    "summarize_mortality_cv_fold_metrics",
]
