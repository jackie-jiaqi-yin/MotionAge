"""Statistical utilities for MotionAge reports."""

from motionage.stats.paired_auc import (
    fold_structured_paired_bootstrap_auc_delta,
    make_paired_score_frame,
    paired_auc_delta,
    paired_bootstrap_auc_delta,
    safe_auc,
)

__all__ = [
    "fold_structured_paired_bootstrap_auc_delta",
    "make_paired_score_frame",
    "paired_auc_delta",
    "paired_bootstrap_auc_delta",
    "safe_auc",
]
