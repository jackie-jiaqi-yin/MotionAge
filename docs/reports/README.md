# Reports

This directory will hold neutral, publication-facing summaries generated from reproducible scripts.

Planned reports:

- Table 1 uncertainty and paired AUROC intervals
- PhenoAge missingness, imputation, and complete-case sensitivity
- Wear-aware preprocessing sensitivity
- Activity-profile interpretability

These reports should not include internal response text, ownership notes, or private planning context.

## Table Utilities

The `motionage.reporting.tables` module provides small utilities for report replay:

- summarize fold-level metrics as mean and fold standard deviation,
- format table entries as `mean +/- SD`,
- render lower-triangle paired confidence-interval matrices.

These utilities operate on in-memory tables so report scripts can use released
artifact inputs without hard-coding private experiment paths.

## Figure Utilities

The `motionage.visualization` module provides figure helpers for public report
inputs, including aggregate confidence-interval forest plots and hourly
activity trajectories by age group. These helpers operate on in-memory summary
tables so report scripts do not need to hard-code private experiment paths or
participant-level records.
Use `build_public_metric_interval_frame` before interval forest plots when
report artifacts include extra internal columns. It keeps only aggregate labels,
point estimates, confidence bounds, and metric labels, excluding paths, raw
predictions, and private resample draws.
Use `build_public_lower_triangle_ci_heatmap_frame` before lower-triangle paired confidence-interval heatmaps.
It keeps only ordered aggregate row/column labels, deltas, confidence bounds,
significance flags, and annotations, excluding paths and private resample draws.
Use `build_public_motionage_mapping_frame` before MotionAge mapping diagnostic
plots when report artifacts include extra internal columns. It keeps only
aggregate age-bin mapping diagnostics, sex labels, representative/fitted
probabilities, logits, and optional aggregate counts, excluding source paths,
raw row references, split identifiers, and participant-level records.
