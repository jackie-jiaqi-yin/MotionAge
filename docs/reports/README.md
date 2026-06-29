# Reports

This directory will hold neutral, publication-facing summaries generated from reproducible scripts.

Planned reports:

- Table 1 uncertainty and paired AUROC intervals
- PhenoAge missingness, imputation, and complete-case sensitivity
- Wear-aware preprocessing sensitivity
- Activity-profile interpretability

Wear-aware preprocessing sensitivity tables should report aggregate retained
window, participant, and event counts together with `eligible_windows`,
`window_retention_rate`, `participant_retention_rate`, and
`event_retention_rate`. These rows are intended to compare Choi non-wear
duration, tau, coverage cutoff, and sequence-length settings without exposing
participant identifiers or row-level activity traces.

These reports should not include internal response text, ownership notes, or private planning context.

## Table Utilities

The `motionage.reporting.tables` module provides small utilities for report replay:

- summarize fold-level metrics as mean and fold standard deviation,
- format table entries as `mean +/- SD`,
- render lower-triangle paired confidence-interval matrices.

These utilities operate on in-memory tables so report scripts can use released
artifact inputs without hard-coding private experiment paths.
