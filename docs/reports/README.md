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
