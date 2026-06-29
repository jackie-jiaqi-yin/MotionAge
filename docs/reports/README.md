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

Source-prediction reports should use
`motionage.analysis.motionage.summarize_source_predictions` to publish split-level
counts, event rates, window totals, and probability summaries. Use
`motionage.analysis.motionage.build_public_source_prediction_report_table` when a
report needs side-by-side GRU, LSTM, Transformer, or local source-model rows. Do
not commit source prediction tables that contain one row per participant. When
report inputs use internal source-model keys, pass a reader-facing label map so
public tables export display labels rather than internal identifiers.
