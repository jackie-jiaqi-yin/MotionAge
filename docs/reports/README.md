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

Bootstrap and paired-AUROC interval tables should use
`public_bootstrap_interval_row` or `public_bootstrap_interval_table` to normalize
estimates, confidence intervals, resampling metadata, comparison labels, and
optional p-values into public-safe rows. Public interval rows may retain
`analysis` and `population` labels so multi-row robustness tables remain
readable, but should not include raw resample draws, local bootstrap output paths,
participant rows, private notes, or raw prediction tables.
When interval summaries carry comparison ids, pass a reader-facing comparison
label map before exporting public tables.
For paired model comparisons, `public_paired_auc_interval_table` builds public
pooled participant, stratified participant, and fold-stratified interval rows
directly from an in-memory paired score frame. The resulting rows include only
aggregate AUROC estimates, deltas, sample counts, event counts, confidence
intervals, p-values, and resampling metadata.

These reports should not include internal response text, ownership notes, or private planning context.

## Table Utilities

The `motionage.reporting.tables` module provides small utilities for report replay:

- summarize fold-level metrics as mean and fold standard deviation,
- format table entries as `mean +/- SD`,
- render lower-triangle paired confidence-interval matrices.

These utilities operate on in-memory tables so report scripts can use released
artifact inputs without hard-coding private experiment paths.
