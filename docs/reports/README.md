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

The `motionage.reporting.mortality_cv` module provides mortality-CV report
helpers that collect fold summaries, aggregate official and secondary feature
sets, and format aggregate summary rows for publication tables.
`build_public_mortality_cv_summary_table` converts those summaries into
reader-facing labels and omits internal model ids, feature ids, and private
experiment paths from public tables.
`build_public_mortality_cv_rank_table` ranks aggregate model rows by a selected
summary metric such as AUROC or AUPRC while keeping only reader-facing labels,
fold counts, metric means, fold SDs, and formatted mean/SD cells.
Both public mortality-CV helpers require complete public label maps by default
so report scripts fail before internal ids appear in exported tables.

These utilities operate on in-memory tables so report scripts can use released
artifact inputs without hard-coding private experiment paths.
