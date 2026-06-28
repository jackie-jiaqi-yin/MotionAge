# Reports

This directory will hold neutral, publication-facing summaries generated from reproducible scripts.

Planned reports:

- Table 1 uncertainty and paired AUROC intervals
- PhenoAge missingness, imputation, and complete-case sensitivity
- Wear-aware preprocessing sensitivity
- Activity-profile interpretability

Bootstrap and paired-AUROC interval tables should use
`public_bootstrap_interval_row` to normalize estimates, confidence intervals,
resampling metadata, and optional p-values into public-safe rows. Report rows
should not include private resample draws, participant rows, or raw prediction
tables.

These reports should not include internal response text, ownership notes, or private planning context.
