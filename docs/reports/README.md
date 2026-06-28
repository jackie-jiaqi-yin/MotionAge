# Reports

This directory will hold neutral, publication-facing summaries generated from reproducible scripts.

Planned reports:

- Table 1 uncertainty and paired AUROC intervals
- PhenoAge missingness, imputation, and complete-case sensitivity
- Wear-aware preprocessing sensitivity
- Activity-profile interpretability

Bootstrap and paired-AUROC interval tables should use
`public_bootstrap_interval_row` or `public_bootstrap_interval_table` to normalize
estimates, confidence intervals, resampling metadata, comparison labels, and
optional p-values into public-safe rows. Report rows should not include raw
resample draws, local bootstrap output paths, participant rows, or raw prediction
tables.

These reports should not include internal response text, ownership notes, or private planning context.
