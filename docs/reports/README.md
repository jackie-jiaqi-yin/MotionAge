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
For paired model comparisons, `public_paired_auc_interval_table` builds public
pooled participant, stratified participant, and fold-stratified interval rows
directly from an in-memory paired score frame. The resulting rows include only
aggregate AUROC estimates, deltas, sample counts, event counts, confidence
intervals, p-values, and resampling metadata.

These reports should not include internal response text, ownership notes, or private planning context.
