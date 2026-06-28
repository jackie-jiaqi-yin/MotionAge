# Reports

This directory will hold neutral, publication-facing summaries generated from reproducible scripts.

Planned reports:

- Table 1 uncertainty and paired AUROC intervals
- PhenoAge missingness, imputation, and complete-case sensitivity
- Wear-aware preprocessing sensitivity
- Activity-profile interpretability

Binary model report tables should be assembled from aggregate metrics with
`build_public_binary_evaluation_row`, which preserves public model and split
labels while filtering non-report fields such as participant identifiers,
prediction paths, and raw probability arrays.
Use `build_public_binary_evaluation_table` when starting from split-level
evaluation output; it carries threshold-selection metadata onto each public
split row without exposing raw predictions.

These reports should not include internal response text, ownership notes, or private planning context.
