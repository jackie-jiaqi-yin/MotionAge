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

Binary model report tables should be assembled from aggregate metrics with
`build_public_binary_evaluation_row`, which preserves public model and split
labels while filtering non-report fields such as participant identifiers,
prediction paths, and raw probability arrays.
Use `build_public_binary_evaluation_table` when starting from split-level
evaluation output; it carries threshold-selection metadata onto each public
split row without exposing raw predictions.
When report scripts start from a model id, pass a reader-facing label map so
the exported table uses public model names rather than internal identifiers.
PhenoAge and other benchmark robustness tables should use
`build_public_benchmark_sensitivity_table` to normalize primary, imputation
sensitivity, and complete-case sensitivity rows into aggregate public fields
only: analysis label, benchmark/comparator labels, n, events, AUROCs, deltas,
confidence intervals, and p-values.
MotionAge second-stage comparisons should use `evaluate_secondary_feature_sets`
when reporting aggregate secondary-evaluation rows for chronological-age,
MotionAge, MotionAgeAccel, or benchmark feature sets. Report outputs should
include aggregate metrics and counts only, not participant-level prediction tables.

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
The mapping frame requires sex values to resolve through default or explicit
public labels before they can appear in figure inputs.
