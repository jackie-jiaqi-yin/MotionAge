# Benchmarks

MotionAge comparisons include chronological age and external biological-age
benchmarks. The repository should keep benchmark handling explicit because
missing-data choices can change the evaluated cohort.

## PhenoAge

The `motionage.benchmarks.phenoage` module exposes reusable helpers for:

- computing PhenoAge from the published formula inputs,
- summarizing pre-imputation missingness for each PhenoAge input,
- fitting train-only median imputers,
- fitting train-only sex by age-band median imputers with sex and global fallbacks,
- selecting complete-case PhenoAge cohorts for sensitivity analyses.

Report and reproduction scripts should fit imputers on training folds only and
then apply the fitted values to held-out participants. Complete-case analyses
should be reported as sensitivity cohorts because they condition evaluation on
laboratory availability.
