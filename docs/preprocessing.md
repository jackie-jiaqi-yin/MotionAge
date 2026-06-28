# Preprocessing

MotionAge uses wear-aware accelerometer preprocessing before sequence modeling.
The public utilities in `motionage.preprocessing.wear` cover the reusable pieces
needed by reproduction scripts and sensitivity reports.

## Wear-Time Masking

`detect_nonwear_choi` implements a Choi-style non-wear detector based on long
zero-count intervals with optional isolated low-count spikes. Reproduction
scripts should record the minimum non-wear duration, spike tolerance, spike
count bounds, and required zero-count surround length.

## Epoch Aggregation

`downsample_wear_epochs` aggregates minute-level rows into fixed-length epochs.
It reports total intensity, wear minutes, a wear-minute mean intensity, and an
`attention_flag` set by the epoch wear proportion threshold `tau`.

For 5-minute epochs, thresholds `tau=0.1` and `tau=0.2` both retain epochs with
at least one observed wear minute. Higher thresholds require more wear minutes
inside the epoch.

## Window Retention

`summarize_window_retention` evaluates how sequence length and window coverage
cutoff affect retained windows, retained participants, and retained events.
These diagnostics are intended for preprocessing sensitivity reports before
running expensive end-to-end model refits.

## Covariate Missingness

`summarize_covariate_missingness` reports aggregate diagnostics for selected
static covariates: total rows, observed rows, missing rows, and `missing_rate`.
The helper is intended for public missingness and robustness summaries without
printing participant rows or split ID files.

## NHANES Feature Bundles

`describe_feature_bundles` returns public NHANES covariate schema metadata for
the paper-facing level 1/2/3 feature bundles. Each row includes the level
description, feature counts, numeric feature names, and categorical feature
names. This helper is for documentation, manifests, and reproduction checks; it
does not read raw data, emit participant rows, or reference private data paths.
