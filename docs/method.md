# Method Overview

MotionAge converts wearable activity patterns into a mortality-risk representation and then maps that representation onto a chronological-age scale.

## Stage 1: Mortality Risk Model

The first-stage model predicts fixed-horizon mortality from minute-level accelerometer windows. Paper-visible model families include:

- GRU,
- LSTM,
- Transformer.

Covariate variants include late-fusion and residual-fusion designs for the same
paper-visible model families. The public package exposes these model families as
inspectable implementations and lightweight synthetic-test targets; trained
weights and local run outputs remain outside git.
`public_model_family_catalog` returns aggregate model-family rows for public
summaries so readers can see the covered GRU, LSTM, Transformer, and static
covariate variants without exposing checkpoints, weights, or private run paths.
`public_model_architecture_summary` turns a public model config into allowlisted
architecture metadata, public hyperparameters, and parameter counts for report
tables; it does not read checkpoint files, trained weights, or local run
directories.

## Model Initialization

Checkpoint initialization can be used to warm-start paper-visible GRU, LSTM,
and Transformer variants when reproducing experiments locally. Public reports
should summarize initialization with aggregate compatibility counts and loaded
tensor fractions, while omitting checkpoint paths, weights, and private run
state by default.

## MotionAge Mapping

For participant `i` and window `t`, let `z_it` be the first-stage mortality logit.

```text
p_it = sigmoid(z_it)
p_i = mean_t p_it
logit(q_s,a) = alpha_s + beta_s * age_bin_a
MotionAge_i = (logit(p_i) - alpha_s) / beta_s
MotionAgeAccel_i = MotionAge_i - chronological_age_i
```

The mapping is fit by sex using training participants only. Public configs should expose the fit partitions, clipping epsilon, weighting behavior, and whether mapped ages are clamped to the fit age range.

## Secondary Evaluation

Secondary evaluation compares chronological age, benchmark biological-age measures, first-stage risk scores, MotionAge, and MotionAge acceleration for mortality prediction. Report scripts should separate:

- point estimates,
- fold mean and standard deviation,
- paired confidence intervals,
- robustness and sensitivity analyses.

Bootstrap summaries should record enough metadata to make the reported interval
auditable: `n_resamples_requested`, `ci_level`, whether participant resampling
was `stratified`, and `fold_stratified` when the interval averages
within-fold paired AUROC deltas.

Public second-stage logistic comparisons can use `evaluate_secondary_feature_sets`
to produce aggregate-only train/test metric rows for MotionAge, MotionAgeAccel,
and benchmark feature sets.

Binary evaluation summaries should include aggregate split context only:
`n`, `events`, `non_events`, and `event_rate`. These counts make metrics
auditable without exposing participant rows or split ID files.
`build_public_binary_evaluation_row` converts split-level metric dictionaries
into allowlisted public table rows for GRU, LSTM, Transformer, or benchmark
comparisons.

## Leakage Controls

All preprocessing that learns parameters from data should fit on training partitions only. This includes covariate imputation, scaling, categorical encoding, MotionAge mapping, and secondary classifiers.
