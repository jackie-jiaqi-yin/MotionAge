# Method Overview

MotionAge converts wearable activity patterns into a mortality-risk representation and then maps that representation onto a chronological-age scale.

## Stage 1: Mortality Risk Model

The first-stage model predicts fixed-horizon mortality from minute-level accelerometer windows. Paper-visible model families include:

- GRU,
- LSTM,
- Transformer.

Covariate variants include late-fusion and residual-fusion designs. The public package exposes these model families as inspectable implementations and lightweight synthetic-test targets; trained weights and local run outputs remain outside git.

## MotionAge Mapping

For participant `i` and window `t`, let `z_it` be the first-stage mortality logit.

```text
p_it = sigmoid(z_it)
p_i = mean_t p_it
logit(q_s,a) = alpha_s + beta_s * age_bin_a
MotionAge_i = (logit(p_i) - alpha_s) / beta_s
MotionAgeAccel_i = MotionAge_i - chronological_age_i
```

`build_participant_source_predictions` implements the public source-prediction
aggregation step for synthetic or local run outputs only: it converts
window-level logits or probabilities into participant-level mean probability,
optional participant logit, split label, target, and window count.
`build_public_source_prediction_report_table` then combines aggregate-only
source-model summaries for GRU, LSTM, Transformer, or another local source
model label without exposing participant rows, checkpoint paths, or split IDs.

The mapping is fit by sex using training participants only. Public configs should expose the fit partitions, clipping epsilon, weighting behavior, and whether mapped ages are clamped to the fit age range.

## Secondary Evaluation

Secondary evaluation compares chronological age, benchmark biological-age measures, first-stage risk scores, MotionAge, and MotionAge acceleration for mortality prediction. Report scripts should separate:

- point estimates,
- fold mean and standard deviation,
- paired confidence intervals,
- robustness and sensitivity analyses.

## Leakage Controls

All preprocessing that learns parameters from data should fit on training partitions only. This includes covariate imputation, scaling, categorical encoding, MotionAge mapping, and secondary classifiers.
