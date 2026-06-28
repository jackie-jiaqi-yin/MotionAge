# Method Overview

MotionAge converts wearable activity patterns into a mortality-risk representation and then maps that representation onto a chronological-age scale.

## Stage 1: Mortality Risk Model

The first-stage model predicts fixed-horizon mortality from minute-level accelerometer windows. Paper-visible model families include:

- GRU,
- LSTM,
- Transformer.

Covariate variants include late-fusion and residual-fusion designs. These implementations will be added in dedicated model PRs.

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

## Leakage Controls

All preprocessing that learns parameters from data should fit on training partitions only. This includes covariate imputation, scaling, categorical encoding, MotionAge mapping, and secondary classifiers.
