# MotionAge and General Evaluation

The analysis workflow consumes local participant probabilities produced by
training or by your own model. It fits the existing sex-specific inverse-logit
age mapping and optionally evaluates held-out predictions and secondary models.

```bash
uv run python scripts/analyze/run_motionage.py --config configs/paper/motionage_analysis.yaml
```

`analysis.predictions_path` selects one CSV/Parquet table. Alternatively,
`analysis.participant_predictions_dir` selects a training output directory or a
completed CV directory with `cv_manifest.json`. CV entries are processed
independently, with one mapping per model/fold. The runner does not search
historical experiment directories or combine training rows across folds.

Required columns are a unique participant ID, probability, chronological age,
sex, and `split` for fitting. Column names are configurable under `analysis`.
`analysis.participants_path` can join age/sex and other features from a local
participant table. Missing participants and conflicting shared values are errors.
The target column is needed for evaluation but not for applying a saved mapping.

## Mapping Configuration

`mapping.fit_partitions` defaults to `[train]`; held-out test participants cannot
be used for fitting. `representative_probability` chooses mean or median risk
within rounded one-year age bins, separately for each sex. `weighted_fit` weights
those bins by their participant counts. `clip_eps`, `min_stratum_participants`,
`require_positive_beta`, and `clamp_output_to_fit_age_range` are explicit settings.
Unknown strata, insufficient age bins, non-finite inputs and invalid slopes fail
clearly. These errors can indicate that a source risk model is unsuitable for
the mapping; the workflow does not silently change the mapping to force a result.

Local outputs include participant MotionAge/MotionAgeAccel scores, mapping
parameters, a reusable `mapping.json`, age-bin diagnostics and aggregate metrics.
CV outputs are under `<output_dir>/<model_id>/<fold>`. For a single run, the
optional `outputs` block can override score/parameter/evaluation paths.
Use a fresh output directory; existing results are not overwritten.

Apply a mapping to new predictions by setting `mapping.parameters_path` to a
previous `mapping.json`, `analysis.predictions_path` to the new table, and
`evaluation.enabled: false` for unlabeled inputs. This reuses the fitted mapping
without learning from new participants. The saved mapping's column definitions
and fitting settings remain authoritative.

## Evaluation and Uncertainty

`evaluation.feature_sets` configures secondary logistic comparisons, for example
chronological age, MotionAge, and chronological age plus MotionAgeAccel. Fitting
uses development partitions only. The secondary helper reports complete-case
sample counts per feature set; these rows are not automatically paired comparisons
when the available populations differ.

The independent evaluator accepts one or more probability columns on a shared
participant table:

```bash
uv run python scripts/analyze/evaluate_predictions.py --config configs/examples/evaluation.yaml
```

It writes AUROC, AUPRC, log loss, Brier score and participant/event counts. Configure
`bootstrap.n_resamples`, `ci_level` and `random_seed`; zero resamples disables
uncertainty. Single-model AUROC intervals resample participants, dropping draws
with only one outcome class. Multiple columns also produce paired AUROC
differences using stratified participant resampling. Paired intervals currently
use 95% confidence. Adding `fold_column` uses within-fold stratified paired
resampling and reports the mean within-fold delta as a distinct estimand from
the pooled delta. Each participant must appear once, and every fold must contain
both classes.

These intervals condition on fixed predictions; they do not include model refit
uncertainty or NHANES survey-design weighting. Inputs and participant-level scores
stay local. Outputs are ordinary method results, with no publication-specific
table names, row ordering, or presentation templates.

PhenoAge calculation/imputation, external LLM-age comparison, wear-time
sensitivity and activity-profile summaries remain available as Python APIs.
See [benchmarks](benchmarks.md), [LLM-age](llm_age_benchmark.md),
[preprocessing](preprocessing.md), and [interpretability](interpretability.md).
