# Reproduction

This package provides a runnable methodology workflow with explicit configs and
synthetic examples. It does not provide publication-specific table/figure replay.
Core library smoke tests and end-to-end workflow tests run without NHANES data.

## Synthetic End-to-End Example

```bash
uv sync --extra test
uv run python examples/synthetic/run_workflow.py
uv run python examples/synthetic/run_workflow.py --cv --output-dir outputs/synthetic_cv
```

The example generates raw synthetic activity, participant and mortality files,
then executes preparation, splitting, training and MotionAge analysis. The CV
variant trains GRU, LSTM and Transformer models on two folds. Outputs and editable
YAML configs stay under the chosen ignored output directory. These examples
demonstrate methodology and executable interfaces, not realistic performance or
numeric agreement with publication results.

Use `--family lstm` or `--family transformer` for another single-model run. Choose
a new `--output-dir` each time. `--prepare-only` writes raw inputs and configs so
you can execute each stage separately:

```bash
uv run python examples/synthetic/run_workflow.py --prepare-only --output-dir outputs/manual_example
uv run python scripts/data/prepare_nhanes_inputs.py --config outputs/manual_example/configs/prepare.yaml
uv run python scripts/data/make_splits.py --config outputs/manual_example/configs/splits.yaml
uv run python scripts/train/run_training.py --config outputs/manual_example/configs/gru.yaml
uv run python scripts/analyze/run_motionage.py --config outputs/manual_example/configs/analysis.yaml
```

## Local NHANES Workflow

Acquire the files and review schemas and cohort settings in [data preparation](data.md).
Edit the input paths and feature bundles for your local files. Then run:

```bash
uv run python scripts/data/prepare_nhanes_inputs.py --config configs/data/nhanes.yaml
uv run python scripts/data/make_splits.py --config configs/data/splits.yaml
uv run python scripts/train/run_mortality_cv.py --config configs/paper/mortality_cv_primary_60m.yaml
uv run python scripts/analyze/run_motionage.py --config configs/paper/motionage_analysis.yaml
```

This sequence uses the configured model presets and five participant folds.
The preparation config exposes age eligibility, follow-up policy, wear detection,
epoch aggregation and covariate fields. Training records actual input dimensions
and train-fitted preprocessing. Review these settings for your analysis rather
than treating a config filename as proof of publication-result equivalence.

See [training and prediction](training.md) for single models, refitting and saved-model
inference, and [analysis](analysis.md) for mapping reuse and general uncertainty.
All data, IDs, weights and participant scores remain local and ignored by Git.

## General Evaluation

After the default synthetic example, evaluate its held-out predictions separately:

```bash
uv run python scripts/analyze/evaluate_predictions.py --config configs/examples/evaluation.yaml
```

For your own predictions, configure ID, target and probability columns, optional
fold identifiers, and bootstrap settings. CI/bootstrap, PhenoAge missingness,
wear sensitivity and activity-profile utilities are reusable methods. They do not
depend on a particular publication table layout or internal experiment history.

## Environment and Tests

```bash
uv run motionage --version
uv run motionage-validate-paper-models --version
uv run motionage doctor
uv run motionage doctor --json
uv run motionage doctor --json --output reports/doctor.json
uv run python -m compileall src scripts examples
uv run pytest
```

Inspect the curated model configuration inventory without reading data:

```bash
uv run motionage-validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
uv run motionage validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

The result covers GRU, LSTM and Transformer config metadata. `analysis_template`,
`public_boundary` and `manifest_readiness` describe config structure and declared
ready/not-ready inventory, not completed training, scientific validation, or an
available dataset. See the [CLI reference](cli.md) for command details.
