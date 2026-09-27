# MotionAge

MotionAge is the public companion repository for a study of wearable activity patterns, mortality risk, and biological-age-style representations in NHANES accelerometer data.

The package exposes a complete, configuration-driven methodology workflow:
local data preparation, participant splitting, model training and CV, prediction,
MotionAge calculation, and general evaluation. Readers can inspect the methods,
run synthetic examples, or adapt the configs to their own inputs. Publication-specific
table/figure assembly and historical experiment management are outside its scope.

## Repository Status

The workflows are exercised end to end on synthetic inputs, including GRU, LSTM
and Transformer cross-validation. These tests verify executable interfaces and
method behavior, not numeric agreement with a publication or a completed full-scale
NHANES run. Model presets under `configs/paper/` are inspectable starting points;
input features and fitted dimensions are recorded in each run's resolved config.

## What Is Included

- Source package namespace: `motionage`
- Public data utilities for dataset validation, split manifests, and accelerometer window construction
- Model implementations for GRU, LSTM, Transformer, and static-covariate variants
- MotionAge risk-to-age mapping utilities
- PhenoAge benchmark helpers
- Paired-AUROC statistics and public report-table helpers
- Public activity-profile interpretability summaries
- Paper model configuration templates under `configs/paper/`
- A manifest validation CLI: `motionage-validate-paper-models`
- Executable data preparation, splitting, training/CV, prediction, MotionAge and evaluation commands
- Documentation for data schemas, method configuration, benchmark scope and local outputs
- CI checks and synthetic smoke tests
- A public boundary that keeps implementation code separate from generated data, checkpoints, and private research notes

## What Is Not Committed

- Raw NHANES files
- Processed participant-level datasets
- Trained model checkpoints
- Exact participant split ID files
- Collaborator-only prediction files
- Generated experiment directories
- Internal response text or private planning material

Large or restricted artifacts should be distributed separately only when they are approved for release.

## Quickstart

Prerequisites:

- Python 3.11 or newer
- `uv` package manager

Install dependencies:

```bash
uv sync --extra test
```

Run the complete synthetic workflow, including actual model optimization:

```bash
uv run python examples/synthetic/run_workflow.py
```

This creates synthetic raw inputs, editable configs, fitted models and results in
the ignored `outputs/synthetic_workflow/` directory. Use a fresh `--output-dir`
for another run. Run all three model families across two folds with:

```bash
uv run python examples/synthetic/run_workflow.py --cv --output-dir outputs/synthetic_cv
```

Record the installed package version for reproduction logs:

```bash
uv run motionage --version
uv run motionage-validate-paper-models --version
```

Capture a compact environment report:

```bash
uv run motionage doctor
uv run motionage doctor --json
uv run motionage doctor --json --output reports/doctor.json
```

Run the current smoke checks:

```bash
uv run python -m compileall src
uv run pytest
```

Validate the paper-visible model manifest without requiring NHANES data:

```bash
uv run motionage-validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

Equivalent top-level CLI form:

```bash
uv run motionage validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

The manifest check verifies that the curated paper configs cover the GRU, LSTM,
and Transformer families. The default text output reports the analysis-template
path, public-boundary flags, readiness status, and family counts. JSON output
includes `analysis_template`, `public_boundary`, and `manifest_readiness` blocks
with MotionAge analysis-template metadata, public-scope flags, and ready/not-ready
aggregate counts; Markdown output includes the same information in
`MotionAge Analysis Template`, `Public Boundary`, and `Manifest Readiness`
sections.

See [docs/cli.md](docs/cli.md) for the current public command reference.

## Methodology Workflow

1. [Prepare data and participant splits](docs/data.md) with explicit cohort and preprocessing configs.
2. [Train, cross-validate and predict](docs/training.md) with train-fitted preprocessing.
3. [Calculate MotionAge and evaluate predictions](docs/analysis.md), including reusable mappings and uncertainty.

See [docs/reproduction.md](docs/reproduction.md) for the runnable commands and
[synthetic examples](examples/synthetic/README.md) for generated input/config details.

## Citation

Citation metadata will be updated after the final publication record is available. For now, see [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
