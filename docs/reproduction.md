# Reproduction

This repository supports three reproduction levels. Core library smoke tests and paper model manifest validation are available in the current public stack. Public-data commands are staged as topic PRs add command-line wrappers and approved artifact-replay entry points.

## Level 1: Synthetic Smoke and Paper Config Checks

Purpose: verify that the package imports, model constructors, MotionAge mapping utilities, benchmark helpers, statistics, reporting helpers, CLI entry points, and paper-visible model config inventory behave correctly without NHANES data.

```bash
uv sync --extra test
uv run motionage --version
uv run motionage-validate-paper-models --version
uv run motionage doctor
uv run motionage doctor --json
uv run motionage doctor --json --output reports/doctor.json
uv run python -m compileall src
uv run pytest
```

The paper manifest can be checked without raw NHANES files, processed participant tables, or trained checkpoints:

```bash
uv run motionage-validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

Equivalent top-level CLI form:

```bash
uv run motionage validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

This command validates the curated GRU, LSTM, and Transformer model families
listed in `configs/paper/mortality_cv_primary_60m.yaml`. Default text output
includes the analysis-template path, compact public-boundary flags, readiness
status, and family counts. JSON output includes `analysis_template`,
`public_boundary`, and `manifest_readiness` blocks with MotionAge
analysis-template metadata, public-scope flags, and ready/not-ready aggregate
counts; Markdown output includes the same information in
`MotionAge Analysis Template`, `Public Boundary`, and `Manifest Readiness`
sections. Use `--markdown --summary-only` when preparing a compact
reader-facing summary of the same manifest metadata.

See the [CLI reference](cli.md) for the current public command list.

## Level 2: Public-Data Reproduction

Purpose: regenerate local NHANES-derived inputs and run paper model configurations.

Staged command structure:

```bash
uv run python scripts/data/prepare_nhanes_inputs.py --config configs/data/nhanes.yaml
uv run python scripts/data/make_splits.py --config configs/data/splits.yaml
uv run python scripts/train/run_mortality_cv.py --config configs/paper/mortality_cv_primary_60m.yaml
uv run python scripts/analyze/run_motionage.py --config configs/paper/motionage_analysis.yaml
```

Training summaries should use public aggregate report rows that indicate whether
the fit was validation-selected or fixed-epoch without validation. Do not include
checkpoint paths, model weights, private run directories, or per-epoch logs in
publication-facing summaries.
When summaries include paper-visible model families such as GRU, LSTM, or
Transformer, provide reader-facing model labels so exported training rows avoid
internal model ids.
Use `motionage.training.build_public_fixed_epoch_plan_summary` before a
fixed-epoch refit when a manifest needs the planned epoch budget, resume offset,
remaining epochs, and final-refit strategy without exposing checkpoint paths,
run directories, or split identifiers.
Use `motionage.training.build_public_training_log_summary` when a report needs
training-log context; it reduces per-epoch logs to aggregate epoch, learning-rate,
freeze-stage, trainable-parameter, and selection-metric metadata.

## Level 3: Artifact Replay

Purpose: regenerate paper tables and robustness reports from approved prediction tables or model artifacts.

Staged command structure:

```bash
uv run python scripts/reproduce/table1_uncertainty.py --artifact-dir artifacts
uv run python scripts/reproduce/paired_auc_uncertainty.py --artifact-dir artifacts
uv run python scripts/reproduce/phenoage_benchmark.py --artifact-dir artifacts
uv run python scripts/reproduce/wear_preprocessing_sensitivity.py --artifact-dir artifacts
uv run python scripts/reproduce/activity_profile_interpretability.py --artifact-dir artifacts
```

Artifact replay commands should fail clearly when required artifact files are missing.

## Public Config Snapshots

Use `motionage.config.public_config_snapshot` when showing resolved configs in
reports, PR summaries, or reproduction notes. It preserves reproducibility knobs
such as model, windowing, training, and mapping settings while redacting
path-like values by default.

Use `motionage.config.public_config_rows` when a report needs a stable table for
one resolved config. The helper flattens public reproducibility leaves and omits
path-like keys such as local artifact files, output directories, and checkpoints.

Use `motionage.config.public_config_diff` when a robustness report needs to
summarize how two resolved configs differ. The helper emits stable, flattened
rows for public reproducibility settings and omits path-like keys such as local
artifact files, output directories, and checkpoints.
