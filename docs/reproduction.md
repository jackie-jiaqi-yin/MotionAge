# Reproduction

This repository supports three reproduction levels. Core library smoke tests are available in the public foundation. Public-data commands are staged as topic PRs add command-line wrappers and approved artifact-replay entry points.

## Level 1: Synthetic Smoke Test

Purpose: verify that the package imports, model constructors, MotionAge mapping utilities, benchmark helpers, statistics, and reporting helpers behave correctly without NHANES data.

```bash
uv sync --extra test
uv run python -m compileall src
uv run pytest
```

## Level 2: Public-Data Reproduction

Purpose: regenerate local NHANES-derived inputs and run paper model configurations.

Staged command structure:

```bash
uv run python scripts/data/prepare_nhanes_inputs.py --config configs/data/nhanes.yaml
uv run python scripts/data/make_splits.py --config configs/data/splits.yaml
uv run python scripts/train/run_mortality_cv.py --config configs/paper/mortality_cv_primary_60m.yaml
uv run python scripts/analyze/run_motionage.py --config configs/paper/motionage_analysis.yaml
```

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
