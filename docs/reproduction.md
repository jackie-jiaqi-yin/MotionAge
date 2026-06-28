# Reproduction

This repository supports three reproduction levels. The current public stack includes package checks plus paper model manifest validation. Full public-data preparation, model training, MotionAge analysis, and report replay commands are added in focused topic PRs as they become public-safe.

## Level 1: Synthetic Smoke and Paper Config Checks

Purpose: verify that the package imports, CI environment, and paper-visible model config inventory are healthy.

```bash
uv sync --extra test
uv run motionage --version
uv run motionage-validate-paper-models --version
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

This command validates the curated GRU, LSTM, and Transformer model families listed in `configs/paper/mortality_cv_primary_60m.yaml`. Use `--markdown --summary-only` when preparing a compact reader-facing summary of the same manifest metadata.

## Level 2: Public-Data Reproduction

Purpose: regenerate local NHANES-derived inputs and run paper model configurations.

Planned command structure:

```bash
uv run python scripts/data/prepare_nhanes_inputs.py --config configs/data/nhanes.yaml
uv run python scripts/data/make_splits.py --config configs/data/splits.yaml
uv run python scripts/train/run_mortality_cv.py --config configs/paper/mortality_cv_primary_60m.yaml
uv run python scripts/analyze/run_motionage.py --config configs/paper/motionage_analysis.yaml
```

## Level 3: Artifact Replay

Purpose: regenerate paper tables and robustness reports from approved prediction tables or model artifacts.

Planned command structure:

```bash
uv run python scripts/reproduce/table1_uncertainty.py --artifact-dir artifacts
uv run python scripts/reproduce/paired_auc_uncertainty.py --artifact-dir artifacts
uv run python scripts/reproduce/phenoage_benchmark.py --artifact-dir artifacts
uv run python scripts/reproduce/wear_preprocessing_sensitivity.py --artifact-dir artifacts
uv run python scripts/reproduce/activity_profile_interpretability.py --artifact-dir artifacts
```

Artifact replay commands should fail clearly when required artifact files are missing.
