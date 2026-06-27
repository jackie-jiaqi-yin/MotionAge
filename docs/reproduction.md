# Reproduction

This repository supports three reproduction levels. The initial scaffold only includes package and documentation checks; model and report commands will be added in later PRs.

## Level 1: Synthetic Smoke Test

Purpose: verify that the package imports and CI environment are healthy.

```bash
uv sync --extra test
uv run python -m compileall src
uv run pytest
```

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
