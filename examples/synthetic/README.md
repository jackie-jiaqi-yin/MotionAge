# Synthetic Inputs

This folder contains a tiny deterministic data generator for local smoke tests.
It does not use or recreate NHANES participant records.

Generate example inputs:

```bash
uv run python examples/synthetic/make_synthetic_inputs.py --output-dir examples/synthetic/generated
```

The command writes:

- `activity_mortstat_joined.parquet`
- `nhanes_mortality_covariates_l1.parquet`
- `metadata_l1.yaml`
- `manifest.yaml`

`manifest.yaml` records the synthetic public-boundary declaration, row counts,
and column schema so downstream smoke tests can validate these examples without
assuming they are real NHANES participant records or trained model artifacts.

The generated directory is for local use and should not be committed.
