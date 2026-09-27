# Synthetic Examples

## Complete Raw-Input Workflow

```bash
uv run python examples/synthetic/run_workflow.py
uv run python examples/synthetic/run_workflow.py --cv --output-dir outputs/synthetic_cv
```

The example creates 64 fictional participants with minute-level activity,
demographics and mortality follow-up. It writes configs and invokes the same
preparation, split, training and analysis scripts used with local real inputs.
The default trains a small GRU; `--family lstm` and `--family transformer` select
another single model. `--cv` executes all three model families across two folds.

The synthetic age/activity/outcome relationship is deliberately strong so that a
small CPU model can demonstrate a positive-slope risk-to-age mapping. Its scores
and metrics are not research findings or realistic performance estimates.

Outputs include:

- `raw/`: generated CSV inputs, containing no real participants;
- `configs/`: editable preparation, split, model and analysis YAML files;
- `processed/`: model-ready inputs and covariate metadata;
- `splits/`: local participant IDs;
- `training/`: local weights, preprocessing state, predictions and metrics;
- `analysis/`: MotionAge scores, mapping parameters and aggregate evaluation;
- `example_manifest.json`: synthetic provenance, seed and commands.

Use a fresh `--output-dir` for another run. `--prepare-only` generates raw files
and configs without execution. Paths written into these generated configs refer
to the chosen local output directory; none of these outputs belongs in Git.

## Lightweight Prepared-Input Generator

The older small generator remains useful for dataset/model unit experiments:

```bash
uv run python examples/synthetic/make_synthetic_inputs.py --output-dir examples/synthetic/generated --participants 64
uv run python scripts/train/run_training.py --config configs/examples/synthetic_smoke.yaml
```

It writes prepared activity/covariate Parquet plus metadata and a synthetic
manifest. These are hourly mock observations, not raw NHANES minute data.
`PAXDAY` follows the 1-7 weekday convention and `PAXN` provides chronological
ordering. Its random outcomes may yield unsuitable mappings for some small
samples; use the complete workflow above for the end-to-end demonstration.

The generator's Python helpers `validate_synthetic_inputs` and
`build_synthetic_smoke_summary` validate files and summarize aggregate counts.
Generated files stay under ignored directories.
