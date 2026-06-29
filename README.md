# MotionAge

MotionAge is the public companion repository for a study of wearable activity patterns, mortality risk, and biological-age-style representations in NHANES accelerometer data.

The repository is organized as a paper-reproduction codebase. It is intended for readers who want to inspect the implementation, prepare public NHANES inputs locally, run smoke tests, and reproduce paper tables or robustness reports from approved artifacts.

## Repository Status

This repository is being built in focused pull requests. The public foundation includes the core library modules needed to inspect the paper implementation. This paper-configuration stack adds public config templates and manifest validation CLI coverage for paper-visible GRU, LSTM, and Transformer model families.

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
- Documentation for data preparation, method overview, benchmark scope, artifact policy, and reproduction levels
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

## Reproduction Levels

1. **Synthetic smoke test:** verifies package imports and current public library behavior without NHANES data.
2. **Public-data reproduction:** prepares NHANES accelerometer, demographics, mortality, and covariate inputs from public sources on the user's machine.
3. **Artifact replay:** regenerates paper tables and reports from approved prediction tables or model artifacts placed outside git.

See [docs/reproduction.md](docs/reproduction.md) for the staged command structure.

## Citation

Citation metadata will be updated after the final publication record is available. For now, see [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
