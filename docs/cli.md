# CLI Reference

Workflow commands read local inputs specified by configuration. The synthetic
example creates its own inputs. Metadata and environment commands need no data.

## Executable Workflows

All workflow scripts accept `--config FILE` and repeated `--set KEY=VALUE` overrides.
Run from the repository root; config paths refer to the working directory.

| Script | Purpose |
| --- | --- |
| `scripts/data/prepare_nhanes_inputs.py` | Prepare local raw inputs and covariate bundles |
| `scripts/data/make_splits.py` | Stratified participant splits or outer CV folds |
| `scripts/train/run_training.py` | Train one model or perform a fixed-epoch refit |
| `scripts/train/run_mortality_cv.py` | Execute a CV model manifest |
| `scripts/predict/predict.py` | Predict using saved weights and preprocessing |
| `scripts/analyze/run_motionage.py` | Fit/apply MotionAge and evaluate local predictions |
| `scripts/analyze/evaluate_predictions.py` | General metrics and bootstrap comparisons |

See [the runnable workflow](reproduction.md) for commands and configuration links.
The optional `scripts/reproduce/validate_artifacts.py` checks a local artifact
manifest; it does not generate publication tables.

## Version

Record the installed package version before sharing reproduction logs:

```bash
uv run motionage --version
uv run motionage-validate-paper-models --version
```

## Environment Report

Capture runtime and dependency metadata:

```bash
uv run motionage doctor
uv run motionage doctor --json
uv run motionage doctor --json --output reports/doctor.json
```

The doctor report includes the MotionAge package version, Python version, platform string, and key dependency versions.

## Paper Model Manifest

Validate that the public paper model manifest resolves the curated GRU, LSTM, and Transformer configuration set:

```bash
uv run motionage-validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

Equivalent top-level command:

```bash
uv run motionage validate-paper-models --json --summary-only configs/paper/mortality_cv_primary_60m.yaml
```

Default text output includes the analysis-template path, a compact
`public_boundary` line, `all_models_ready`, ready/not-ready config counts, and
GRU/LSTM/Transformer family counts.

JSON output includes a `public_boundary` block that declares the summary excludes raw data, participant-level rows, exact split IDs, trained checkpoints, and private notes while including public config metadata. Markdown output includes the same information in a `Public Boundary` section with fields such as `contains_raw_data` and `contains_public_config_metadata`.

JSON output also includes a `manifest_readiness` block with ready and not-ready aggregate counts. Markdown output includes the same information in a `Manifest Readiness` section.

JSON output includes an `analysis_template` block with the public MotionAge
mapping-template metadata from `configs/paper/motionage_analysis.yaml`: fit
partitions, strata, clip epsilon, mapping outputs, and public column names.
Markdown output includes the same information in a `MotionAge Analysis Template`
section. The template validation checks configuration structure and
repository-relative paths only; it does not require raw data, participant-level
prediction files, or generated outputs to exist.

Use `--markdown --summary-only` when preparing a compact reader-facing report, and omit `--summary-only` when inspecting per-model config metadata.
