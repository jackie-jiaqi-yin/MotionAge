# CLI Reference

This page lists the public MotionAge commands that are available without raw NHANES files, processed participant tables, trained checkpoints, or artifact bundles.

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

JSON output includes a `public_boundary` block that declares the summary excludes raw data, participant-level rows, exact split IDs, trained checkpoints, and private notes while including public config metadata. Markdown output includes the same information in a `Public Boundary` section with fields such as `contains_raw_data` and `contains_public_config_metadata`.

JSON output also includes a `manifest_readiness` block with ready and not-ready aggregate counts. Markdown output includes the same information in a `Manifest Readiness` section.

Use `--markdown --summary-only` when preparing a compact reader-facing report, and omit `--summary-only` when inspecting per-model config metadata.
