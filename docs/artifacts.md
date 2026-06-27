# Artifact Policy

This repository should stay small and reviewable. It should not store raw data, processed participant-level tables, model checkpoints, or generated experiment directories in git.

## Git-Tracked Content

The repository may track:

- source code,
- small synthetic fixtures,
- configuration files,
- documentation,
- tests,
- CI workflows.

## External Artifacts

Approved release artifacts may be distributed through GitHub Releases, Zenodo, or another durable archive. These artifacts should be optional for installing the package but useful for reproducing paper tables without retraining models.

Recommended artifact layout:

```text
artifacts/
  README.md
  checksums.sha256
  predictions/
    out_of_fold_predictions.parquet
    participant_motionage.parquet
  reports/
    table1_inputs.parquet
    robustness_inputs.parquet
  models/
    model-card.md
```

## Required Metadata

Every released artifact bundle should include:

- creation date,
- repository commit SHA,
- data source versions,
- command used to create the artifact,
- checksum file,
- license or usage constraints,
- clear note if an input cannot be redistributed.

## Local Placement

Users should place downloaded artifacts in a local `artifacts/` directory at the repository root. The directory is ignored by git.
