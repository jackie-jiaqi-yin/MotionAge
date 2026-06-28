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
  reports/
    table1_inputs.csv
    robustness_inputs.csv
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

Manifest paths must be relative to the artifact bundle and must not use parent
directory segments. The validator rejects absolute paths and `..` escapes so a
public manifest cannot point at local files outside the approved bundle.

## Manifest Checks

Optional artifact bundles can include a `manifest.yaml` file that is readable by
`motionage.artifacts`. Manifest paths must be relative to the manifest file.

Example:

```yaml
version: 1
artifacts:
  - id: aggregate_table_inputs
    path: reports/table1_inputs.csv
    description: Aggregate synthetic inputs for reproducing report formatting.
    sha256: null
    required: true
```

The manifest helper reports present files, missing required files, missing
optional files, and checksum mismatches without requiring artifacts to be stored
in git. The JSON report also includes a `summary` block with
`total_artifacts`, `required_artifacts`, `optional_artifacts`, and
`checksum_protected_artifacts` counts so readers can inspect expected bundle
coverage before downloading or validating private artifact contents.

The report also includes `public_boundary_issues`. This field flags manifest
entries whose ids, paths, descriptions, or other manifest metadata look unsafe
for the public repository boundary, including raw data, participant or subject
rows, checkpoints, model weight files, private paths, and restricted response
materials.
