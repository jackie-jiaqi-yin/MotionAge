from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

import motionage.artifacts as motionage_artifacts
from motionage.artifacts import (
    ArtifactManifest,
    ArtifactSpec,
    load_artifact_manifest,
    public_boundary_issues,
    validate_artifact_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_artifact_policy_documents_manifest_summary_output() -> None:
    artifact_policy = (REPO_ROOT / "docs" / "artifacts.md").read_text(encoding="utf-8")

    assert "summary" in artifact_policy
    assert "total_artifacts" in artifact_policy
    assert "required_artifacts" in artifact_policy
    assert "optional_artifacts" in artifact_policy
    assert "checksum_protected_artifacts" in artifact_policy
    assert "public_boundary_issues" in artifact_policy


def test_load_artifact_manifest_resolves_relative_paths_and_expected_checksums(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    predictions = artifact_dir / "predictions.csv"
    predictions.write_text("SEQN,probability\n1,0.7\n", encoding="utf-8")
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()

    manifest_path = artifact_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "artifacts": [
                    {
                        "id": "participant_predictions",
                        "path": "predictions.csv",
                        "description": "Synthetic participant-level probabilities.",
                        "sha256": digest,
                        "required": True,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = load_artifact_manifest(manifest_path)

    assert manifest.version == 1
    assert manifest.root == artifact_dir
    assert manifest.artifacts == (
        ArtifactSpec(
            artifact_id="participant_predictions",
            path=Path("predictions.csv"),
            description="Synthetic participant-level probabilities.",
            sha256=digest,
            required=True,
        ),
    )
    assert manifest.artifacts[0].resolve(artifact_dir) == predictions


def test_summarize_artifact_manifest_counts_required_optional_and_checksum_files(
    tmp_path: Path,
) -> None:
    assert hasattr(motionage_artifacts, "summarize_artifact_manifest")
    manifest = ArtifactManifest(
        version=1,
        root=tmp_path,
        artifacts=(
            ArtifactSpec("required_with_hash", Path("a.csv"), "Required hashed.", "a" * 64, True),
            ArtifactSpec("required_no_hash", Path("b.csv"), "Required unhashed.", None, True),
            ArtifactSpec("optional_no_hash", Path("c.csv"), "Optional unhashed.", None, False),
        ),
    )

    summary = motionage_artifacts.summarize_artifact_manifest(manifest)

    assert summary == {
        "total_artifacts": 3,
        "required_artifacts": 2,
        "optional_artifacts": 1,
        "checksum_protected_artifacts": 1,
    }


def test_validate_artifact_manifest_reports_missing_and_checksum_failures(tmp_path: Path) -> None:
    present = tmp_path / "present.csv"
    present.write_text("x\n1\n", encoding="utf-8")
    manifest = ArtifactManifest(
        version=1,
        root=tmp_path,
        artifacts=(
            ArtifactSpec("present", Path("present.csv"), "Present but wrong digest.", "0" * 64, True),
            ArtifactSpec("missing", Path("missing.csv"), "Missing required file.", None, True),
            ArtifactSpec("optional", Path("optional.csv"), "Optional missing file.", None, False),
        ),
    )

    report = validate_artifact_manifest(manifest)

    assert report["ok"] is False
    assert report["missing_required"] == ["missing"]
    assert report["missing_optional"] == ["optional"]
    assert report["checksum_mismatches"] == ["present"]
    assert report["present"] == ["present"]
    assert report["summary"] == {
        "total_artifacts": 3,
        "required_artifacts": 2,
        "optional_artifacts": 1,
        "checksum_protected_artifacts": 1,
    }


def test_public_boundary_issues_flags_non_release_manifest_entries(tmp_path: Path) -> None:
    manifest = ArtifactManifest(
        version=1,
        root=tmp_path,
        artifacts=(
            ArtifactSpec("raw_nhanes_export", Path("raw/participant_rows.parquet"), "Raw export.", None, True),
            ArtifactSpec("model_checkpoint", Path("checkpoints/fold0.ckpt"), "Checkpoint.", None, True),
            ArtifactSpec("reviewer_rebuttal", Path("docs/rebuttal/reply.md"), "Review reply.", None, False),
            ArtifactSpec("private_path", Path("reports/private-summary.csv"), "Private summary.", None, False),
        ),
    )

    issues = public_boundary_issues(manifest)

    assert issues == {
        "raw_nhanes_export": [
            "artifact id contains blocked public-boundary term: raw",
            "artifact path contains blocked public-boundary term: raw",
            "artifact path contains blocked public-boundary term: participant",
        ],
        "model_checkpoint": [
            "artifact id contains blocked public-boundary term: checkpoint",
            "artifact path contains blocked public-boundary term: checkpoint",
            "artifact path uses blocked artifact extension: .ckpt",
        ],
        "reviewer_rebuttal": [
            "artifact id contains blocked public-boundary term: rebuttal",
            "artifact id contains blocked public-boundary term: reviewer",
            "artifact path contains blocked public-boundary term: rebuttal",
        ],
        "private_path": [
            "artifact id contains blocked public-boundary term: private",
            "artifact path contains blocked public-boundary term: private",
        ],
    }


def test_validate_artifact_manifest_fails_public_boundary_issues(tmp_path: Path) -> None:
    manifest = ArtifactManifest(
        version=1,
        root=tmp_path,
        artifacts=(
            ArtifactSpec("public_table", Path("reports/table1_inputs.csv"), "Synthetic aggregate table.", None, True),
            ArtifactSpec("fold_checkpoint", Path("models/fold0.pt"), "Model weights.", None, False),
        ),
    )

    report = validate_artifact_manifest(manifest)

    assert report["ok"] is False
    assert report["public_boundary_issues"] == {
        "fold_checkpoint": [
            "artifact id contains blocked public-boundary term: checkpoint",
            "artifact path uses blocked artifact extension: .pt",
        ]
    }


def test_validate_artifact_manifest_passes_complete_bundle(tmp_path: Path) -> None:
    table = tmp_path / "table.csv"
    table.write_text("metric,value\nauroc,0.8\n", encoding="utf-8")
    manifest = ArtifactManifest(
        version=1,
        root=tmp_path,
        artifacts=(
            ArtifactSpec(
                artifact_id="table",
                path=Path("table.csv"),
                description="Synthetic table.",
                sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                required=True,
            ),
        ),
    )

    report = validate_artifact_manifest(manifest)

    assert report == {
        "ok": True,
        "present": ["table"],
        "missing_required": [],
        "missing_optional": [],
        "checksum_mismatches": [],
        "public_boundary_issues": {},
        "summary": {
            "total_artifacts": 1,
            "required_artifacts": 1,
            "optional_artifacts": 0,
            "checksum_protected_artifacts": 1,
        },
    }


def test_load_artifact_manifest_rejects_absolute_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "artifacts": [
                    {
                        "id": "bad",
                        "path": str(tmp_path / "bad.csv"),
                        "description": "Absolute paths should stay out of manifests.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative path"):
        load_artifact_manifest(manifest_path)
