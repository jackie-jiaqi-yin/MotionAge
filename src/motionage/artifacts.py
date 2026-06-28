"""Manifest helpers for optional MotionAge artifact bundles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_BLOCKED_PUBLIC_BOUNDARY_TERMS = (
    "raw",
    "participant",
    "subject",
    "checkpoint",
    "weight",
    "rebuttal",
    "reviewer",
    "openreview",
    "private",
)
_BLOCKED_ARTIFACT_EXTENSIONS = (".ckpt", ".pt", ".pth")


@dataclass(frozen=True)
class ArtifactSpec:
    """One file expected in an optional local artifact bundle."""

    artifact_id: str
    path: Path
    description: str
    sha256: str | None = None
    required: bool = True

    def resolve(self, root: Path) -> Path:
        """Resolve the artifact path against a bundle root."""
        return root / self.path


@dataclass(frozen=True)
class ArtifactManifest:
    """Parsed artifact manifest with paths relative to `root`."""

    version: int
    root: Path
    artifacts: tuple[ArtifactSpec, ...]


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    """Load an artifact manifest from YAML."""
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact manifest must be a mapping: {manifest_path}")

    version = int(payload.get("version", 1))
    rows = payload.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("Artifact manifest must include an artifacts list.")

    artifacts = tuple(_artifact_spec_from_mapping(row) for row in rows)
    return ArtifactManifest(version=version, root=manifest_path.parent, artifacts=artifacts)


def validate_artifact_manifest(manifest: ArtifactManifest) -> dict[str, Any]:
    """Validate file presence and optional SHA-256 digests for a manifest."""
    present: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    checksum_mismatches: list[str] = []
    boundary_issues = public_boundary_issues(manifest)

    for artifact in manifest.artifacts:
        path = artifact.resolve(manifest.root)
        if not path.exists():
            if artifact.required:
                missing_required.append(artifact.artifact_id)
            else:
                missing_optional.append(artifact.artifact_id)
            continue

        present.append(artifact.artifact_id)
        if artifact.sha256 is not None and _sha256(path) != artifact.sha256:
            checksum_mismatches.append(artifact.artifact_id)

    return {
        "ok": not missing_required and not checksum_mismatches and not boundary_issues,
        "present": present,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "checksum_mismatches": checksum_mismatches,
        "public_boundary_issues": boundary_issues,
        "summary": summarize_artifact_manifest(manifest),
    }


def public_boundary_issues(manifest: ArtifactManifest) -> dict[str, list[str]]:
    """Return manifest entries that look unsafe for a public release manifest."""
    issues: dict[str, list[str]] = {}
    for artifact in manifest.artifacts:
        artifact_issues: list[str] = []
        normalized_id = artifact.artifact_id.lower()
        normalized_path = artifact.path.as_posix().lower()

        for term in _BLOCKED_PUBLIC_BOUNDARY_TERMS:
            if term in normalized_id:
                artifact_issues.append(f"artifact id contains blocked public-boundary term: {term}")
        for term in _BLOCKED_PUBLIC_BOUNDARY_TERMS:
            if term in normalized_path:
                artifact_issues.append(f"artifact path contains blocked public-boundary term: {term}")

        suffix = artifact.path.suffix.lower()
        if suffix in _BLOCKED_ARTIFACT_EXTENSIONS:
            artifact_issues.append(f"artifact path uses blocked artifact extension: {suffix}")

        if artifact_issues:
            issues[artifact.artifact_id] = artifact_issues
    return issues


def summarize_artifact_manifest(manifest: ArtifactManifest) -> dict[str, int]:
    """Summarize manifest requirements without inspecting artifact contents."""
    return {
        "total_artifacts": len(manifest.artifacts),
        "required_artifacts": sum(1 for artifact in manifest.artifacts if artifact.required),
        "optional_artifacts": sum(1 for artifact in manifest.artifacts if not artifact.required),
        "checksum_protected_artifacts": sum(
            1 for artifact in manifest.artifacts if artifact.sha256 is not None
        ),
    }


def _artifact_spec_from_mapping(row: Any) -> ArtifactSpec:
    if not isinstance(row, dict):
        raise ValueError("Each artifact entry must be a mapping.")

    artifact_id = str(row.get("id", "")).strip()
    if not artifact_id:
        raise ValueError("Each artifact entry must include id.")

    raw_path = row.get("path")
    if raw_path in (None, ""):
        raise ValueError(f"Artifact {artifact_id} must include path.")
    artifact_path = Path(str(raw_path))
    if artifact_path.is_absolute():
        raise ValueError(f"Artifact {artifact_id} must use a relative path.")
    if ".." in artifact_path.parts:
        raise ValueError(f"Artifact {artifact_id} path must stay inside the artifact bundle.")

    description = str(row.get("description", "")).strip()
    if not description:
        raise ValueError(f"Artifact {artifact_id} must include description.")

    sha256 = row.get("sha256")
    if sha256 in ("", None):
        digest = None
    else:
        digest = str(sha256).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Artifact {artifact_id} has invalid sha256.")

    return ArtifactSpec(
        artifact_id=artifact_id,
        path=artifact_path,
        description=description,
        sha256=digest,
        required=bool(row.get("required", True)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
