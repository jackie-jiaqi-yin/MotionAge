"""MotionAge public package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from motionage.paper_manifest import (
    MotionAgeAnalysisTemplate,
    PaperManifestReadiness,
    PaperModelManifestEntry,
    PaperStudyManifest,
    load_motionage_analysis_template,
    load_paper_manifest_readiness,
    load_paper_model_manifest,
    load_paper_study_manifest,
    validate_paper_model_manifest,
)

try:
    __version__ = version("motionage")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "MotionAgeAnalysisTemplate",
    "PaperManifestReadiness",
    "PaperModelManifestEntry",
    "PaperStudyManifest",
    "__version__",
    "load_motionage_analysis_template",
    "load_paper_manifest_readiness",
    "load_paper_model_manifest",
    "load_paper_study_manifest",
    "validate_paper_model_manifest",
]
