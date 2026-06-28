"""MotionAge public package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from motionage.paper_manifest import (
    PaperModelManifestEntry,
    load_paper_model_manifest,
    validate_paper_model_manifest,
)

try:
    __version__ = version("motionage")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "PaperModelManifestEntry",
    "__version__",
    "load_paper_model_manifest",
    "validate_paper_model_manifest",
]
