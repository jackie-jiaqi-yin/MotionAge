from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_docs_include_paper_manifest_validation_recipe() -> None:
    readme = _read_doc("README.md")
    reproduction = _read_doc("docs/reproduction.md")

    for text in (readme, reproduction):
        assert "motionage-validate-paper-models" in text
        assert "motionage validate-paper-models" in text
        assert "motionage --version" in text
        assert "motionage-validate-paper-models --version" in text
        assert "configs/paper/mortality_cv_primary_60m.yaml" in text
        assert "--summary-only" in text
        for family in ("GRU", "LSTM", "Transformer"):
            assert family in text


def _read_doc(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")
