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
        assert "motionage doctor" in text
        assert "motionage doctor --json" in text
        assert "motionage doctor --json --output" in text
        assert "configs/paper/mortality_cv_primary_60m.yaml" in text
        assert "--summary-only" in text
        assert "manifest_readiness" in text
        assert "not-ready" in text
        for family in ("GRU", "LSTM", "Transformer"):
            assert family in text


def test_public_docs_include_cli_reference() -> None:
    readme = _read_doc("README.md")
    reproduction = _read_doc("docs/reproduction.md")
    cli_reference = _read_doc("docs/cli.md")

    assert "[docs/cli.md](docs/cli.md)" in readme
    assert "[CLI reference](cli.md)" in reproduction
    assert cli_reference.startswith("# CLI Reference\n")
    assert "Manifest Readiness" in cli_reference
    for command in (
        "uv run motionage --version",
        "uv run motionage doctor",
        "uv run motionage doctor --json",
        "uv run motionage doctor --json --output reports/doctor.json",
        "uv run motionage-validate-paper-models --json --summary-only",
        "uv run motionage validate-paper-models --json --summary-only",
    ):
        assert command in cli_reference


def _read_doc(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")
