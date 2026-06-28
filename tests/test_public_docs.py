from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_inventory_matches_public_scaffold() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for term in (
        "GRU",
        "LSTM",
        "Transformer",
        "MotionAge risk-to-age mapping",
        "PhenoAge benchmark",
        "activity-profile interpretability",
    ):
        assert term in readme

    assert "Model implementations, MotionAge analysis, benchmark scripts" not in readme


def test_reproduction_doc_describes_current_public_scope() -> None:
    reproduction = (REPO_ROOT / "docs" / "reproduction.md").read_text(encoding="utf-8")

    assert "initial scaffold only includes package and documentation checks" not in reproduction
    assert "Core library smoke tests" in reproduction
    assert "Public-data commands are staged" in reproduction
