from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


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


def test_public_metadata_links_are_publication_ready() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert (REPO_ROOT / "CITATION.cff").exists()
    assert (REPO_ROOT / "LICENSE").exists()
    assert "[CITATION.cff](CITATION.cff)" in readme
    assert "[LICENSE](LICENSE)" in readme
    assert pyproject["project"]["urls"]["Repository"] == "https://github.com/jackie-jiaqi-yin/MotionAge"
    assert citation["repository-code"] == pyproject["project"]["urls"]["Repository"]
    assert citation["title"] == "MotionAge"
    assert citation["version"] == pyproject["project"]["version"]
    assert citation["license"] == "MIT"
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
