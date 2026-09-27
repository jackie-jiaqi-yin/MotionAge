from __future__ import annotations

import tomllib
import re
import shlex
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_inventory_matches_public_scaffold() -> None:
    readme = _read_doc("README.md")

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
    reproduction = _read_doc("docs/reproduction.md")

    assert (
        "initial scaffold only includes package and documentation checks"
        not in reproduction
    )
    assert "Core library smoke tests" in reproduction
    for line in reproduction.splitlines():
        if not line.startswith("uv run python "):
            continue
        command = shlex.split(line)
        if command[3] == "-m":
            continue
        assert (REPO_ROOT / command[3]).is_file(), line
        if "--config" in command:
            config = command[command.index("--config") + 1]
            if config.startswith("configs/"):
                assert (REPO_ROOT / config).is_file(), line


def test_documented_script_references_exist() -> None:
    for path in [REPO_ROOT / "README.md", *list((REPO_ROOT / "docs").rglob("*.md"))]:
        for script in re.findall(r"`(scripts/[A-Za-z0-9_/]+\.py)`", path.read_text()):
            assert (REPO_ROOT / script).is_file(), (path, script)


def test_tracked_files_exclude_generated_data_and_private_workspaces() -> None:
    paths = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    excluded_roots = {
        "data",
        "outputs",
        "experiments",
        "artifacts",
        "logs",
        ".planning",
        "rebuttal",
    }
    excluded_suffixes = {
        ".parquet",
        ".csv",
        ".tsv",
        ".xpt",
        ".dat",
        ".pt",
        ".pth",
        ".ckpt",
        ".npy",
        ".npz",
        ".ipynb",
    }
    for item in filter(None, paths):
        path = Path(item)
        assert path.parts[0] not in excluded_roots, item
        if not item.startswith("tests/fixtures/"):
            assert path.suffix.lower() not in excluded_suffixes, item


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
        assert "analysis_template" in text
        assert "public_boundary" in text
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
    assert "MotionAge Analysis Template" in cli_reference
    assert "Public Boundary" in cli_reference
    assert "contains_raw_data" in cli_reference
    assert "contains_public_config_metadata" in cli_reference
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


def test_public_metadata_links_are_publication_ready() -> None:
    readme = _read_doc("README.md")
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert (REPO_ROOT / "CITATION.cff").exists()
    assert (REPO_ROOT / "LICENSE").exists()
    assert "[CITATION.cff](CITATION.cff)" in readme
    assert "[LICENSE](LICENSE)" in readme
    assert (
        pyproject["project"]["urls"]["Repository"]
        == "https://github.com/jackie-jiaqi-yin/MotionAge"
    )
    assert citation["repository-code"] == pyproject["project"]["urls"]["Repository"]
    assert citation["title"] == "MotionAge"
    assert citation["version"] == pyproject["project"]["version"]
    assert citation["license"] == "MIT"
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
