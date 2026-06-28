from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_DIR = REPO_ROOT / "configs" / "paper"


def test_validate_paper_models_cli_reports_family_coverage(capsys: pytest.CaptureFixture[str]) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main([str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")])

    captured = capsys.readouterr()
    assert status == 0
    assert "Validated 10 paper model configs" in captured.out
    assert "gru: 4" in captured.out
    assert "lstm: 3" in captured.out
    assert "transformer: 3" in captured.out
    assert captured.err == ""


def test_validate_paper_models_cli_returns_error_for_invalid_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    config_dir = tmp_path / "configs" / "paper"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "gru_only.yaml",
        {"task": {"type": "binary_classification"}, "model": {"type": "gru_binary"}},
    )
    manifest = config_dir / "manifest.yaml"
    _write_yaml(
        manifest,
        {
            "models": [
                {
                    "model_id": "gru_only",
                    "family": "gru",
                    "source_config_path": "configs/paper/gru_only.yaml",
                }
            ],
            "not_ready_models": [],
        },
    )

    status = validate_paper_models_main([str(manifest), "--required-family", "gru", "--required-family", "lstm"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "Missing required model families" in captured.err


def test_validate_paper_models_console_script_is_registered() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["motionage-validate-paper-models"]
        == "motionage.cli:validate_paper_models_main"
    )


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
