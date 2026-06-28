from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import yaml


def _load_cli_main():
    script_path = Path(__file__).parents[1] / "scripts" / "reproduce" / "validate_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_artifacts_cli", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CLI script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


main = _load_cli_main()


def _write_manifest(
    root: Path,
    artifacts: list[dict[str, object]],
) -> Path:
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump({"version": 1, "artifacts": artifacts}, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def test_validate_artifacts_cli_reports_complete_bundle(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("SEQN,score\n1,0.8\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "participant_predictions",
                "path": "predictions.csv",
                "description": "Synthetic predictions.",
                "required": True,
            }
        ],
    )

    exit_code = main([str(manifest_path)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["ok"] is True
    assert report["present"] == ["participant_predictions"]


def test_validate_artifacts_cli_fails_missing_required(tmp_path: Path, capsys) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "participant_predictions",
                "path": "missing.csv",
                "description": "Missing required artifact.",
                "required": True,
            }
        ],
    )

    exit_code = main([str(manifest_path)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["ok"] is False
    assert report["missing_required"] == ["participant_predictions"]


def test_validate_artifacts_cli_can_fail_on_missing_optional(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "required.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "required",
                "path": "required.csv",
                "description": "Present required artifact.",
                "required": True,
            },
            {
                "id": "optional",
                "path": "optional.csv",
                "description": "Missing optional artifact.",
                "required": False,
            },
        ],
    )

    exit_code = main([str(manifest_path), "--fail-on-optional-missing"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["ok"] is True
    assert report["missing_optional"] == ["optional"]
