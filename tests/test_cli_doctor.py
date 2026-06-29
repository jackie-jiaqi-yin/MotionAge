from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_DEPENDENCIES = ("numpy", "pandas", "pyyaml", "scipy", "scikit-learn", "torch")


def test_motionage_doctor_reports_text_environment() -> None:
    result = subprocess.run(
        ["uv", "run", "motionage", "doctor"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert result.stdout.splitlines()[0] == "MotionAge environment report"
    assert "motionage: " in result.stdout
    assert "python: " in result.stdout
    assert "platform: " in result.stdout
    assert "dependencies:" in result.stdout
    for dependency in DOCTOR_DEPENDENCIES:
        assert f"  {dependency}: " in result.stdout


def test_motionage_doctor_reports_json_environment() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    result = subprocess.run(
        ["uv", "run", "motionage", "doctor", "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.stderr == ""
    assert payload["motionage"] == pyproject["project"]["version"]
    assert isinstance(payload["python"], str)
    assert isinstance(payload["platform"], str)
    assert set(payload["dependencies"]) == set(DOCTOR_DEPENDENCIES)
    for version in payload["dependencies"].values():
        assert isinstance(version, str)
        assert version


def test_motionage_doctor_can_write_text_output_file(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "doctor.txt"

    result = subprocess.run(
        ["uv", "run", "motionage", "doctor", "--output", str(output_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = output_path.read_text(encoding="utf-8")
    assert result.stdout == ""
    assert result.stderr == ""
    assert report.startswith("MotionAge environment report\n")
    assert "dependencies:\n" in report
    assert report.endswith("\n")


def test_motionage_doctor_can_write_json_output_file(tmp_path: Path) -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    output_path = tmp_path / "reports" / "doctor.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage",
            "doctor",
            "--json",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.stdout == ""
    assert result.stderr == ""
    assert payload["motionage"] == pyproject["project"]["version"]
    assert set(payload["dependencies"]) == set(DOCTOR_DEPENDENCIES)
