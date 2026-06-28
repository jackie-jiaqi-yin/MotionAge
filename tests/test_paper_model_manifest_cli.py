from __future__ import annotations

import json
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


def test_validate_paper_models_cli_can_emit_json_summary(capsys: pytest.CaptureFixture[str]) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        ["--json", str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["manifest_path"].endswith("mortality_cv_primary_60m.yaml")
    assert payload["study"] == {
        "study_id": "mortality_cv_primary_60m",
        "task": "mortality_60m",
        "n_folds": 5,
        "fold_root": "data/processed/splits/mortstat_60m_cv_seed42",
        "training_seed": 42,
        "analysis_template_path": "configs/paper/motionage_analysis.yaml",
        "official_feature_set": "motionage_accel",
    }
    assert payload["model_count"] == 10
    assert payload["family_counts"] == {"gru": 4, "lstm": 3, "transformer": 3}
    assert payload["model_ids_by_family"]["lstm"] == [
        "lstm_fitbit_only",
        "lstm_level1_latefusion",
        "lstm_level1_residual",
    ]
    assert payload["models"][0] == {
        "model_id": "gru_fitbit_only",
        "family": "gru",
        "source_config_path": "configs/paper/gru_fitbit_only_60m.yaml",
        "source_model_type": "gru_binary",
        "task_type": "binary_classification",
        "prediction_mode": None,
    }


def test_validate_paper_models_cli_can_emit_markdown_table(capsys: pytest.CaptureFixture[str]) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        ["--markdown", str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert captured.out.splitlines()[:2] == [
        "| Family | Model ID | Model type | Prediction mode | Source config |",
        "| --- | --- | --- | --- | --- |",
    ]
    assert (
        "| gru | gru_fitbit_only | gru_binary | - | configs/paper/gru_fitbit_only_60m.yaml |"
        in captured.out
    )
    assert (
        "| lstm | lstm_level1_residual | lstm_covariates_binary | residual | "
        "configs/paper/lstm_level1_residual_60m.yaml |"
        in captured.out
    )
    assert (
        "| transformer | transformer_level1_latefusion | transformer_covariates_binary | "
        "late_fusion | configs/paper/transformer_level1_latefusion_60m.yaml |"
        in captured.out
    )


def test_validate_paper_models_cli_can_filter_markdown_by_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--markdown",
            "--family",
            "lstm",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert "lstm_fitbit_only" in captured.out
    assert "lstm_level1_residual" in captured.out
    assert "gru_fitbit_only" not in captured.out
    assert "transformer_fitbit_only" not in captured.out


def test_validate_paper_models_cli_can_filter_json_by_multiple_families(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--json",
            "--family",
            "Transformer",
            "--family",
            "lstm",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["model_count"] == 6
    assert payload["family_counts"] == {"lstm": 3, "transformer": 3}
    assert set(payload["model_ids_by_family"]) == {"lstm", "transformer"}
    assert {model["family"] for model in payload["models"]} == {"lstm", "transformer"}


def test_validate_paper_models_cli_returns_error_for_unknown_output_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--family",
            "cnn",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "Unknown paper model family filters: ['cnn']" in captured.err


def test_validate_paper_models_cli_rejects_unknown_family_with_matching_filter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--family",
            "lstm",
            "--family",
            "cnn",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "Unknown paper model family filters: ['cnn']" in captured.err


def test_validate_paper_models_cli_can_write_text_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    output_path = tmp_path / "reports" / "manifest.txt"

    status = validate_paper_models_main(
        [
            "--output",
            str(output_path),
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == ""
    assert captured.err == ""
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        f"Validated 10 paper model configs from {PAPER_CONFIG_DIR / 'mortality_cv_primary_60m.yaml'}.",
        "gru: 4",
        "lstm: 3",
        "transformer: 3",
    ]


def test_validate_paper_models_cli_can_write_json_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    output_path = tmp_path / "manifest.json"

    status = validate_paper_models_main(
        [
            "--json",
            "--output",
            str(output_path),
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert status == 0
    assert captured.out == ""
    assert captured.err == ""
    assert payload["family_counts"] == {"gru": 4, "lstm": 3, "transformer": 3}


def test_validate_paper_models_cli_can_write_markdown_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    output_path = tmp_path / "reports" / "manifest.md"

    status = validate_paper_models_main(
        [
            "--markdown",
            "--output",
            str(output_path),
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    markdown = output_path.read_text(encoding="utf-8")
    assert status == 0
    assert captured.out == ""
    assert captured.err == ""
    assert markdown.startswith("| Family | Model ID | Model type | Prediction mode | Source config |\n")
    assert "lstm_level1_residual" in markdown


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
    _write_yaml(
        config_dir / "motionage_analysis.yaml",
        {"analysis": {"name": "synthetic_motionage_analysis"}},
    )
    manifest = config_dir / "manifest.yaml"
    _write_yaml(
        manifest,
        {
            "study": {
                "study_id": "synthetic_manifest",
                "task": "mortality_60m",
                "n_folds": 5,
                "fold_root": "data/processed/splits/synthetic_cv",
                "training_seed": 42,
                "analysis_template_path": "configs/paper/motionage_analysis.yaml",
                "official_feature_set": "motionage_accel",
            },
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


def test_validate_paper_models_console_script_json_output() -> None:
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage-validate-paper-models",
            "--json",
            "configs/paper/mortality_cv_primary_60m.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["model_count"] == 10
    assert payload["family_counts"]["transformer"] == 3


def test_validate_paper_models_console_script_markdown_output() -> None:
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage-validate-paper-models",
            "--markdown",
            "configs/paper/mortality_cv_primary_60m.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.startswith("| Family | Model ID | Model type | Prediction mode | Source config |")
    assert "transformer_level1_residual" in result.stdout


def test_validate_paper_models_console_script_output_file(tmp_path: Path) -> None:
    import subprocess

    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage-validate-paper-models",
            "--json",
            "--output",
            str(output_path),
            "configs/paper/mortality_cv_primary_60m.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.stdout == ""
    assert result.stderr == ""
    assert payload["model_count"] == 10
    assert payload["family_counts"]["transformer"] == 3


def test_validate_paper_models_console_script_output_file_with_family_filter(tmp_path: Path) -> None:
    import subprocess

    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage-validate-paper-models",
            "--json",
            "--family",
            "transformer",
            "--output",
            str(output_path),
            "configs/paper/mortality_cv_primary_60m.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.stdout == ""
    assert result.stderr == ""
    assert payload["model_count"] == 3
    assert payload["family_counts"] == {"transformer": 3}
    assert set(payload["model_ids_by_family"]) == {"transformer"}
    assert payload["study"]["study_id"] == "mortality_cv_primary_60m"
    assert payload["study"]["n_folds"] == 5


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
