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
    assert payload["analysis_template"] == {
        "analysis_name": "motionage_primary_60m",
        "output_dir": "outputs/motionage_primary_60m",
        "participant_predictions_dir": "outputs/mortality_cv_primary_60m",
        "id_column": "SEQN",
        "age_column": "RIDAGEYR",
        "sex_column": "RIAGENDR",
        "probability_column": "probability",
        "target_column": "mortstat",
        "probability_transform": "logit",
        "age_bin_column": "age_bin",
        "fit_partitions": ["train"],
        "strata": ["sex"],
        "clip_eps": 0.0001,
        "weighted_fit": True,
        "clamp_output_to_fit_age_range": False,
        "min_stratum_participants": 100,
        "outputs": {
            "participant_scores": "outputs/motionage_primary_60m/participant_scores.csv",
            "mapping_parameters": "outputs/motionage_primary_60m/mapping_parameters.csv",
            "evaluation_tables": "outputs/motionage_primary_60m/evaluation_tables",
        },
    }
    assert payload["output_filters"] == {"families": [], "model_ids": []}
    assert payload["validation_requirements"] == {
        "required_families": ["gru", "lstm", "transformer"]
    }
    assert payload["public_boundary"] == {
        "contains_raw_data": False,
        "contains_participant_level_rows": False,
        "contains_exact_split_ids": False,
        "contains_trained_checkpoints": False,
        "contains_private_notes": False,
        "contains_public_config_metadata": True,
    }
    assert payload["manifest_readiness"] == {
        "ready_model_count": 10,
        "not_ready_model_count": 0,
        "total_model_count": 10,
        "all_models_ready": True,
    }
    assert payload["model_summary"] == {
        "families": {"gru": 4, "lstm": 3, "transformer": 3},
        "prediction_modes": {"late_fusion": 4, "none": 3, "residual": 3},
        "covariates": {"disabled": 3, "enabled": 7},
        "covariate_levels": {"1": 6, "age": 1, "none": 3},
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
        "architecture": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.117,
            "hour_emb_dim": 8,
            "day_emb_dim": 2,
        },
        "covariates_enabled": False,
        "covariate_levels": [],
        "num_numeric_features": None,
        "selection_metric": "auprc",
        "seq_len": 1008,
        "stride_ratio": 1.0,
        "max_epochs": 80,
        "batch_size": 128,
        "learning_rate": 0.001,
    }


def test_validate_paper_models_cli_can_emit_summary_only_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--json",
            "--summary-only",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["study"]["study_id"] == "mortality_cv_primary_60m"
    assert payload["output_filters"] == {"families": [], "model_ids": []}
    assert payload["validation_requirements"] == {
        "required_families": ["gru", "lstm", "transformer"]
    }
    assert payload["manifest_readiness"] == {
        "ready_model_count": 10,
        "not_ready_model_count": 0,
        "total_model_count": 10,
        "all_models_ready": True,
    }
    assert payload["model_summary"] == {
        "families": {"gru": 4, "lstm": 3, "transformer": 3},
        "prediction_modes": {"late_fusion": 4, "none": 3, "residual": 3},
        "covariates": {"disabled": 3, "enabled": 7},
        "covariate_levels": {"1": 6, "age": 1, "none": 3},
    }
    assert payload["model_count"] == 10
    assert payload["family_counts"] == {"gru": 4, "lstm": 3, "transformer": 3}
    assert "model_ids_by_family" not in payload
    assert "models" not in payload


def test_validate_paper_models_cli_can_emit_markdown_table(capsys: pytest.CaptureFixture[str]) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        ["--markdown", str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert captured.out.splitlines()[:5] == [
        "## Study",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| study_id | mortality_cv_primary_60m |",
    ]
    assert "## Models" in captured.out
    assert "| n_folds | 5 |" in captured.out
    assert "| analysis_template_path | configs/paper/motionage_analysis.yaml |" in captured.out
    assert "| official_feature_set | motionage_accel |" in captured.out
    assert "## MotionAge Analysis Template" in captured.out
    assert "| analysis_name | motionage_primary_60m |" in captured.out
    assert "| fit_partitions | train |" in captured.out
    assert "| strata | sex |" in captured.out
    assert "| output:participant_scores | outputs/motionage_primary_60m/participant_scores.csv |" in captured.out
    assert "## Public Boundary" in captured.out
    assert "| contains_raw_data | false |" in captured.out
    assert "| contains_participant_level_rows | false |" in captured.out
    assert "| contains_exact_split_ids | false |" in captured.out
    assert "| contains_trained_checkpoints | false |" in captured.out
    assert "| contains_private_notes | false |" in captured.out
    assert "| contains_public_config_metadata | true |" in captured.out
    assert "## Validation Requirements" in captured.out
    assert "| Requirement | Values |" in captured.out
    assert "| required_families | gru, lstm, transformer |" in captured.out
    assert "## Manifest Readiness" in captured.out
    assert "| ready_model_count | 10 |" in captured.out
    assert "| not_ready_model_count | 0 |" in captured.out
    assert "| total_model_count | 10 |" in captured.out
    assert "| all_models_ready | true |" in captured.out
    assert "## Model Summary" in captured.out
    assert "| Metric | Value | Count |" in captured.out
    assert "| family | gru | 4 |" in captured.out
    assert "| prediction_mode | late_fusion | 4 |" in captured.out
    assert "| prediction_mode | none | 3 |" in captured.out
    assert "| covariates | enabled | 7 |" in captured.out
    assert "| covariate_level | 1 | 6 |" in captured.out
    assert captured.out.index("## Output Filters") < captured.out.index(
        "## MotionAge Analysis Template"
    )
    assert captured.out.index("## MotionAge Analysis Template") < captured.out.index(
        "## Public Boundary"
    )
    assert captured.out.index("## Public Boundary") < captured.out.index(
        "## Validation Requirements"
    )
    assert captured.out.index("## Validation Requirements") < captured.out.index(
        "## Manifest Readiness"
    )
    assert captured.out.index("## Manifest Readiness") < captured.out.index(
        "## Model Summary"
    )
    assert captured.out.index("## Model Summary") < captured.out.index("## Models")
    assert "\n".join(
        [
            "| Family | Model ID | Model type | Prediction mode | Architecture | Covariates | "
            "Levels | Numeric features | Seq len | Stride | Epochs | Batch | LR | "
            "Selection | Source config |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- |",
        ]
    ) in captured.out
    assert (
        "| gru | gru_fitbit_only | gru_binary | - | hidden_size=64; num_layers=2; "
        "dropout=0.117; hour_emb_dim=8; day_emb_dim=2 | no | - | - | 1008 | 1.0 | "
        "80 | 128 | 0.001 | auprc | configs/paper/gru_fitbit_only_60m.yaml |"
        in captured.out
    )
    assert (
        "| lstm | lstm_level1_residual | lstm_covariates_binary | residual | hidden_size=64; "
        "num_layers=2; dropout=0.15; hour_emb_dim=8; day_emb_dim=4 | yes | 1 | 8 | "
        "1008 | 0.5 | 80 | 128 | 0.001 | auprc | "
        "configs/paper/lstm_level1_residual_60m.yaml |"
        in captured.out
    )
    assert (
        "| transformer | transformer_level1_latefusion | transformer_covariates_binary | "
        "late_fusion | d_model=96; nhead=4; num_layers=3; dim_feedforward=128; "
        "dropout=0.188; hour_emb_dim=16; day_emb_dim=8; intensity_proj_dim=16; "
        "max_seq_len=1008 | yes | 1 | 8 | 576 | 1.0 | 80 | 64 | 0.00039 | auprc | "
        "configs/paper/transformer_level1_latefusion_60m.yaml |"
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
    assert "| study_id | mortality_cv_primary_60m |" in captured.out
    assert "| task | mortality_60m |" in captured.out
    assert "## Models" in captured.out
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
    assert payload["output_filters"] == {
        "families": ["transformer", "lstm"],
        "model_ids": [],
    }
    assert payload["model_summary"] == {
        "families": {"lstm": 3, "transformer": 3},
        "prediction_modes": {"late_fusion": 2, "none": 2, "residual": 2},
        "covariates": {"disabled": 2, "enabled": 4},
        "covariate_levels": {"1": 4, "none": 2},
    }
    assert payload["family_counts"] == {"lstm": 3, "transformer": 3}
    assert set(payload["model_ids_by_family"]) == {"lstm", "transformer"}
    assert {model["family"] for model in payload["models"]} == {"lstm", "transformer"}


def test_validate_paper_models_cli_reports_custom_required_families_in_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--json",
            "--required-family",
            "Transformer",
            "--required-family",
            "GRU",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["validation_requirements"] == {
        "required_families": ["transformer", "gru"]
    }


def test_validate_paper_models_cli_can_filter_json_by_model_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--json",
            "--model-id",
            "lstm_level1_residual",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["study"]["study_id"] == "mortality_cv_primary_60m"
    assert payload["output_filters"] == {
        "families": [],
        "model_ids": ["lstm_level1_residual"],
    }
    assert payload["model_count"] == 1
    assert payload["family_counts"] == {"lstm": 1}
    assert payload["model_ids_by_family"] == {"lstm": ["lstm_level1_residual"]}
    assert payload["models"][0]["covariates_enabled"] is True
    assert payload["models"][0]["covariate_levels"] == ["1"]
    assert payload["models"][0]["num_numeric_features"] == 8
    assert payload["models"][0]["architecture"] == {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "hour_emb_dim": 8,
        "day_emb_dim": 4,
    }
    assert (
        payload["models"][0]["source_config_path"]
        == "configs/paper/lstm_level1_residual_60m.yaml"
    )


def test_validate_paper_models_cli_can_filter_markdown_by_family_and_model_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--markdown",
            "--family",
            "transformer",
            "--model-id",
            "transformer_level1_residual",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert "| study_id | mortality_cv_primary_60m |" in captured.out
    assert "## Output Filters" in captured.out
    assert "## Model Summary" in captured.out
    assert "| families | transformer |" in captured.out
    assert "| model_ids | transformer_level1_residual |" in captured.out
    assert "| family | transformer | 1 |" in captured.out
    assert "| prediction_mode | residual | 1 |" in captured.out
    assert "| covariates | enabled | 1 |" in captured.out
    assert "| covariate_level | 1 | 1 |" in captured.out
    assert "transformer_level1_residual" in captured.out
    assert "transformer_fitbit_only" not in captured.out
    assert "lstm_level1_residual" not in captured.out


def test_validate_paper_models_cli_can_emit_summary_only_markdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--markdown",
            "--summary-only",
            "--model-id",
            "transformer_level1_residual",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert "## Study" in captured.out
    assert "## Output Filters" in captured.out
    assert "## Validation Requirements" in captured.out
    assert "## Model Summary" in captured.out
    assert "## Models" not in captured.out
    assert "| Family | Model ID |" not in captured.out
    assert "| model_ids | transformer_level1_residual |" in captured.out
    assert "| family | transformer | 1 |" in captured.out
    assert "| prediction_mode | residual | 1 |" in captured.out
    assert "| covariates | enabled | 1 |" in captured.out
    assert "| covariate_level | 1 | 1 |" in captured.out


def test_validate_paper_models_cli_returns_error_for_unknown_model_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--model-id",
            "missing_model",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "Unknown paper model_id filters: ['missing_model']" in captured.err


def test_validate_paper_models_cli_returns_error_for_empty_family_model_intersection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from motionage.cli import validate_paper_models_main

    status = validate_paper_models_main(
        [
            "--family",
            "lstm",
            "--model-id",
            "transformer_fitbit_only",
            str(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "No paper model entries matched output filters" in captured.err


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
        "analysis_template_path: configs/paper/motionage_analysis.yaml",
        (
            "public_boundary: raw_data=false participant_level_rows=false "
            "exact_split_ids=false trained_checkpoints=false private_notes=false "
            "public_config_metadata=true"
        ),
        "all_models_ready: true",
        "ready_model_configs: 10",
        "not_ready_model_placeholders: 0",
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
    assert markdown.startswith("## Study\n\n| Field | Value |\n")
    assert "| study_id | mortality_cv_primary_60m |" in markdown
    assert "## Validation Requirements\n\n| Requirement | Values |\n" in markdown
    assert "## Model Summary\n\n| Metric | Value | Count |\n" in markdown
    assert (
        "## Models\n\n| Family | Model ID | Model type | Prediction mode | Architecture | "
        "Covariates | Levels | Numeric features | Seq len | Stride | Epochs | Batch | "
        "LR | Selection | Source config |"
    ) in markdown
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
        _source_config("gru_binary"),
    )
    _write_yaml(
        config_dir / "motionage_analysis.yaml",
        _motionage_template(),
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


def test_motionage_console_script_is_registered() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["motionage"] == "motionage.cli:main"


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


def test_motionage_console_script_validate_paper_models_summary_only_json() -> None:
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage",
            "validate-paper-models",
            "--json",
            "--summary-only",
            "configs/paper/mortality_cv_primary_60m.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.stderr == ""
    assert payload["model_count"] == 10
    assert payload["family_counts"] == {"gru": 4, "lstm": 3, "transformer": 3}
    assert "model_ids_by_family" not in payload
    assert "models" not in payload


def test_motionage_console_script_reports_version() -> None:
    import subprocess

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    result = subprocess.run(
        ["uv", "run", "motionage", "--version"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == f"motionage {pyproject['project']['version']}\n"
    assert result.stderr == ""


def test_validate_paper_models_console_script_reports_version() -> None:
    import subprocess

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    result = subprocess.run(
        ["uv", "run", "motionage-validate-paper-models", "--version"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == f"motionage {pyproject['project']['version']}\n"
    assert result.stderr == ""


def test_validate_paper_models_console_script_summary_only_json_output_file(
    tmp_path: Path,
) -> None:
    import subprocess

    output_path = tmp_path / "manifest_summary.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage-validate-paper-models",
            "--json",
            "--summary-only",
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
    assert payload["family_counts"] == {"gru": 4, "lstm": 3, "transformer": 3}
    assert "model_ids_by_family" not in payload
    assert "models" not in payload


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

    assert result.stdout.startswith("## Study\n\n| Field | Value |")
    assert "| study_id | mortality_cv_primary_60m |" in result.stdout
    assert "## Models" in result.stdout
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


def test_validate_paper_models_console_script_output_file_with_model_id_filter(tmp_path: Path) -> None:
    import subprocess

    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "motionage-validate-paper-models",
            "--json",
            "--model-id",
            "gru_level1_latefusion",
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
    assert payload["model_count"] == 1
    assert payload["family_counts"] == {"gru": 1}
    assert payload["models"][0]["model_id"] == "gru_level1_latefusion"


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _source_config(model_type: str) -> dict[str, object]:
    if model_type.startswith("transformer"):
        model: dict[str, object] = {
            "type": model_type,
            "d_model": 96,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 128,
            "dropout": 0.188,
            "hour_emb_dim": 16,
            "day_emb_dim": 8,
            "intensity_proj_dim": 16,
            "max_seq_len": 1008,
        }
    else:
        model = {
            "type": model_type,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "hour_emb_dim": 8,
            "day_emb_dim": 2,
        }
    return {
        "task": {
            "type": "binary_classification",
            "selection_metric": "auprc",
        },
        "model": model,
        "windowing": {
            "seq_len": 1008,
            "stride_ratio": 1.0,
        },
        "training": {
            "max_epochs": 80,
            "batch_size": 128,
            "learning_rate": 0.001,
        },
    }


def _motionage_template() -> dict[str, object]:
    return {
        "analysis": {
            "name": "synthetic_motionage_analysis",
            "output_dir": "outputs/synthetic_motionage",
            "participant_predictions_dir": "outputs/synthetic_predictions",
            "id_column": "SEQN",
            "age_column": "RIDAGEYR",
            "sex_column": "RIAGENDR",
            "probability_column": "probability",
            "target_column": "mortstat",
        },
        "mapping": {
            "probability_transform": "logit",
            "age_bin_column": "age_bin",
            "fit_partitions": ["train"],
            "strata": ["sex"],
            "clip_eps": 0.0001,
            "weighted_fit": True,
            "clamp_output_to_fit_age_range": False,
            "min_stratum_participants": 10,
        },
        "outputs": {
            "participant_scores": "outputs/synthetic_motionage/participant_scores.csv",
            "mapping_parameters": "outputs/synthetic_motionage/mapping_parameters.csv",
            "evaluation_tables": "outputs/synthetic_motionage/evaluation_tables",
        },
    }
