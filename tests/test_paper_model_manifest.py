from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import motionage

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_DIR = REPO_ROOT / "configs" / "paper"


def test_validate_paper_model_manifest_covers_all_paper_visible_families() -> None:
    assert hasattr(motionage, "validate_paper_model_manifest")

    entries = motionage.validate_paper_model_manifest(
        PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"
    )

    assert len(entries) == 10
    assert {entry.family for entry in entries} == {"gru", "lstm", "transformer"}
    assert [entry.model_id for entry in entries if entry.family == "lstm"] == [
        "lstm_fitbit_only",
        "lstm_level1_latefusion",
        "lstm_level1_residual",
    ]
    assert [entry.model_id for entry in entries if entry.family == "transformer"] == [
        "transformer_fitbit_only",
        "transformer_level1_latefusion",
        "transformer_level1_residual",
    ]
    assert all(entry.task_type == "binary_classification" for entry in entries)


def test_load_paper_model_manifest_records_source_config_details() -> None:
    assert hasattr(motionage, "load_paper_model_manifest")

    entries = motionage.load_paper_model_manifest(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")
    by_id = {entry.model_id: entry for entry in entries}

    assert by_id["gru_fitbit_only"].source_model_type == "gru_binary"
    assert by_id["gru_fitbit_only"].prediction_mode is None
    assert by_id["gru_fitbit_only"].architecture == {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.117,
        "hour_emb_dim": 8,
        "day_emb_dim": 2,
    }
    assert by_id["gru_fitbit_only"].covariates_enabled is False
    assert by_id["gru_fitbit_only"].covariate_levels == ()
    assert by_id["gru_fitbit_only"].num_numeric_features is None
    assert by_id["gru_fitbit_only"].selection_metric == "auprc"
    assert by_id["gru_fitbit_only"].seq_len == 1008
    assert by_id["gru_fitbit_only"].stride_ratio == 1.0
    assert by_id["gru_fitbit_only"].max_epochs == 80
    assert by_id["gru_fitbit_only"].batch_size == 128
    assert by_id["gru_fitbit_only"].learning_rate == 0.001
    assert by_id["gru_age_only_latefusion"].covariates_enabled is True
    assert by_id["gru_age_only_latefusion"].covariate_levels == ("age",)
    assert by_id["gru_age_only_latefusion"].num_numeric_features == 1
    assert by_id["gru_level1_latefusion"].source_model_type == "gru_covariates_binary"
    assert by_id["gru_level1_latefusion"].prediction_mode == "late_fusion"
    assert by_id["gru_level1_latefusion"].covariate_levels == ("1",)
    assert by_id["gru_level1_latefusion"].num_numeric_features == 8
    assert by_id["lstm_level1_residual"].source_model_type == "lstm_covariates_binary"
    assert by_id["lstm_level1_residual"].prediction_mode == "residual"
    assert by_id["lstm_level1_residual"].architecture == {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "hour_emb_dim": 8,
        "day_emb_dim": 4,
    }
    assert by_id["lstm_level1_residual"].covariates_enabled is True
    assert by_id["lstm_level1_residual"].covariate_levels == ("1",)
    assert by_id["lstm_level1_residual"].seq_len == 1008
    assert by_id["lstm_level1_residual"].stride_ratio == 0.5
    assert by_id["transformer_level1_residual"].source_model_type == "transformer_covariates_binary"
    assert by_id["transformer_level1_residual"].architecture == {
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
    assert by_id["transformer_level1_residual"].covariate_levels == ("1",)
    assert by_id["transformer_level1_residual"].num_numeric_features == 8
    assert by_id["transformer_level1_residual"].seq_len == 576
    assert by_id["transformer_level1_residual"].batch_size == 64
    assert by_id["transformer_level1_residual"].learning_rate == 0.00039


def test_load_paper_study_manifest_records_public_study_metadata() -> None:
    assert hasattr(motionage, "PaperStudyManifest")
    assert hasattr(motionage, "load_paper_study_manifest")

    study = motionage.load_paper_study_manifest(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")

    assert study.study_id == "mortality_cv_primary_60m"
    assert study.task == "mortality_60m"
    assert study.n_folds == 5
    assert study.fold_root == "data/processed/splits/mortstat_60m_cv_seed42"
    assert study.training_seed == 42
    assert study.analysis_template_path == "configs/paper/motionage_analysis.yaml"
    assert study.official_feature_set == "motionage_accel"


def test_validate_paper_model_manifest_rejects_missing_family_or_type_mismatch(
    tmp_path: Path,
) -> None:
    assert hasattr(motionage, "validate_paper_model_manifest")

    config_dir = tmp_path / "configs" / "paper"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "gru_only.yaml",
        _source_config("gru_binary"),
    )
    _write_yaml(
        config_dir / "motionage_analysis.yaml",
        {"analysis": {"name": "synthetic_motionage_analysis"}},
    )
    manifest = config_dir / "manifest.yaml"
    _write_yaml(
        manifest,
        {
            "study": _default_study_metadata(),
            "models": [
                {
                    "model_id": "wrong_lstm",
                    "family": "lstm",
                    "source_config_path": "configs/paper/gru_only.yaml",
                }
            ],
            "not_ready_models": [],
        },
    )

    with pytest.raises(ValueError, match="does not match source model.type"):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru", "lstm"))

    _write_yaml(
        manifest,
        {
            "study": _default_study_metadata(),
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
    with pytest.raises(ValueError, match="Missing required model families"):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru", "lstm"))


def test_validate_paper_model_manifest_rejects_missing_study_metadata(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest_with_models(
        tmp_path,
        [
            {
                "model_id": "gru_primary",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            }
        ],
        include_study=False,
    )

    with pytest.raises(ValueError, match="Paper model manifest must define a study mapping"):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru",))


def test_validate_paper_model_manifest_rejects_invalid_study_fold_count(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest_with_models(
        tmp_path,
        [
            {
                "model_id": "gru_primary",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            }
        ],
        study_overrides={"n_folds": 0},
    )

    with pytest.raises(ValueError, match="study.n_folds must be a positive integer"):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru",))


def test_validate_paper_model_manifest_rejects_absolute_study_template_path(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest_with_models(
        tmp_path,
        [
            {
                "model_id": "gru_primary",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            }
        ],
        study_overrides={"analysis_template_path": "/tmp/private_template.yaml"},
    )

    with pytest.raises(
        ValueError,
        match="study.analysis_template_path must be repository-relative",
    ):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru",))


def test_validate_paper_model_manifest_rejects_missing_study_template_path(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest_with_models(
        tmp_path,
        [
            {
                "model_id": "gru_primary",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            }
        ],
        study_overrides={"analysis_template_path": "configs/paper/missing_analysis.yaml"},
    )

    with pytest.raises(FileNotFoundError, match="Study analysis template not found"):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru",))


def test_validate_paper_model_manifest_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    manifest = _write_manifest_with_models(
        tmp_path,
        [
            {
                "model_id": "duplicate_model",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            },
            {
                "model_id": "duplicate_model",
                "family": "lstm",
                "source_config_path": "configs/paper/lstm_primary.yaml",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate paper model_id values: \\['duplicate_model'\\]",
    ):
        motionage.validate_paper_model_manifest(
            manifest,
            required_families=("gru", "lstm"),
        )


def test_validate_paper_model_manifest_rejects_duplicate_source_config_paths(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest_with_models(
        tmp_path,
        [
            {
                "model_id": "gru_primary",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            },
            {
                "model_id": "gru_duplicate",
                "family": "gru",
                "source_config_path": "configs/paper/gru_primary.yaml",
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate paper source_config_path values"):
        motionage.validate_paper_model_manifest(manifest, required_families=("gru",))


def _write_manifest_with_models(
    tmp_path: Path,
    models: list[dict[str, str]],
    *,
    include_study: bool = True,
    study_overrides: dict[str, object] | None = None,
) -> Path:
    config_dir = tmp_path / "configs" / "paper"
    config_dir.mkdir(parents=True)
    _write_yaml(
        config_dir / "motionage_analysis.yaml",
        {"analysis": {"name": "synthetic_motionage_analysis"}},
    )
    for family in {model["family"] for model in models}:
        _write_yaml(
            config_dir / f"{family}_primary.yaml",
            _source_config(f"{family}_binary"),
        )

    manifest = config_dir / "manifest.yaml"
    payload: dict[str, object] = {"models": models, "not_ready_models": []}
    if include_study:
        payload["study"] = _default_study_metadata() | (study_overrides or {})
    _write_yaml(manifest, payload)
    return manifest


def _default_study_metadata() -> dict[str, object]:
    return {
        "study_id": "synthetic_manifest",
        "task": "mortality_60m",
        "n_folds": 5,
        "fold_root": "data/processed/splits/synthetic_cv",
        "training_seed": 42,
        "analysis_template_path": "configs/paper/motionage_analysis.yaml",
        "official_feature_set": "motionage_accel",
    }


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


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
