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


def test_load_motionage_analysis_template_records_public_mapping_metadata() -> None:
    assert hasattr(motionage, "MotionAgeAnalysisTemplate")
    assert hasattr(motionage, "load_motionage_analysis_template")

    template = motionage.load_motionage_analysis_template(PAPER_CONFIG_DIR / "motionage_analysis.yaml")

    assert template.analysis_name == "motionage_primary_60m"
    assert template.output_dir == "outputs/motionage_primary_60m"
    assert template.participant_predictions_dir == "outputs/mortality_cv_primary_60m"
    assert template.id_column == "SEQN"
    assert template.age_column == "RIDAGEYR"
    assert template.sex_column == "RIAGENDR"
    assert template.probability_column == "probability"
    assert template.target_column == "mortstat"
    assert template.probability_transform == "logit"
    assert template.fit_partitions == ("train",)
    assert template.strata == ("sex",)
    assert template.clip_eps == 0.0001
    assert template.weighted_fit is True
    assert template.clamp_output_to_fit_age_range is False
    assert template.min_stratum_participants == 100
    assert template.outputs == {
        "participant_scores": "outputs/motionage_primary_60m/participant_scores.csv",
        "mapping_parameters": "outputs/motionage_primary_60m/mapping_parameters.csv",
        "evaluation_tables": "outputs/motionage_primary_60m/evaluation_tables",
    }


def test_load_motionage_analysis_template_normalizes_partition_aliases(tmp_path: Path) -> None:
    template_path = tmp_path / "motionage_analysis.yaml"
    _write_yaml(
        template_path,
        _motionage_template(fit_partitions=["train", "val", "validation"]),
    )

    template = motionage.load_motionage_analysis_template(template_path)

    assert template.fit_partitions == ("train", "validation")


def test_load_motionage_analysis_template_rejects_private_absolute_paths(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "motionage_analysis.yaml"
    _write_yaml(
        template_path,
        _motionage_template(
            analysis_overrides={
                "output_dir": "/tmp/motionage",
            }
        ),
    )

    with pytest.raises(ValueError, match="analysis.output_dir must be repository-relative"):
        motionage.load_motionage_analysis_template(template_path)


def test_load_paper_manifest_readiness_counts_ready_and_not_ready_models() -> None:
    assert hasattr(motionage, "PaperManifestReadiness")
    assert hasattr(motionage, "load_paper_manifest_readiness")

    readiness = motionage.load_paper_manifest_readiness(
        PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml"
    )

    assert readiness.ready_model_count == 10
    assert readiness.not_ready_model_count == 0
    assert readiness.total_model_count == 10
    assert readiness.all_models_ready is True


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
        _motionage_template(),
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
        _motionage_template(),
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


def _motionage_template(
    *,
    fit_partitions: list[str] | None = None,
    analysis_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
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
        }
        | (analysis_overrides or {}),
        "mapping": {
            "probability_transform": "logit",
            "age_bin_column": "age_bin",
            "fit_partitions": fit_partitions or ["train"],
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
