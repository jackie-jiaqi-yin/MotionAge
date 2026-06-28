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
    assert by_id["gru_level1_latefusion"].source_model_type == "gru_covariates_binary"
    assert by_id["gru_level1_latefusion"].prediction_mode == "late_fusion"
    assert by_id["lstm_level1_residual"].source_model_type == "lstm_covariates_binary"
    assert by_id["lstm_level1_residual"].prediction_mode == "residual"
    assert by_id["transformer_level1_residual"].source_model_type == "transformer_covariates_binary"


def test_validate_paper_model_manifest_rejects_missing_family_or_type_mismatch(
    tmp_path: Path,
) -> None:
    assert hasattr(motionage, "validate_paper_model_manifest")

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
) -> Path:
    config_dir = tmp_path / "configs" / "paper"
    config_dir.mkdir(parents=True)
    for family in {model["family"] for model in models}:
        _write_yaml(
            config_dir / f"{family}_primary.yaml",
            {
                "task": {"type": "binary_classification"},
                "model": {"type": f"{family}_binary"},
            },
        )

    manifest = config_dir / "manifest.yaml"
    _write_yaml(manifest, {"models": models, "not_ready_models": []})
    return manifest


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
