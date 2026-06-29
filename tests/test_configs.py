from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_DIR = REPO_ROOT / "configs" / "paper"
SENSITIVITY_CONFIG_DIR = REPO_ROOT / "configs" / "sensitivity"
EXAMPLE_CONFIG_DIR = REPO_ROOT / "configs" / "examples"

EXPECTED_PAPER_CONFIGS = {
    "gru_fitbit_only_60m.yaml",
    "gru_age_only_latefusion_60m.yaml",
    "gru_level1_latefusion_60m.yaml",
    "gru_level1_residual_60m.yaml",
    "transformer_fitbit_only_60m.yaml",
    "transformer_level1_latefusion_60m.yaml",
    "transformer_level1_residual_60m.yaml",
    "lstm_fitbit_only_60m.yaml",
    "lstm_level1_latefusion_60m.yaml",
    "lstm_level1_residual_60m.yaml",
    "mortality_cv_primary_60m.yaml",
    "motionage_analysis.yaml",
}

BANNED_TEXT = (
    "/" + "Users" + "/" + "Placebo",
    "reb" + "uttal",
    "Open" + "Review",
    "submission" + "271",
    "review" + "er",
    "Yi" + "lin",
    "formal" + "_tuning",
    "trial" + "_",
)


def test_paper_config_inventory_is_complete() -> None:
    assert _names(PAPER_CONFIG_DIR) == EXPECTED_PAPER_CONFIGS
    assert _names(SENSITIVITY_CONFIG_DIR) == {"wear_coverage04_60m.yaml"}
    assert _names(EXAMPLE_CONFIG_DIR) == {"synthetic_smoke.yaml"}


def test_model_configs_cover_required_paper_families_and_modes() -> None:
    configs = {path.name: _load_yaml(path) for path in PAPER_CONFIG_DIR.glob("*.yaml")}

    assert configs["gru_fitbit_only_60m.yaml"]["model"]["type"] == "gru_binary"
    assert configs["lstm_fitbit_only_60m.yaml"]["model"]["type"] == "lstm_binary"
    assert configs["transformer_fitbit_only_60m.yaml"]["model"]["type"] == "transformer_binary"

    for family in ("gru", "lstm", "transformer"):
        late = configs[f"{family}_level1_latefusion_60m.yaml"]
        residual = configs[f"{family}_level1_residual_60m.yaml"]
        assert late["model"]["type"] == f"{family}_covariates_binary"
        assert residual["model"]["type"] == f"{family}_covariates_binary"
        assert late["model"]["prediction_mode"] == "late_fusion"
        assert residual["model"]["prediction_mode"] == "residual"
        assert late["data"]["covariates"]["levels"] == [1]
        assert residual["data"]["covariates"]["levels"] == [1]


def test_training_configs_use_public_relative_inputs_and_ignored_outputs() -> None:
    config_paths = [
        *PAPER_CONFIG_DIR.glob("*.yaml"),
        *SENSITIVITY_CONFIG_DIR.glob("*.yaml"),
        *EXAMPLE_CONFIG_DIR.glob("*.yaml"),
    ]
    for path in config_paths:
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_TEXT:
            assert banned not in text

        payload = _load_yaml(path)
        _assert_no_absolute_paths(payload)
        experiment = payload.get("experiment")
        if isinstance(experiment, dict) and "output_dir" in experiment:
            assert str(experiment["output_dir"]).startswith(("outputs/", "experiments/"))
        data = payload.get("data")
        if isinstance(data, dict) and "parquet_path" in data:
            assert str(data["parquet_path"]).startswith(("data/processed/", "examples/"))


def test_mortality_cv_and_motionage_analysis_reference_curated_configs() -> None:
    mortality_cv = _load_yaml(PAPER_CONFIG_DIR / "mortality_cv_primary_60m.yaml")
    model_ids = [row["model_id"] for row in mortality_cv["models"]]

    assert mortality_cv["study"]["task"] == "mortality_60m"
    assert set(model_ids) == {
        "gru_fitbit_only",
        "gru_age_only_latefusion",
        "gru_level1_latefusion",
        "gru_level1_residual",
        "transformer_fitbit_only",
        "transformer_level1_latefusion",
        "transformer_level1_residual",
        "lstm_fitbit_only",
        "lstm_level1_latefusion",
        "lstm_level1_residual",
    }
    for row in mortality_cv["models"]:
        assert row["source_config_path"].startswith("configs/paper/")

    motionage = _load_yaml(PAPER_CONFIG_DIR / "motionage_analysis.yaml")
    assert motionage["mapping"]["fit_partitions"] == ["train"]
    assert motionage["mapping"]["strata"] == ["sex"]
    assert motionage["mapping"]["clip_eps"] == 0.0001
    assert motionage["mapping"]["weighted_fit"] is True


def _names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.yaml")}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_no_absolute_paths(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_paths(child)
    elif isinstance(value, str):
        assert not value.startswith("/")
