from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "examples" / "synthetic" / "make_synthetic_inputs.py"


def test_generate_synthetic_inputs_writes_expected_files(tmp_path: Path) -> None:
    module = _load_generator()

    outputs = module.generate_synthetic_inputs(tmp_path, participants=12, days=2, seed=7)

    assert set(outputs) == {
        "activity_mortstat",
        "covariates",
        "metadata",
        "manifest",
    }
    for path in outputs.values():
        assert path.exists()

    activity = pd.read_parquet(outputs["activity_mortstat"])
    covariates = pd.read_parquet(outputs["covariates"])
    metadata = yaml.safe_load(outputs["metadata"].read_text(encoding="utf-8"))
    manifest = yaml.safe_load(outputs["manifest"].read_text(encoding="utf-8"))

    assert activity["SEQN"].nunique() == 12
    assert len(activity) == 12 * 2 * 24
    assert {"SEQN", "PAXDAY", "PAXHOUR", "intensity_mean", "attention_flag", "mortstat", "permth_int"}.issubset(
        activity.columns
    )
    assert covariates["SEQN"].nunique() == 12
    assert metadata["id_column"] == "SEQN"
    assert metadata["numeric_columns"] == ["RIDAGEYR", "BMXBMI", "SMQ020"]
    assert manifest["participants"] == 12
    assert manifest["days"] == 2


def test_generate_synthetic_inputs_is_deterministic_for_seed(tmp_path: Path) -> None:
    module = _load_generator()

    first = module.generate_synthetic_inputs(tmp_path / "first", participants=8, days=1, seed=11)
    second = module.generate_synthetic_inputs(tmp_path / "second", participants=8, days=1, seed=11)

    first_activity = pd.read_parquet(first["activity_mortstat"])
    second_activity = pd.read_parquet(second["activity_mortstat"])
    first_covariates = pd.read_parquet(first["covariates"])
    second_covariates = pd.read_parquet(second["covariates"])

    pd.testing.assert_frame_equal(first_activity, second_activity)
    pd.testing.assert_frame_equal(first_covariates, second_covariates)


def test_generate_synthetic_inputs_manifest_declares_public_boundary_and_schema(tmp_path: Path) -> None:
    module = _load_generator()

    outputs = module.generate_synthetic_inputs(tmp_path, participants=10, days=3, seed=19)
    activity = pd.read_parquet(outputs["activity_mortstat"])
    covariates = pd.read_parquet(outputs["covariates"])
    manifest = yaml.safe_load(outputs["manifest"].read_text(encoding="utf-8"))

    assert manifest["public_boundary"] == {
        "synthetic": True,
        "contains_real_participants": False,
        "contains_trained_weights": False,
        "safe_for_public_smoke_tests": True,
    }
    assert manifest["row_counts"] == {
        "activity_mortstat": len(activity),
        "covariates": len(covariates),
    }
    assert manifest["columns"]["activity_mortstat"] == activity.columns.tolist()
    assert manifest["columns"]["covariates"] == covariates.columns.tolist()


def _load_generator():
    spec = importlib.util.spec_from_file_location("synthetic_inputs", GENERATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
