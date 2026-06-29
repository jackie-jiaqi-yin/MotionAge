from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from motionage.config import (
    apply_overrides,
    load_yaml_config,
    public_config_diff,
    public_config_rows,
    public_config_snapshot,
    resolve_config,
    save_resolved_config,
)


class _Trial:
    def suggest_categorical(self, name: str, choices: list[str]) -> str:
        assert name == "model.type"
        return choices[-1]

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        assert name == "model.hidden_size"
        assert low == 64
        assert high == 256
        assert step == 64
        assert log is False
        return 128


def test_load_yaml_config_inherits_anchor_sections_and_resolves_checkpoint(tmp_path: Path) -> None:
    anchor_dir = tmp_path / "anchor_run"
    checkpoint_path = anchor_dir / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"")
    _write_yaml(
        anchor_dir / "config.yaml",
        {
            "windowing": {"seq_len": 2016, "stride_ratio": 0.5},
            "model": {"hidden_size": 256, "num_layers": 3, "dropout": 0.21},
        },
    )

    config_path = tmp_path / "configs" / "cfg.yaml"
    _write_yaml(
        config_path,
        {
            "experiment": {
                "init_checkpoint": "../anchor_run/checkpoints/best_model.pt",
                "inherit_from_anchor": {
                    "windowing": True,
                    "model_keys": ["hidden_size", "num_layers", "dropout"],
                },
            },
            "model": {"type": "gru_binary"},
        },
    )

    loaded = load_yaml_config(config_path)

    assert loaded["experiment"]["init_checkpoint"] == str(checkpoint_path.resolve())
    assert loaded["windowing"] == {"seq_len": 2016, "stride_ratio": 0.5}
    assert loaded["model"] == {
        "type": "gru_binary",
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.21,
    }


def test_load_yaml_config_raises_on_anchor_inheritance_conflict(tmp_path: Path) -> None:
    anchor_dir = tmp_path / "anchor_run"
    checkpoint_path = anchor_dir / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"")
    _write_yaml(anchor_dir / "config.yaml", {"windowing": {"seq_len": 2016}})

    config_path = tmp_path / "cfg.yaml"
    _write_yaml(
        config_path,
        {
            "experiment": {
                "init_checkpoint": str(checkpoint_path),
                "inherit_from_anchor": {"windowing": True},
            },
            "windowing": {"seq_len": 1008},
        },
    )

    with pytest.raises(ValueError, match="conflict"):
        load_yaml_config(config_path)


def test_resolve_config_uses_defaults_or_trial_suggestions() -> None:
    raw = {
        "model": {
            "type": {"default": "gru", "search": {"type": "categorical", "choices": ["gru", "lstm"]}},
            "hidden_size": {"default": 64, "search": {"type": "int", "low": 64, "high": 256, "step": 64}},
        },
        "data": {"feature_columns": ["intensity_mean"]},
    }

    assert resolve_config(raw) == {
        "model": {"type": "gru", "hidden_size": 64},
        "data": {"feature_columns": ["intensity_mean"]},
    }
    assert resolve_config(raw, trial=_Trial())["model"] == {"type": "lstm", "hidden_size": 128}


def test_apply_overrides_parses_nested_scalar_values() -> None:
    config = {"model": {"hidden_size": 64}, "training": {"use_amp": False}}

    resolved = apply_overrides(
        config,
        ["model.hidden_size=128", "training.use_amp=true", "experiment.name=smoke"],
    )

    assert resolved == {
        "model": {"hidden_size": 128},
        "training": {"use_amp": True},
        "experiment": {"name": "smoke"},
    }
    assert config["model"]["hidden_size"] == 64


def test_public_config_snapshot_redacts_path_like_values_by_default() -> None:
    config = {
        "experiment": {
            "name": "mortality_cv_primary",
            "init_checkpoint": "local-checkpoints/fold0.pt",
            "output_dir": "local-runs/fold0",
        },
        "data": {
            "input_path": "local-data/features.parquet",
            "feature_columns": ["intensity_mean"],
        },
        "model": {"type": "transformer_covariates_binary", "d_model": 64},
        "windowing": {"seq_len": 2016, "stride_ratio": 0.5},
        "mapping": {
            "fit_partitions": ["train"],
            "clip_eps": 0.0001,
            "weighted_fit": True,
            "clamp_output_to_fit_age_range": False,
        },
    }

    snapshot = public_config_snapshot(config)

    assert snapshot == {
        "experiment": {
            "name": "mortality_cv_primary",
            "init_checkpoint": "<redacted>",
            "output_dir": "<redacted>",
        },
        "data": {
            "input_path": "<redacted>",
            "feature_columns": ["intensity_mean"],
        },
        "model": {"type": "transformer_covariates_binary", "d_model": 64},
        "windowing": {"seq_len": 2016, "stride_ratio": 0.5},
        "mapping": {
            "fit_partitions": ["train"],
            "clip_eps": 0.0001,
            "weighted_fit": True,
            "clamp_output_to_fit_age_range": False,
        },
    }


def test_public_config_snapshot_preserves_non_path_profile_keys() -> None:
    config = {
        "report": {
            "profile_label": "high acceleration",
            "artifact_uri": "local-artifacts/table1.csv",
        }
    }

    snapshot = public_config_snapshot(config)

    assert snapshot == {
        "report": {
            "profile_label": "high acceleration",
            "artifact_uri": "<redacted>",
        }
    }


def test_public_config_diff_reports_only_public_reproducibility_changes() -> None:
    reference = {
        "experiment": {
            "name": "mortality_cv_primary",
            "output_dir": "local-runs/lstm",
        },
        "data": {
            "source_file": "local-data/features.parquet",
            "feature_columns": ["intensity_mean", "MIMS"],
        },
        "model": {
            "type": "lstm_covariates_binary",
            "hidden_size": 128,
        },
        "windowing": {"coverage_threshold": 0.5},
    }
    candidate = {
        "experiment": {
            "name": "mortality_cv_primary",
            "output_dir": "local-runs/transformer",
        },
        "data": {
            "source_file": "local-artifacts/approved_features.parquet",
            "feature_columns": ["intensity_mean", "MIMS"],
        },
        "model": {
            "type": "transformer_covariates_binary",
            "hidden_size": 128,
            "num_attention_heads": 4,
        },
        "windowing": {"coverage_threshold": 0.75},
    }

    diff = public_config_diff(reference, candidate)

    assert diff == [
        {"key": "model.num_attention_heads", "reference_value": None, "candidate_value": 4},
        {
            "key": "model.type",
            "reference_value": "lstm_covariates_binary",
            "candidate_value": "transformer_covariates_binary",
        },
        {"key": "windowing.coverage_threshold", "reference_value": 0.5, "candidate_value": 0.75},
    ]
    assert "local-runs" not in str(diff)
    assert "local-artifacts" not in str(diff)


def test_public_config_rows_flatten_reproducibility_knobs_without_paths() -> None:
    config = {
        "experiment": {
            "name": "mortality_cv_primary",
            "output_dir": "local-runs/primary",
            "init_checkpoint": "local-checkpoints/fold0.pt",
        },
        "data": {
            "input_path": "local-data/features.parquet",
            "feature_columns": ["intensity_mean", "MIMS"],
        },
        "model": {
            "type": "lstm_covariates_binary",
            "hidden_size": 128,
        },
        "mapping": {
            "fit_partitions": ["train"],
            "clip_eps": 0.0001,
            "weighted_fit": True,
            "clamp_output_to_fit_age_range": False,
        },
    }

    rows = public_config_rows(config)

    assert rows == [
        {"key": "data.feature_columns", "value": ["intensity_mean", "MIMS"]},
        {"key": "experiment.name", "value": "mortality_cv_primary"},
        {"key": "mapping.clamp_output_to_fit_age_range", "value": False},
        {"key": "mapping.clip_eps", "value": 0.0001},
        {"key": "mapping.fit_partitions", "value": ["train"]},
        {"key": "mapping.weighted_fit", "value": True},
        {"key": "model.hidden_size", "value": 128},
        {"key": "model.type", "value": "lstm_covariates_binary"},
    ]
    assert "local-runs" not in str(rows)
    assert "local-checkpoints" not in str(rows)
    assert "local-data" not in str(rows)


def test_save_resolved_config_writes_yaml(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "config.yaml"

    save_resolved_config({"model": {"type": "gru"}}, output_path)

    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == {"model": {"type": "gru"}}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
