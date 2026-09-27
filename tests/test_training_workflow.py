import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from motionage.data.partitioning import make_splits
from motionage.training.cross_validation import run_cross_validation
from motionage.training.data_pipeline import load_partitions, prepare_training_data
from motionage.training.inference import run_prediction
from motionage.training.workflow import run_training


@pytest.fixture
def training_config(tmp_path):
    n, steps = 40, 12
    participants = pd.DataFrame(
        {
            "SEQN": np.arange(n),
            "mortstat": np.arange(n) % 2,
            "RIDAGEYR": 40 + np.arange(n),
            "RIAGENDR": np.arange(n) % 2 + 1,
        }
    )
    participants.to_parquet(tmp_path / "participants.parquet", index=False)
    frame = participants.loc[participants.index.repeat(steps)].reset_index(drop=True)
    frame["PAXN"] = np.tile(np.arange(1, steps + 1), n)
    frame["PAXDAY"] = 1
    frame["PAXHOUR"] = np.tile(np.arange(steps), n)
    frame["intensity_mean"] = frame.RIDAGEYR.astype(float)
    frame["attention_flag"] = 1
    frame.to_parquet(tmp_path / "activity.parquet", index=False)
    return {
        "experiment": {"output_dir": str(tmp_path / "run")},
        "task": {"type": "binary_classification", "selection_metric": "auprc"},
        "data": {
            "parquet_path": str(tmp_path / "activity.parquet"),
            "participants_path": str(tmp_path / "participants.parquet"),
            "id_column": "SEQN",
            "target_column": "mortstat",
            "normalize_intensity": True,
        },
        "split": {
            "mode": "runtime",
            "test_size": 0.2,
            "val_size": 0.25,
            "random_seed": 42,
        },
        "windowing": {"seq_len": 6, "stride_ratio": 1.0, "min_attention_ratio": 0.1},
        "model": {
            "type": "gru_binary",
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "hour_emb_dim": 2,
            "day_emb_dim": 2,
        },
        "training": {
            "max_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.01,
            "eval_batch_size": 128,
        },
        "device": "cpu",
    }


@pytest.mark.parametrize("family", ["gru", "lstm", "transformer"])
@pytest.mark.parametrize("fusion", [None, "late_fusion", "residual"])
def test_training_and_reloaded_prediction(training_config, tmp_path, family, fusion):
    config = training_config
    model = config["model"]
    model["type"] = (
        f"{family}_binary" if fusion is None else f"{family}_covariates_binary"
    )
    if family == "transformer":
        model.update(
            d_model=8, nhead=2, dim_feedforward=16, intensity_proj_dim=4, max_seq_len=12
        )
    if fusion:
        model["prediction_mode"] = fusion
        config["data"]["covariates"] = {
            "enabled": True,
            "parquet_path": str(tmp_path / "participants.parquet"),
            "numeric_columns": ["RIDAGEYR"],
            "categorical_columns": ["RIAGENDR"],
        }
    summary = run_training(config)
    assert summary["fit_mode"] == "validation_selected"
    assert summary["epochs_run"] == 2
    assert summary["threshold_source"] == "validation"
    expected = pd.read_parquet(
        tmp_path / "run/participant_predictions.parquet"
    ).sort_values("SEQN")
    assert len(expected) == 40
    assert expected.probability.between(0, 1).all()
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    assert checkpoint["model_state_dict"]
    result = run_prediction(
        {
            "checkpoint_dir": str(tmp_path / "run"),
            "data": config["data"],
            "output_path": str(tmp_path / "predictions.parquet"),
        }
    )
    assert result["participants"] == 40
    actual = pd.read_parquet(tmp_path / "predictions.parquet").sort_values("SEQN")
    np.testing.assert_allclose(actual.probability, expected.probability, atol=1e-6)


def test_scalers_fit_training_only_and_reject_overlap(training_config, tmp_path):
    config = training_config
    original = pd.read_parquet(config["data"]["parquet_path"])
    partitions = load_partitions(original, config)
    _, before, _ = prepare_training_data(copy.deepcopy(config))
    changed = original.copy()
    changed.loc[changed.SEQN.isin(partitions["test"]), "intensity_mean"] = 100000
    changed.to_parquet(config["data"]["parquet_path"])
    _, after, _ = prepare_training_data(copy.deepcopy(config))
    assert before["intensity"] == after["intensity"]
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for part, ids in partitions.items():
        pd.DataFrame({"SEQN": ids}).to_csv(split_dir / f"{part}_ids.csv", index=False)
    pd.DataFrame({"SEQN": partitions["train"]}).to_csv(
        split_dir / "test_ids.csv", index=False
    )
    config["split"] = {"mode": "precomputed", "precomputed_dir": str(split_dir)}
    with pytest.raises(ValueError, match="overlap"):
        prepare_training_data(config)


def test_fixed_epoch_refit_and_seed_reproducibility(training_config, tmp_path):
    config = training_config
    config["final_refit"] = {"enabled": True, "epochs": 2, "combine_train_val": True}
    summary = run_training(config)
    assert summary["fit_mode"] == "fixed_epochs"
    assert "val" not in summary["metrics"]
    first = pd.read_parquet(tmp_path / "run/participant_predictions.parquet")
    config["experiment"]["output_dir"] = str(tmp_path / "repeat")
    run_training(config)
    pd.testing.assert_frame_equal(
        first, pd.read_parquet(tmp_path / "repeat/participant_predictions.parquet")
    )


def test_cv_records_all_folds(training_config, tmp_path):
    splits = tmp_path / "splits"
    make_splits(
        {
            "participants_path": tmp_path / "participants.parquet",
            "output_dir": splits,
            "n_folds": 2,
        }
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(training_config))
    result = run_cross_validation(
        {
            "study": {
                "study_id": "test",
                "output_dir": str(tmp_path / "cv"),
                "fold_root": str(splits),
                "n_folds": 2,
            },
            "models": [{"model_id": "gru", "source_config_path": str(model_path)}],
        }
    )
    assert result == {"status": "complete", "models": 1, "runs": 2}
    manifest = json.loads((tmp_path / "cv/cv_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == 2


def test_covariate_imputation_does_not_learn_test_values(training_config, tmp_path):
    config = training_config
    config["model"]["type"] = "gru_covariates_binary"
    config["data"]["covariates"] = {
        "enabled": True,
        "parquet_path": str(tmp_path / "participants.parquet"),
        "numeric_columns": ["RIDAGEYR"],
        "categorical_columns": ["RIAGENDR"],
    }
    frame = pd.read_parquet(config["data"]["parquet_path"])
    partitions = load_partitions(frame, config)
    _, before, _ = prepare_training_data(copy.deepcopy(config))
    participants = pd.read_parquet(tmp_path / "participants.parquet")
    participants.loc[participants.SEQN.isin(partitions["test"]), "RIDAGEYR"] = 9999
    participants.loc[participants.SEQN.isin(partitions["test"]), "RIAGENDR"] = 99
    participants.to_parquet(tmp_path / "participants.parquet")
    arrays, after, _ = prepare_training_data(copy.deepcopy(config))
    assert before["covariates"] == after["covariates"]
    assert np.all(arrays["test"]["static_cat"] == 0)


def test_cv_rejects_reused_test_fold(training_config, tmp_path):
    splits = tmp_path / "splits"
    make_splits(
        {
            "participants_path": tmp_path / "participants.parquet",
            "output_dir": splits,
            "n_folds": 2,
        }
    )
    for part in ("train", "val", "test"):
        content = (splits / f"fold_0/{part}_ids.csv").read_text()
        (splits / f"fold_1/{part}_ids.csv").write_text(content)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(training_config))
    with pytest.raises(ValueError, match="overlap across folds"):
        run_cross_validation(
            {
                "study": {
                    "study_id": "test",
                    "output_dir": str(tmp_path / "cv"),
                    "fold_root": str(splits),
                    "n_folds": 2,
                },
                "models": [{"model_id": "gru", "source_config_path": str(model_path)}],
            }
        )
