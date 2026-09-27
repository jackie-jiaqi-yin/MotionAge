"""Apply a saved model and preprocessing state to another local activity table."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from motionage.config import load_yaml_config
from motionage.models import build_model
from motionage.training.data_pipeline import apply_preprocessing, window_arrays
from motionage.training.device import resolve_device
from motionage.training.workflow import attach_metadata, participant_predictions
from motionage.workflow_io import read_table, validate_ids


def run_prediction(request: dict) -> dict:
    root = Path(request["checkpoint_dir"])
    config = load_yaml_config(root / "resolved_config.yaml")
    config["data"].update(request["data"])
    config["device"] = request.get("device", "cpu")
    state = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    id_column = config["data"].get("id_column", "SEQN")
    target = config["data"].get("target_column", "mortstat")
    activity = validate_ids(
        read_table(config["data"]["parquet_path"]), id_column, unique=False
    )
    labeled = target in activity
    if not labeled:
        activity[target] = 0
    arrays = window_arrays(activity, config)
    covariates = None
    if state["covariates"] is not None:
        covariates = validate_ids(
            read_table(config["data"]["covariates"]["parquet_path"]), id_column
        )
    apply_preprocessing(arrays, state, covariates)
    device = resolve_device(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(root / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = participant_predictions(model, arrays, config, device)
    if not labeled:
        predictions = predictions.drop(columns=target)
    predictions = attach_metadata(predictions, config)
    output = Path(request["output_path"])
    if output.exists():
        raise ValueError("Prediction output already exists; choose a new output_path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, index=False)
    return {"participants": len(predictions), "output_path": str(output)}
