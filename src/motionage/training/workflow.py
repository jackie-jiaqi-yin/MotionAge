"""Executable binary training, fixed-epoch refitting and participant prediction."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from motionage.config import save_resolved_config
from motionage.evaluation.metrics import (
    binary_probability_metrics,
    binary_threshold_metrics,
    select_binary_threshold,
)
from motionage.models import build_model
from motionage.models.initialization import initialize_model_from_config
from motionage.training.data_pipeline import make_loader, prepare_training_data
from motionage.training.device import resolve_device
from motionage.training.evaluate import logits_to_probabilities, predict
from motionage.training.loss import build_loss_function
from motionage.training.model_inputs import build_model_inputs
from motionage.training.scheduler import build_optimizer_and_scheduler
from motionage.training.task import (
    is_binary_classification,
    selection_metric_direction,
    selection_metric_name,
    threshold_metric_name,
)
from motionage.workflow_io import read_table, validate_ids, write_json


def participant_predictions(model, arrays: dict, config: dict, device) -> pd.DataFrame:
    logits, target = predict(
        model,
        make_loader(
            arrays, batch_size=int(config["training"].get("eval_batch_size", 128))
        ),
        device,
    )
    rows = pd.DataFrame(
        {
            config["data"].get("id_column", "SEQN"): arrays["meta"]["id"].to_numpy(),
            config["data"].get("target_column", "mortstat"): target,
            "probability": logits_to_probabilities(logits),
        }
    )
    id_column = config["data"].get("id_column", "SEQN")
    target_column = config["data"].get("target_column", "mortstat")
    return rows.groupby(id_column, as_index=False).agg(
        **{
            target_column: (target_column, "first"),
            "probability": ("probability", "mean"),
            "n_windows": ("probability", "size"),
        }
    )


def attach_metadata(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    path = config["data"].get("participants_path")
    if not path:
        return predictions
    id_column = config["data"].get("id_column", "SEQN")
    target = config["data"].get("target_column", "mortstat")
    metadata = validate_ids(read_table(path), id_column)
    if not set(predictions[id_column]).issubset(set(metadata[id_column])):
        raise ValueError("Participant metadata is missing predicted participants.")
    if target in metadata and target in predictions:
        aligned = predictions.merge(
            metadata[[id_column, target]], on=id_column, suffixes=("", "_metadata")
        )
        if not aligned[target].eq(aligned[f"{target}_metadata"]).all():
            raise ValueError(
                "Participant metadata and prepared activity targets disagree."
            )
    columns = [id_column, *[column for column in metadata if column not in predictions]]
    return predictions.merge(metadata[columns], on=id_column, validate="many_to_one")


def run_training(raw_config: dict) -> dict:
    config = copy.deepcopy(raw_config)
    if not is_binary_classification(config):
        raise ValueError(
            "The public training workflow requires a binary_classification task."
        )
    train_cfg = config["training"]
    train_cfg.setdefault("weight_decay", 0.0)
    train_cfg.setdefault("scheduler_patience", 5)
    train_cfg.setdefault("scheduler_factor", 0.5)
    seed = int(config.get("seed", config.get("split", {}).get("random_seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(train_cfg.get("cpu_threads", 1)))
    torch.use_deterministic_algorithms(bool(train_cfg.get("deterministic", False)))
    refit = config.get("final_refit", {})
    fixed_epochs = (
        refit.get("epochs")
        if refit.get("enabled", False)
        else train_cfg.get("fixed_epochs")
    )
    if refit.get("enabled", False) and fixed_epochs is None:
        raise ValueError("final_refit.epochs must be explicitly configured.")
    if config["data"].get("target_source", "parquet") != "parquet":
        raise ValueError(
            "The workflow expects prepared horizon labels in data.parquet_path."
        )
    if train_cfg.get("freeze_schedule"):
        raise ValueError(
            "Staged freezing is not supported by this workflow; use the low-level training API."
        )
    epochs = int(fixed_epochs if fixed_epochs is not None else train_cfg["max_epochs"])
    if epochs < 1:
        raise ValueError("Training epoch budget must be positive.")
    output = Path(config["experiment"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Training output directory is not empty; choose a new output_dir."
        )
    arrays, state, _ = prepare_training_data(
        config,
        combine_train_val=bool(
            refit.get("enabled", False) and refit.get("combine_train_val", True)
        ),
    )
    device = resolve_device(config)
    model = build_model(config).to(device)
    initialization = initialize_model_from_config(model, config, device)
    optimizer, scheduler, _ = build_optimizer_and_scheduler(model, config)
    criterion = build_loss_function(config, device)
    metric = selection_metric_name(config)
    maximize = selection_metric_direction(config) == "maximize"
    best = -np.inf if maximize else np.inf
    best_state = None
    best_epoch = 0
    patience = 0
    logs = []
    loader = make_loader(
        arrays["train"],
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    limit = train_cfg.get("limit_train_batches")
    if limit is not None and int(limit) < 1:
        raise ValueError("limit_train_batches must be positive when provided.")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loader):
            if limit is not None and batch_index >= int(limit):
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(**build_model_inputs(batch, device)).reshape(-1)
            loss = criterion(logits, batch["y"].to(device).reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("grad_clip_norm", 1.0))
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch["y"])
            sample_count += len(batch["y"])
        row = {"epoch": epoch, "train_loss": loss_sum / sample_count}
        if fixed_epochs is None:
            val = participant_predictions(model, arrays["val"], config, device)
            metrics = binary_probability_metrics(
                val[config["data"].get("target_column", "mortstat")].to_numpy(),
                val.probability.to_numpy(),
            )
            score = float(metrics[metric])
            row.update({f"val_{key}": value for key, value in metrics.items()})
            scheduler.step(score)
            improved = score > best if maximize else score < best
            if improved:
                best, best_epoch, patience = score, epoch, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                patience += 1
        else:
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        logs.append(row)
        if fixed_epochs is None and patience >= int(
            train_cfg.get("early_stopping_patience", 10)
        ):
            break
    if best_state is None:
        raise ValueError("Training did not produce a valid selected model.")
    model.load_state_dict(best_state)
    predictions = []
    metrics_by_split = {}
    target = config["data"].get("target_column", "mortstat")
    for split, values in arrays.items():
        frame = participant_predictions(model, values, config, device)
        frame["split"] = split
        predictions.append(frame)
        metrics_by_split[split] = binary_probability_metrics(
            frame[target].to_numpy(), frame.probability.to_numpy()
        )
    combined = attach_metadata(pd.concat(predictions, ignore_index=True), config)
    threshold = 0.5
    if fixed_epochs is None:
        validation = combined.loc[combined.split == "val"]
        threshold, _ = select_binary_threshold(
            validation[target].to_numpy(),
            validation.probability.to_numpy(),
            metric=threshold_metric_name(config),
        )
    for name in metrics_by_split:
        part = combined.loc[combined.split == name]
        metrics_by_split[name].update(
            binary_threshold_metrics(
                part[target].to_numpy(),
                part.probability.to_numpy(),
                threshold=threshold,
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": best_state, "epoch": best_epoch}, output / "model.pt"
    )
    combined.to_parquet(output / "participant_predictions.parquet", index=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    write_json(output / "preprocessing.json", state)
    write_json(output / "training_log.json", logs)
    summary = {
        "seed": seed,
        "model_type": config["model"]["type"],
        "best_epoch": best_epoch,
        "epochs_run": len(logs),
        "fit_mode": "validation_selected" if fixed_epochs is None else "fixed_epochs",
        "threshold": threshold,
        "threshold_source": "validation" if fixed_epochs is None else "fixed_0.5",
        "initialization": {"applied": initialization.applied},
        "metrics": metrics_by_split,
        "partitions": state["partition_counts"],
    }
    write_json(output / "summary.json", summary)
    return summary
