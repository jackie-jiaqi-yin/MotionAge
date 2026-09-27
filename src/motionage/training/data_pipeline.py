"""Train-fitted preprocessing and dataloaders for the public binary workflow."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from motionage.config import load_yaml_config
from motionage.data.dataset import (
    FitbitSequenceDataset,
    FitbitSequenceWithCovariatesDataset,
)
from motionage.data.partitioning import build_partitions, validate_partitions
from motionage.data.windowing import create_windows
from motionage.preprocessing.covariates import (
    StaticCovariatePreprocessor,
    fit_static_covariate_preprocessor,
    transform_static_covariates,
)
from motionage.workflow_io import read_table, require_columns, validate_ids


def load_activity(config: dict) -> pd.DataFrame:
    data = config["data"]
    frame = validate_ids(
        read_table(data["parquet_path"]), data.get("id_column", "SEQN"), unique=False
    )
    target = data.get("target_column", "mortstat")
    require_columns(frame, [target])
    if not frame[target].isin([0, 1]).all():
        raise ValueError("Prepared targets must be non-missing binary horizon labels.")
    if frame.groupby(data.get("id_column", "SEQN"))[target].nunique().max() != 1:
        raise ValueError("Conflicting participant targets in activity input.")
    return frame


def load_partitions(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    data = config["data"]
    id_column = data.get("id_column", "SEQN")
    target = data.get("target_column", "mortstat")
    participants = frame[[id_column, target]].drop_duplicates()
    split = config["split"]
    if split.get("mode", "runtime") == "runtime":
        return build_partitions(
            participants,
            {**split, "id_column": id_column, "target_column": target, "n_folds": 1},
        )["fold_0"]
    if split["mode"] != "precomputed":
        raise ValueError("split.mode must be runtime or precomputed.")
    from pathlib import Path

    partitions = {
        name: validate_ids(
            read_table(Path(split["precomputed_dir"]) / f"{name}_ids.csv"), id_column
        )[id_column].to_numpy()
        for name in ("train", "val", "test")
    }
    validate_partitions(partitions, participants[id_column].to_numpy())
    return partitions


def window_arrays(frame: pd.DataFrame, config: dict) -> dict:
    data, window = config["data"], config["windowing"]
    fields = data.get(
        "feature_columns", ["intensity_mean", "PAXHOUR", "PAXDAY", "attention_flag"]
    )
    expected = ["intensity_mean", "PAXHOUR", "PAXDAY", "attention_flag"]
    if set(fields) != set(expected) or len(fields) != 4:
        raise ValueError(
            f"Sequence features must be exactly {expected}; order may vary."
        )
    values = frame[expected].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Sequence features must be finite; mask missing minutes during preparation."
        )
    for column, lower, upper in [
        ("PAXHOUR", 0, 23),
        ("PAXDAY", 1, 7),
        ("attention_flag", 0, 1),
    ]:
        if (
            not frame[column].between(lower, upper).all()
            or (frame[column] % 1 != 0).any()
        ):
            raise ValueError(f"Invalid sequence field {column}.")
    seq_len = int(window["seq_len"])
    result = create_windows(
        frame,
        feature_columns=expected,
        target_column=data.get("target_column", "mortstat"),
        id_column=data.get("id_column", "SEQN"),
        seq_len=seq_len,
        stride=int(
            window.get(
                "stride", max(1, int(seq_len * float(window.get("stride_ratio", 1.0))))
            )
        ),
        time_columns=window.get("time_columns"),
        min_attention_ratio=float(window.get("min_attention_ratio", 0.0)),
    )
    if not len(result["y"]):
        raise ValueError(
            "No windows remain; check sequence length and attention coverage."
        )
    x = result["X"]
    return {
        "intensity": x[:, :, 0],
        "hour_idx": x[:, :, 1].astype(int),
        "day_idx": (x[:, :, 2] - 1).astype(int),
        "mask": x[:, :, 3],
        "targets": result["y"],
        "meta": result["meta"],
    }


def apply_preprocessing(
    arrays: dict, state: dict, covariates: pd.DataFrame | None
) -> None:
    norm = state["intensity"]
    arrays["intensity"] = ((arrays["intensity"] - norm["mean"]) / norm["std"]).astype(
        np.float32
    )
    if state.get("covariates") is not None:
        preprocessor = StaticCovariatePreprocessor(**state["covariates"])
        if covariates is None:
            raise ValueError("This model requires a covariate table.")
        transformed = transform_static_covariates(covariates, preprocessor)
        indices = pd.Index(transformed[preprocessor.id_column]).get_indexer(
            arrays["meta"]["id"]
        )
        if (indices < 0).any():
            raise ValueError(
                "Covariate table is missing participants with activity windows."
            )
        for key in ("static_num", "static_cat", "static_num_missing"):
            arrays[key] = transformed[key][indices]


def prepare_training_data(
    config: dict, *, combine_train_val: bool = False
) -> tuple[dict, dict, pd.DataFrame]:
    frame = load_activity(config)
    id_column = config["data"].get("id_column", "SEQN")
    partitions = load_partitions(frame, config)
    if combine_train_val:
        partitions["train"] = np.sort(
            np.concatenate([partitions["train"], partitions.pop("val")])
        )
    arrays = {
        name: window_arrays(frame.loc[frame[id_column].isin(ids)], config)
        for name, ids in partitions.items()
    }
    for name, values in arrays.items():
        if len(np.unique(values["targets"])) != 2:
            raise ValueError(
                f"Partition {name} must retain both classes after window filtering."
            )
    train = arrays["train"]
    valid = train["mask"] > 0
    if not valid.any():
        raise ValueError("Training windows contain no observed activity.")
    values = train["intensity"][valid]
    normalize = config["data"].get("normalize_intensity", True)
    state = {
        "intensity": {
            "mean": float(values.mean()) if normalize else 0.0,
            "std": max(float(values.std()), 1e-8) if normalize else 1.0,
        },
        "covariates": None,
    }
    if normalize and float(values.std()) < 1e-8:
        state["intensity"]["std"] = 1.0
    cov_cfg = config["data"].get("covariates", {})
    covariates = None
    if cov_cfg.get("enabled", False):
        covariates = validate_ids(read_table(cov_cfg["parquet_path"]), id_column)
        metadata = (
            load_yaml_config(cov_cfg["metadata_path"])
            if cov_cfg.get("metadata_path")
            else {}
        )
        numeric = cov_cfg.get(
            "numeric_columns",
            metadata.get(
                "numeric_columns", metadata.get("selected_numeric_columns", [])
            ),
        )
        categorical = cov_cfg.get(
            "categorical_columns",
            metadata.get(
                "categorical_columns", metadata.get("selected_categorical_columns", [])
            ),
        )
        if not numeric and not categorical:
            raise ValueError("Covariate fields must be declared in config or metadata.")
        if config["data"].get("target_column", "mortstat") in [*numeric, *categorical]:
            raise ValueError("Target cannot be used as a covariate.")
        training_ids = train["meta"]["id"].unique()
        fitted = fit_static_covariate_preprocessor(
            covariates.loc[covariates[id_column].isin(training_ids)],
            id_column=id_column,
            numeric_columns=numeric,
            categorical_columns=categorical,
        )
        state["covariates"] = asdict(fitted)
        model = config["model"].get("params", config["model"])
        model["num_numeric_features"] = len(numeric)
        model["categorical_cardinalities"] = [
            len(fitted.categorical_maps[column]) + 1 for column in categorical
        ]
    if ("covariates" in config["model"]["type"]) != bool(cov_cfg.get("enabled", False)):
        raise ValueError("Covariate model type and data.covariates.enabled must agree.")
    for value in arrays.values():
        apply_preprocessing(value, state, covariates)
    state["partition_counts"] = {
        name: {
            "assigned_participants": len(partitions[name]),
            "retained_participants": int(value["meta"]["id"].nunique()),
            "windows": len(value["targets"]),
        }
        for name, value in arrays.items()
    }
    return arrays, state, frame


def make_loader(
    arrays: dict, *, batch_size: int, shuffle: bool = False, seed: int = 42
) -> DataLoader:
    fields = {key: value for key, value in arrays.items() if key != "meta"}
    cls = (
        FitbitSequenceWithCovariatesDataset
        if "static_num" in fields
        else FitbitSequenceDataset
    )
    return DataLoader(
        cls(**fields),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
