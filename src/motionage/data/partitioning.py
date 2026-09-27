"""Reproducible, stratified participant splits and outer cross-validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from motionage.workflow_io import read_table, require_columns, validate_ids, write_json


def validate_partitions(partitions: dict[str, np.ndarray], all_ids: np.ndarray) -> None:
    seen = set()
    for name, ids in partitions.items():
        values = set(ids.tolist())
        if not values or len(values) != len(ids):
            raise ValueError(f"Partition {name} is empty or has duplicate IDs.")
        if seen & values:
            raise ValueError("Participant partitions overlap.")
        seen.update(values)
    if seen != set(all_ids.tolist()):
        raise ValueError("Partitions must cover exactly the input cohort.")


def build_partitions(participants: pd.DataFrame, config: dict) -> dict[str, dict[str, np.ndarray]]:
    id_column = config.get("id_column", "SEQN")
    target = config.get("target_column", "mortstat")
    frame = validate_ids(participants, id_column).sort_values(id_column).reset_index(drop=True)
    require_columns(frame, [target])
    if frame[target].isna().any() or not frame[target].isin([0, 1]).all() or frame[target].nunique() != 2:
        raise ValueError("Stratified splitting requires a non-missing binary target with both classes.")
    seed = int(config.get("random_seed", 42))
    val_size = float(config.get("val_size", 0.25))
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between zero and one (fraction of the outer training set).")
    n_folds = int(config.get("n_folds", 1))
    if n_folds < 1:
        raise ValueError("n_folds must be positive.")
    if n_folds > 1:
        if frame[target].value_counts().min() < n_folds:
            raise ValueError("Each target class needs at least n_folds participants.")
        outer = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(frame, frame[target]))
    else:
        outer = [train_test_split(np.arange(len(frame)), test_size=float(config.get("test_size", 0.2)), stratify=frame[target], random_state=seed)]
    folds = {}
    for index, (development, test) in enumerate(outer):
        train, val = train_test_split(development, test_size=val_size, stratify=frame.iloc[development][target], random_state=seed + index)
        partitions = {name: np.sort(frame.iloc[indices][id_column].to_numpy()) for name, indices in [("train", train), ("val", val), ("test", test)]}
        validate_partitions(partitions, frame[id_column].to_numpy())
        for name, ids in partitions.items():
            if frame.loc[frame[id_column].isin(ids), target].nunique() != 2:
                raise ValueError(f"{name} needs both target classes; use a larger cohort.")
        folds[f"fold_{index}"] = partitions
    return folds


def make_splits(config: dict) -> dict:
    participants = read_table(config["participants_path"])
    folds = build_partitions(participants, config)
    root = Path(config["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError("Split output directory is not empty; choose a new output_dir.")
    root.mkdir(parents=True, exist_ok=True)
    id_column = config.get("id_column", "SEQN")
    target = config.get("target_column", "mortstat")
    summary = {"random_seed": config.get("random_seed", 42), "val_size": config.get("val_size", 0.25), "n_folds": len(folds), "folds": {}}
    for fold, partitions in folds.items():
        destination = root / fold if len(folds) > 1 else root
        destination.mkdir(exist_ok=True)
        summary["folds"][fold] = {}
        for name, ids in partitions.items():
            pd.DataFrame({id_column: ids}).to_csv(destination / f"{name}_ids.csv", index=False)
            subset = participants.loc[participants[id_column].isin(ids)]
            summary["folds"][fold][name] = {"participants": len(ids), "events": int(subset[target].sum())}
    write_json(root / "manifest.json", summary)
    return summary
