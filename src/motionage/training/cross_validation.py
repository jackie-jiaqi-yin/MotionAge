"""Sequential CV runner with explicit model configs and reusable outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from motionage.config import apply_overrides, load_yaml_config, resolve_config
from motionage.training.workflow import run_training
from motionage.workflow_io import read_table, validate_ids, write_json


def run_cross_validation(config: dict) -> dict:
    if "study" not in config:
        return run_training(config)
    study = config["study"]
    root = Path(study.get("output_dir", f"outputs/{study['study_id']}"))
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "CV output directory is not empty; choose a new study.output_dir."
        )
    fold_root = Path(study["fold_root"])
    folds = [fold_root / f"fold_{index}" for index in range(int(study["n_folds"]))]
    if not folds:
        raise ValueError("CV requires at least one fold.")
    for fold in folds:
        for split in ("train", "val", "test"):
            if not (fold / f"{split}_ids.csv").is_file():
                raise FileNotFoundError(f"Missing {split} IDs in {fold}")
    models = config.get("models", [])
    ids = [entry["model_id"] for entry in models]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("CV requires unique model IDs.")
    for model_id in ids:
        if not model_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Model IDs must be simple directory names.")
    selected = study.get("model_ids", ids)
    if not selected or set(selected) - set(ids):
        raise ValueError("study.model_ids must select known model IDs.")
    reference = load_yaml_config(models[0]["source_config_path"])
    id_column = reference["data"].get("id_column", "SEQN")
    seen_test_ids = set()
    cohort = None
    for fold in folds:
        partitions = {
            name: set(
                validate_ids(read_table(fold / f"{name}_ids.csv"), id_column)[id_column]
            )
            for name in ("train", "val", "test")
        }
        current = set.union(*partitions.values())
        if cohort is not None and current != cohort:
            raise ValueError("CV folds must cover the same participant cohort.")
        cohort = current
        if seen_test_ids & partitions["test"]:
            raise ValueError("Outer CV test partitions overlap across folds.")
        seen_test_ids.update(partitions["test"])
    if seen_test_ids != cohort:
        raise ValueError("Outer CV tests must cover each participant exactly once.")
    runs, rows = [], []
    for entry in models:
        if entry["model_id"] not in selected:
            continue
        for index, fold in enumerate(folds):
            model_config = apply_overrides(
                resolve_config(load_yaml_config(entry["source_config_path"])),
                study.get("overrides", []),
            )
            model_config["seed"] = int(study.get("training_seed", 42)) + index
            model_config["split"] = {
                "mode": "precomputed",
                "precomputed_dir": str(fold),
            }
            run_dir = root / entry["model_id"] / fold.name
            model_config["experiment"]["output_dir"] = str(run_dir)
            summary = run_training(model_config)
            runs.append(
                {
                    "model_id": entry["model_id"],
                    "fold": fold.name,
                    "predictions": str(
                        (run_dir / "participant_predictions.parquet").relative_to(root)
                    ),
                }
            )
            for split, metrics in summary["metrics"].items():
                rows.append(
                    {
                        "model_id": entry["model_id"],
                        "fold": fold.name,
                        "split": split,
                        **metrics,
                    }
                )
            write_json(root / "cv_manifest.json", {"status": "running", "runs": runs})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "metrics.csv", index=False)
    test_metrics = metrics.loc[metrics.split == "test"]
    test_metrics.groupby("model_id")[["auroc", "auprc", "logloss", "brier"]].agg(
        ["mean", "std"]
    ).to_csv(root / "cv_summary.csv")
    manifest = {"status": "complete", "runs": runs}
    write_json(root / "cv_manifest.json", manifest)
    return {
        "status": "complete",
        "models": len({run["model_id"] for run in runs}),
        "runs": len(runs),
    }
