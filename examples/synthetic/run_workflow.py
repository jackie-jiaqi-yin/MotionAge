"""Generate synthetic raw inputs/configs and execute the public methodology commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from motionage.config import load_yaml_config, save_resolved_config
from motionage.workflow_io import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare_example(
    root: Path, *, family: str, cv: bool, seed: int
) -> list[tuple[str, Path]]:
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "Example output directory is not empty; choose a fresh directory."
        )
    raw, configs = root / "raw", root / "configs"
    raw.mkdir(parents=True, exist_ok=True)
    configs.mkdir()
    rng = np.random.default_rng(seed)
    count, minutes = 64, 120
    ids = np.arange(1, count + 1)
    ages = 40 + (np.arange(count) * 7 % 40)
    sex = (np.arange(count) // 2) % 2 + 1
    target = (ages >= 60).astype(int)
    participants = pd.DataFrame(
        {"SEQN": ids, "RIDAGEYR": ages, "RIAGENDR": sex, "BMXBMI": 20 + ages * 0.1}
    )
    participants.loc[participants.index % 9 == 0, "BMXBMI"] = np.nan
    participants.to_csv(raw / "participants.csv", index=False)
    pd.DataFrame(
        {
            "SEQN": ids,
            "mortstat": target,
            "permth_int": np.where(target, 36, 120),
            "eligstat": 1,
        }
    ).to_csv(raw / "mortality.csv", index=False)
    minute = np.tile(np.arange(minutes), count)
    activity = pd.DataFrame(
        {
            "SEQN": np.repeat(ids, minutes),
            "PAXN": minute + 1,
            "PAXDAY": 1,
            "PAXHOUR": minute // 60,
            "PAXMINUT": minute % 60,
            "PAXINTEN": 400
            - np.repeat(ages - 40, minutes) * 7
            + rng.normal(0, 4, count * minutes),
            "PAXSTAT": 1,
            "PAXCAL": 1,
        }
    )
    activity.to_csv(raw / "activity.csv", index=False)
    preparation = {
        "inputs": {
            "activity": [str(raw / "activity.csv")],
            "participants": [{"files": [str(raw / "participants.csv")]}],
            "mortality": [str(raw / "mortality.csv")],
        },
        "output_dir": str(root / "processed"),
        "chunksize": 1000,
        "min_age": 18,
        "followup_months": 60,
        "require_complete_followup": True,
        "preprocessing": {
            "epoch_minutes": 5,
            "attention_tau": 0.2,
            "require_calibration": True,
            "nonwear": {"min_period_len": 90},
        },
        "covariate_bundles": {
            "l1": {
                "numeric_columns": ["RIDAGEYR", "BMXBMI"],
                "categorical_columns": ["RIAGENDR"],
            }
        },
    }
    split = {
        "participants_path": str(root / "processed/participants.parquet"),
        "output_dir": str(root / "splits"),
        "n_folds": 2 if cv else 1,
        "random_seed": seed,
        "val_size": 0.25,
        "test_size": 0.25,
    }
    save_resolved_config(preparation, configs / "prepare.yaml")
    save_resolved_config(split, configs / "splits.yaml")
    entries = []
    for model_family in ["gru", "lstm", "transformer"] if cv else [family]:
        training = load_yaml_config(
            REPO_ROOT / f"configs/paper/{model_family}_fitbit_only_60m.yaml"
        )
        training["experiment"]["output_dir"] = str(root / "training")
        training["data"]["parquet_path"] = str(
            root / "processed/activity_mortstat/activity_mortstat_joined.parquet"
        )
        training["data"]["participants_path"] = str(
            root / "processed/participants.parquet"
        )
        training["split"] = {
            "mode": "precomputed",
            "precomputed_dir": str(root / "splits"),
        }
        training["windowing"] = {
            "seq_len": 12,
            "stride_ratio": 1.0,
            "min_attention_ratio": 0.3,
        }
        training["seed"] = seed
        training["training"].update(
            max_epochs=30,
            fixed_epochs=30,
            batch_size=16,
            learning_rate=0.02,
            weight_decay=0,
            deterministic=True,
        )
        training["model"].update(num_layers=1, dropout=0, hour_emb_dim=2, day_emb_dim=2)
        if model_family == "transformer":
            training["model"].update(
                d_model=8,
                nhead=2,
                dim_feedforward=16,
                intensity_proj_dim=4,
                max_seq_len=24,
            )
        else:
            training["model"]["hidden_size"] = 8
        path = configs / f"{model_family}.yaml"
        save_resolved_config(training, path)
        entries.append(
            {
                "model_id": model_family,
                "family": model_family,
                "source_config_path": str(path),
            }
        )
    commands = [
        ("scripts/data/prepare_nhanes_inputs.py", configs / "prepare.yaml"),
        ("scripts/data/make_splits.py", configs / "splits.yaml"),
    ]
    if cv:
        save_resolved_config(
            {
                "study": {
                    "study_id": "synthetic_cv",
                    "output_dir": str(root / "training"),
                    "fold_root": str(root / "splits"),
                    "n_folds": 2,
                    "training_seed": seed,
                },
                "models": entries,
            },
            configs / "cv.yaml",
        )
        commands.append(("scripts/train/run_mortality_cv.py", configs / "cv.yaml"))
    else:
        commands.append(("scripts/train/run_training.py", configs / f"{family}.yaml"))
    analysis = {
        "analysis": {
            "participant_predictions_dir": str(root / "training"),
            "participants_path": str(root / "processed/participants.parquet"),
            "output_dir": str(root / "analysis"),
        },
        "mapping": {
            "fit_partitions": ["train"],
            "representative_probability": "median",
            "min_stratum_participants": 2,
            "require_positive_beta": True,
            "weighted_fit": True,
        },
        "evaluation": {
            "bootstrap": {"n_resamples": 50, "ci_level": 0.95, "random_seed": seed},
            "feature_sets": [
                {"name": "chronological_age", "columns": ["RIDAGEYR"]},
                {"name": "motionage", "columns": ["MotionAge"]},
                {"name": "motionage_accel", "columns": ["RIDAGEYR", "MotionAgeAccel"]},
            ],
            "baseline_feature_set": "chronological_age",
        },
    }
    save_resolved_config(analysis, configs / "analysis.yaml")
    commands.append(("scripts/analyze/run_motionage.py", configs / "analysis.yaml"))
    write_json(
        root / "example_manifest.json",
        {
            "synthetic": True,
            "contains_real_participants": False,
            "seed": seed,
            "participants": count,
            "cv": cv,
            "commands": [
                {"script": script, "config": str(path)} for script, path in commands
            ],
        },
    )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/synthetic_workflow")
    )
    parser.add_argument(
        "--family", choices=["gru", "lstm", "transformer"], default="gru"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run two folds for each of the three model families.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write synthetic raw inputs and configs without running workflows.",
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    try:
        commands = prepare_example(root, family=args.family, cv=args.cv, seed=args.seed)
    except ValueError as error:
        parser.exit(2, f"{error}\n")
    if not args.prepare_only:
        for script, config in commands:
            print(f"Running {script}", flush=True)
            subprocess.run(
                [sys.executable, str(REPO_ROOT / script), "--config", str(config)],
                check=True,
            )
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "status": "prepared" if args.prepare_only else "complete",
            }
        )
    )


if __name__ == "__main__":
    main()
