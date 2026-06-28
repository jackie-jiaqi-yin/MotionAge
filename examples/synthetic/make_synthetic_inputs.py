"""Create tiny synthetic MotionAge inputs for smoke tests and examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def generate_synthetic_inputs(
    output_dir: Path | str,
    *,
    participants: int = 32,
    days: int = 7,
    seed: int = 42,
) -> dict[str, Path]:
    """Write deterministic synthetic activity, mortality, and covariate inputs."""
    if participants < 4:
        raise ValueError("participants must be at least 4.")
    if days < 1:
        raise ValueError("days must be at least 1.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    participant_ids = np.arange(1, participants + 1, dtype=np.int64)
    ages = rng.integers(40, 86, size=participants)
    sex = rng.integers(1, 3, size=participants)
    bmi = np.round(rng.normal(28.0, 4.5, size=participants), 2)
    smoking = rng.binomial(1, 0.35, size=participants)

    risk_score = -5.0 + 0.055 * ages + 0.045 * (bmi - 27.0) + 0.5 * smoking
    mortality_probability = 1.0 / (1.0 + np.exp(-risk_score))
    mortstat = rng.binomial(1, mortality_probability).astype(np.int64)
    permth_int = np.where(
        mortstat == 1,
        rng.integers(6, 61, size=participants),
        rng.integers(61, 121, size=participants),
    )

    activity_rows: list[dict[str, float | int]] = []
    for index, seqn in enumerate(participant_ids):
        baseline = 42.0 - 0.35 * (ages[index] - 40.0) - 2.5 * mortstat[index] + rng.normal(0.0, 2.0)
        for day in range(days):
            for hour in range(24):
                circadian = 18.0 * max(0.0, np.sin((hour - 6) / 24.0 * 2.0 * np.pi))
                weekend_shift = 2.0 if day % 7 in {5, 6} else 0.0
                intensity = max(0.0, baseline + circadian + weekend_shift + rng.normal(0.0, 3.0))
                attention_flag = 0 if rng.random() < 0.04 else 1
                activity_rows.append(
                    {
                        "SEQN": int(seqn),
                        "PAXDAY": int(day),
                        "PAXHOUR": int(hour),
                        "intensity_mean": round(float(intensity), 4),
                        "attention_flag": int(attention_flag),
                        "mortstat": int(mortstat[index]),
                        "permth_int": int(permth_int[index]),
                        "RIDAGEYR": int(ages[index]),
                        "RIAGENDR": int(sex[index]),
                    }
                )

    activity = pd.DataFrame(activity_rows)
    covariates = pd.DataFrame(
        {
            "SEQN": participant_ids,
            "RIDAGEYR": ages.astype(np.int64),
            "RIAGENDR": sex.astype(np.int64),
            "BMXBMI": bmi.astype(float),
            "SMQ020": smoking.astype(np.int64),
        }
    )
    metadata = {
        "id_column": "SEQN",
        "numeric_columns": ["RIDAGEYR", "BMXBMI", "SMQ020"],
        "categorical_columns": ["RIAGENDR"],
        "target_column": "mortstat",
        "followup_month_column": "permth_int",
    }
    manifest = {
        "participants": int(participants),
        "days": int(days),
        "seed": int(seed),
        "public_boundary": {
            "synthetic": True,
            "contains_real_participants": False,
            "contains_trained_weights": False,
            "safe_for_public_smoke_tests": True,
        },
        "files": {
            "activity_mortstat": "activity_mortstat_joined.parquet",
            "covariates": "nhanes_mortality_covariates_l1.parquet",
            "metadata": "metadata_l1.yaml",
        },
        "row_counts": {
            "activity_mortstat": int(len(activity)),
            "covariates": int(len(covariates)),
        },
        "columns": {
            "activity_mortstat": activity.columns.tolist(),
            "covariates": covariates.columns.tolist(),
        },
    }

    activity_path = output_path / "activity_mortstat_joined.parquet"
    covariate_path = output_path / "nhanes_mortality_covariates_l1.parquet"
    metadata_path = output_path / "metadata_l1.yaml"
    manifest_path = output_path / "manifest.yaml"

    activity.to_parquet(activity_path, index=False)
    covariates.to_parquet(covariate_path, index=False)
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    return {
        "activity_mortstat": activity_path,
        "covariates": covariate_path,
        "metadata": metadata_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/synthetic/generated"))
    parser.add_argument("--participants", type=int, default=32)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outputs = generate_synthetic_inputs(
        args.output_dir,
        participants=args.participants,
        days=args.days,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
