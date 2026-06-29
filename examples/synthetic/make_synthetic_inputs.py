"""Create tiny synthetic MotionAge inputs for smoke tests and examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


_EXPECTED_PUBLIC_BOUNDARY = {
    "synthetic": True,
    "contains_real_participants": False,
    "contains_trained_weights": False,
    "safe_for_public_smoke_tests": True,
}


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
        "public_boundary": _EXPECTED_PUBLIC_BOUNDARY,
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


def validate_synthetic_inputs(output_dir: Path | str) -> dict[str, Any]:
    """Validate generated synthetic inputs and return the public manifest summary."""
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.yaml"
    manifest = _load_yaml_mapping(manifest_path)

    if manifest.get("public_boundary") != _EXPECTED_PUBLIC_BOUNDARY:
        raise ValueError("Synthetic manifest public boundary is not safe for public release.")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Synthetic manifest must include a files mapping.")

    required_files = {
        "activity_mortstat": files.get("activity_mortstat"),
        "covariates": files.get("covariates"),
        "metadata": files.get("metadata"),
    }
    resolved_files = {
        name: _resolve_manifest_file(output_path, value, manifest_path=manifest_path)
        for name, value in required_files.items()
    }

    activity = pd.read_parquet(resolved_files["activity_mortstat"])
    covariates = pd.read_parquet(resolved_files["covariates"])
    metadata = _load_yaml_mapping(resolved_files["metadata"])

    row_counts = {
        "activity_mortstat": int(len(activity)),
        "covariates": int(len(covariates)),
    }
    if manifest.get("row_counts") != row_counts:
        raise ValueError("Synthetic manifest row counts do not match generated files.")

    columns = {
        "activity_mortstat": activity.columns.tolist(),
        "covariates": covariates.columns.tolist(),
    }
    if manifest.get("columns") != columns:
        raise ValueError("Synthetic manifest columns do not match generated files.")

    _validate_metadata(metadata, activity=activity, covariates=covariates)

    return {
        "participants": manifest.get("participants"),
        "days": manifest.get("days"),
        "seed": manifest.get("seed"),
        "public_boundary": dict(manifest["public_boundary"]),
        "files": {name: str(Path(value)) for name, value in required_files.items()},
        "row_counts": row_counts,
        "columns": columns,
    }


def build_synthetic_smoke_summary(output_dir: Path | str) -> dict[str, Any]:
    """Return an aggregate-only overview of generated synthetic smoke inputs."""
    manifest_summary = validate_synthetic_inputs(output_dir)
    output_path = Path(output_dir)
    activity = pd.read_parquet(output_path / manifest_summary["files"]["activity_mortstat"])
    covariates = pd.read_parquet(output_path / manifest_summary["files"]["covariates"])

    participant_count = int(manifest_summary["participants"])
    participant_targets = activity[["SEQN", "mortstat"]].drop_duplicates()
    event_count = int(covariates.merge(participant_targets, on="SEQN")["mortstat"].sum())
    return {
        "participants": participant_count,
        "days": int(manifest_summary["days"]),
        "activity_rows": int(manifest_summary["row_counts"]["activity_mortstat"]),
        "covariate_rows": int(manifest_summary["row_counts"]["covariates"]),
        "event_count": event_count,
        "event_rate": _public_float(event_count / participant_count),
        "mean_intensity": _public_float(activity["intensity_mean"].mean()),
        "mean_attention_coverage": _public_float(activity["attention_flag"].mean()),
        "age_min": int(covariates["RIDAGEYR"].min()),
        "age_max": int(covariates["RIDAGEYR"].max()),
        "sex_values": [int(value) for value in sorted(covariates["RIAGENDR"].unique().tolist())],
        "public_boundary": dict(manifest_summary["public_boundary"]),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _resolve_manifest_file(output_path: Path, value: Any, *, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest_path} has an invalid file reference: {value!r}")
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{manifest_path} file references must stay inside the output directory.")
    resolved = output_path / relative_path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    activity: pd.DataFrame,
    covariates: pd.DataFrame,
) -> None:
    id_column = metadata.get("id_column")
    target_column = metadata.get("target_column")
    followup_month_column = metadata.get("followup_month_column")
    numeric_columns = metadata.get("numeric_columns", [])
    categorical_columns = metadata.get("categorical_columns", [])

    if id_column not in activity.columns or id_column not in covariates.columns:
        raise ValueError("Synthetic metadata id_column must be present in both generated tables.")
    for column in (target_column, followup_month_column):
        if column not in activity.columns:
            raise ValueError(f"Synthetic metadata column is missing from activity table: {column}")
    for column in [*numeric_columns, *categorical_columns]:
        if column not in covariates.columns:
            raise ValueError(f"Synthetic metadata covariate is missing: {column}")


def _public_float(value: float) -> float:
    return round(float(value), 12)


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
