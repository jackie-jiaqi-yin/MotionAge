"""Run or reuse a MotionAge mapping without publication-specific output assembly."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from motionage.analysis.motionage.mapping import (
    LogisticInverseGroupFit,
    MotionAgeMapping,
    apply_motionage_mapping,
    fit_motionage_mapping,
    mapping_to_jsonable,
)
from motionage.evaluation.secondary import evaluate_secondary_feature_sets
from motionage.evaluation.workflow import evaluate_prediction_frame
from motionage.workflow_io import read_table, require_columns, validate_ids, write_json


def _load_mapping(path: str | Path) -> MotionAgeMapping:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["groups"] = tuple(
        LogisticInverseGroupFit(**row) for row in payload["groups"]
    )
    payload["fit_partitions"] = tuple(payload["fit_partitions"])
    mapping = MotionAgeMapping(**payload)
    if not mapping.groups or any(
        not np.isfinite(group.beta)
        or abs(group.beta) < 1e-12
        or not np.isfinite(group.alpha)
        for group in mapping.groups
    ):
        raise ValueError("Saved mapping has invalid or zero slope parameters.")
    if mapping.require_positive_beta and any(
        group.beta <= 0 for group in mapping.groups
    ):
        raise ValueError("Saved mapping requires positive slopes.")
    return mapping


def analyze_participants(
    participants: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, MotionAgeMapping, pd.DataFrame, dict]:
    analysis = config["analysis"]
    settings = config.get("mapping", {})
    id_column = analysis.get("id_column", "SEQN")
    age = analysis.get("age_column", "RIDAGEYR")
    sex = analysis.get("sex_column", "RIAGENDR")
    probability = analysis.get("probability_column", "probability")
    participants = validate_ids(participants, id_column)
    require_columns(participants, [age, sex, probability])
    if (
        not np.isfinite(participants[[age, probability]].to_numpy(dtype=float)).all()
        or participants[sex].isna().any()
    ):
        raise ValueError(
            "Mapping inputs require complete finite ages/probabilities and known strata."
        )
    if not participants[probability].between(0, 1).all():
        raise ValueError("Mapping probabilities must be between zero and one.")
    if settings.get("probability_transform", "logit") != "logit":
        raise ValueError("Only logit probability mapping is supported.")
    if settings.get("age_bin_column", "age_bin") != "age_bin":
        raise ValueError(
            "Age bins are computed by rounding chronological age into age_bin."
        )
    if settings.get("strata", ["sex"]) not in (["sex"], [sex]):
        raise ValueError("MotionAge mapping currently supports sex-specific strata.")
    if settings.get("parameters_path"):
        mapping = _load_mapping(settings["parameters_path"])
        if (mapping.age_column, mapping.sex_column, mapping.probability_column) != (
            age,
            sex,
            probability,
        ):
            raise ValueError(
                "Saved mapping column definitions do not match analysis config."
            )
        diagnostics = pd.DataFrame()
    else:
        require_columns(participants, ["split"])
        fit_partitions = settings.get("fit_partitions", ["train"])
        if not fit_partitions or not set(fit_partitions).issubset(
            {"train", "val", "validation"}
        ):
            raise ValueError(
                "Mapping fit_partitions may only use development partitions, never test."
            )
        fit = participants.loc[participants.split.isin(fit_partitions)]
        counts = fit.groupby(sex).size()
        minimum = int(settings.get("min_stratum_participants", 2))
        if minimum < 2 or counts.empty or (counts < minimum).any():
            raise ValueError("Too few fitting participants in a mapping stratum.")
        mapping, diagnostics = fit_motionage_mapping(
            participants,
            fit_partitions=fit_partitions,
            representative_probability=settings.get(
                "representative_probability", "median"
            ),
            clip_eps=float(settings.get("clip_eps", 0.0001)),
            weighted_fit=bool(settings.get("weighted_fit", True)),
            clamp_output_to_fit_age_range=bool(
                settings.get("clamp_output_to_fit_age_range", False)
            ),
            require_positive_beta=bool(settings.get("require_positive_beta", True)),
            probability_column=probability,
            age_column=age,
            sex_column=sex,
        )
    if any(abs(group.beta) < 1e-12 for group in mapping.groups):
        raise ValueError("Mapping slope is too close to zero to invert.")
    if set(participants[sex]) - {group.sex_value for group in mapping.groups}:
        raise ValueError("Prediction strata are missing from the fitted mapping.")
    scores = apply_motionage_mapping(participants, mapping)
    evaluation = config.get("evaluation", {})
    target = analysis.get("target_column", "mortstat")
    result = {}
    if evaluation.get("enabled", True):
        require_columns(scores, [target, "split"])
        held_out = scores.loc[scores.split == evaluation.get("split", "test")]
        result = evaluate_prediction_frame(
            held_out,
            {
                **evaluation,
                "id_column": id_column,
                "target_column": target,
                "probability_columns": [probability],
            },
        )
        feature_sets = evaluation.get("feature_sets", [])
        if feature_sets:
            fit_partitions = evaluation.get("fit_partitions", ["train"])
            if not fit_partitions or not set(fit_partitions).issubset(
                {"train", "val", "validation"}
            ):
                raise ValueError(
                    "Secondary evaluation must fit on development partitions only."
                )
            result["secondary_metrics"] = evaluate_secondary_feature_sets(
                scores,
                feature_sets=feature_sets,
                fit_partitions=fit_partitions,
                target_column=target,
                baseline_feature_set=evaluation.get("baseline_feature_set"),
            )
    return scores, mapping, diagnostics, result


def _join_metadata(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    analysis = config["analysis"]
    path = analysis.get("participants_path")
    if not path:
        return predictions
    id_column = analysis.get("id_column", "SEQN")
    metadata = validate_ids(read_table(path), id_column)
    shared = sorted((set(predictions) & set(metadata)) - {id_column})
    if shared:
        compared = predictions.merge(
            metadata[[id_column, *shared]],
            on=id_column,
            suffixes=("", "_metadata"),
            validate="one_to_one",
        )
        for column in shared:
            same = compared[column].eq(compared[f"{column}_metadata"]) | (
                compared[column].isna() & compared[f"{column}_metadata"].isna()
            )
            if not same.all():
                raise ValueError(f"Prediction metadata conflict in {column}.")
    if not set(predictions[id_column]).issubset(set(metadata[id_column])):
        raise ValueError("Missing participant metadata for predictions.")
    return predictions.merge(
        metadata[
            [id_column, *[column for column in metadata if column not in predictions]]
        ],
        on=id_column,
        validate="one_to_one",
    )


def run_motionage(config: dict) -> dict:
    analysis = config["analysis"]
    root = Path(analysis["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "Analysis output directory is not empty; choose a new analysis.output_dir."
        )
    runs = []
    if analysis.get("predictions_path"):
        runs = [("", Path(analysis["predictions_path"]))]
    else:
        source = Path(analysis["participant_predictions_dir"])
        manifest_path = source / "cv_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError("CV manifest is incomplete.")
            for run in manifest["runs"]:
                prediction_path = (source / run["predictions"]).resolve()
                if not prediction_path.is_relative_to(source.resolve()):
                    raise ValueError("CV prediction path escapes its source directory.")
                key = f"{run['model_id']}/{run['fold']}"
                if not (root / key).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Invalid CV output key.")
                runs.append((key, prediction_path))
        else:
            runs = [("", source / "participant_predictions.parquet")]
    if not runs or len({key for key, _ in runs}) != len(runs):
        raise ValueError("Analysis requires unique prediction runs.")
    summaries = []
    for key, path in runs:
        participants = _join_metadata(read_table(path), config)
        scores, mapping, diagnostics, metrics = analyze_participants(
            participants, config
        )
        output = root / key
        output.mkdir(parents=True, exist_ok=True)
        outputs = config.get("outputs", {}) if not key else {}
        score_path = Path(
            outputs.get("participant_scores", output / "participant_scores.csv")
        )
        parameter_path = Path(
            outputs.get("mapping_parameters", output / "mapping_parameters.csv")
        )
        evaluation_path = (
            Path(outputs.get("evaluation_tables", output / "evaluation"))
            / "metrics.json"
        )
        for destination in (score_path, parameter_path, evaluation_path):
            if destination.exists():
                raise ValueError(f"Analysis destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(score_path, index=False)
        pd.DataFrame([asdict(group) for group in mapping.groups]).to_csv(
            parameter_path, index=False
        )
        write_json(output / "mapping.json", mapping_to_jsonable(mapping))
        if not diagnostics.empty:
            diagnostics.to_csv(output / "mapping_diagnostics.csv", index=False)
        write_json(evaluation_path, metrics)
        summaries.append(
            {
                "run": key or "single",
                "participants": len(scores),
                "strata": len(mapping.groups),
            }
        )
    write_json(root / "analysis_summary.json", {"runs": summaries})
    return {"runs": summaries}
