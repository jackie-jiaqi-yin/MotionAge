"""Reusable metric and uncertainty evaluation for local participant predictions."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from motionage.evaluation.metrics import binary_probability_metrics
from motionage.stats.bootstrap import bootstrap_binary_auroc
from motionage.stats.paired_auc import (
    fold_structured_paired_bootstrap_auc_delta,
    paired_auc_delta,
    paired_bootstrap_auc_delta,
)
from motionage.workflow_io import read_table, require_columns, validate_ids, write_json


def evaluate_prediction_frame(frame: pd.DataFrame, config: dict) -> dict:
    id_column = config.get("id_column", "SEQN")
    target = config.get("target_column", "mortstat")
    scores = config.get("probability_columns", ["probability"])
    if not scores or len(scores) != len(set(scores)):
        raise ValueError("probability_columns must contain unique probability fields.")
    frame = validate_ids(frame, id_column)
    require_columns(frame, [target, *scores])
    if not frame[target].isin([0, 1]).all() or frame[target].nunique() != 2:
        raise ValueError(
            "Evaluation requires both binary classes without missing targets."
        )
    if (
        not np.isfinite(frame[scores].to_numpy(dtype=float)).all()
        or not frame[scores].ge(0).all().all()
        or not frame[scores].le(1).all().all()
    ):
        raise ValueError(
            "Predicted probabilities must be finite and between zero and one."
        )
    boot = config.get("bootstrap", {})
    count = int(boot.get("n_resamples", 1000))
    ci_level = float(boot.get("ci_level", 0.95))
    seed = int(boot.get("random_seed", 42))
    if count < 0:
        raise ValueError(
            "n_resamples must be nonnegative; zero disables uncertainty estimation."
        )
    metrics = []
    for score in scores:
        row = {
            "probability_column": score,
            **binary_probability_metrics(
                frame[target].to_numpy(), frame[score].to_numpy()
            ),
        }
        if count:
            interval = bootstrap_binary_auroc(
                frame[target].to_numpy(),
                frame[score].to_numpy(),
                n_resamples=count,
                ci_level=ci_level,
                random_seed=seed,
            )
            if interval["n_resamples_valid"] == 0:
                raise ValueError(
                    "No valid bootstrap resamples; increase the cohort or resample count."
                )
            row["auroc_uncertainty"] = {
                **interval,
                "resampling_unit": "participant",
                "stratified": False,
                "random_seed": seed,
            }
        metrics.append(row)
    comparisons = []
    if count and len(scores) > 1:
        if ci_level != 0.95:
            raise ValueError("Paired AUROC comparisons currently use 95% intervals.")
        fold_column = config.get("fold_column")
        if fold_column:
            require_columns(frame, [fold_column])
            if frame[fold_column].isna().any():
                raise ValueError("Fold identifiers cannot be missing.")
            if (frame.groupby(fold_column)[target].nunique() < 2).any():
                raise ValueError(
                    "Every fold needs both outcome classes for fold-structured uncertainty."
                )
        for left, right in combinations(scores, 2):
            paired = pd.DataFrame(
                {
                    "SEQN": frame[id_column],
                    "target": frame[target],
                    "score_left": frame[left],
                    "score_right": frame[right],
                    "fold": frame[fold_column] if fold_column else 0,
                    "RIDAGEYR": frame.get(config.get("age_column", "RIDAGEYR"), np.nan),
                }
            )
            interval = (
                fold_structured_paired_bootstrap_auc_delta(
                    paired, n_resamples=count, random_seed=seed
                )
                if fold_column
                else paired_bootstrap_auc_delta(
                    paired, n_resamples=count, random_seed=seed, stratified=True
                )
            )
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "pooled_observed": paired_auc_delta(paired),
                    "interval": interval,
                    "random_seed": seed,
                }
            )
    return {
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "uncertainty_scope": "fixed_predictions_without_model_refitting",
    }


def run_evaluation(config: dict) -> dict:
    frame = read_table(config["predictions_path"])
    if config.get("split") is not None:
        split_column = config.get("split_column", "split")
        require_columns(frame, [split_column])
        frame = frame.loc[frame[split_column] == config["split"]]
    result = evaluate_prediction_frame(frame, config)
    output = Path(config["output_path"])
    if output.exists():
        raise ValueError("Evaluation output already exists; choose a new output_path.")
    write_json(output, result)
    return result
