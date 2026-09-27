import json

import numpy as np
import pandas as pd
import pytest

from motionage.analysis.motionage.pipeline import analyze_participants, run_motionage
from motionage.evaluation.workflow import evaluate_prediction_frame


def participants():
    rows = []
    for split in ["train", "val", "test"]:
        for sex in [1, 2]:
            for age in range(40, 76, 5):
                rows.append(
                    {
                        "SEQN": len(rows) + 1,
                        "split": split,
                        "RIDAGEYR": age,
                        "RIAGENDR": sex,
                        "mortstat": int(age >= 60),
                        "probability": 1 / (1 + np.exp(-(-6 + 0.1 * age))),
                    }
                )
    return pd.DataFrame(rows)


def config():
    return {
        "analysis": {},
        "mapping": {"fit_partitions": ["train"], "min_stratum_participants": 2},
        "evaluation": {"bootstrap": {"n_resamples": 20, "random_seed": 8}},
    }


def test_complete_mapping_and_training_partition_isolation():
    frame = participants()
    scored, mapping, _, metrics = analyze_participants(frame, config())
    np.testing.assert_allclose(scored.MotionAge, scored.RIDAGEYR, atol=1e-8)
    assert metrics["metrics"][0]["auroc"] == 1
    changed = frame.copy()
    changed.loc[changed.split == "test", "probability"] = 0.9
    _, second, _, _ = analyze_participants(changed, config())
    assert mapping == second
    bad = config()
    bad["mapping"]["fit_partitions"] = ["test"]
    with pytest.raises(ValueError, match="never test"):
        analyze_participants(frame, bad)


def test_run_and_reuse_saved_mapping(tmp_path):
    source = tmp_path / "predictions.parquet"
    participants().to_parquet(source)
    settings = config()
    settings["analysis"] = {
        "predictions_path": str(source),
        "output_dir": str(tmp_path / "analysis"),
    }
    result = run_motionage(settings)
    assert result["runs"][0]["participants"] == len(participants())
    assert (tmp_path / "analysis/mapping.json").exists()
    expected = pd.read_csv(tmp_path / "analysis/participant_scores.csv")
    settings["mapping"]["parameters_path"] = str(tmp_path / "analysis/mapping.json")
    settings["analysis"]["output_dir"] = str(tmp_path / "apply")
    settings["evaluation"]["enabled"] = False
    participants().drop(columns=["mortstat", "split"]).to_parquet(source)
    run_motionage(settings)
    actual = pd.read_csv(tmp_path / "apply/participant_scores.csv")
    np.testing.assert_allclose(actual.MotionAge, expected.MotionAge)


def test_unknown_stratum_and_nonpositive_mapping_are_rejected():
    frame = participants()
    frame.loc[frame.split == "test", "RIAGENDR"] = 3
    with pytest.raises(ValueError, match="strata"):
        analyze_participants(frame, config())
    frame = participants()
    frame["probability"] = 1 - frame.probability
    with pytest.raises(ValueError, match="positive beta"):
        analyze_participants(frame, config())


def test_general_uncertainty_and_fold_pairing_are_repeatable():
    frame = participants().loc[lambda frame: frame.split == "test"].copy()
    frame["comparator"] = 1 - frame.probability
    frame["fold"] = frame.RIAGENDR
    settings = {
        "probability_columns": ["probability", "comparator"],
        "fold_column": "fold",
        "bootstrap": {"n_resamples": 20, "random_seed": 9},
    }
    first = evaluate_prediction_frame(frame, settings)
    assert first == evaluate_prediction_frame(frame, settings)
    assert first["paired_comparisons"][0]["interval"]["observed_auc_delta"] == 1
    assert first["paired_comparisons"][0]["interval"]["valid_folds"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_prediction_frame(pd.concat([frame, frame.iloc[:1]]), settings)


def test_cv_analysis_keeps_runs_separate(tmp_path):
    source = tmp_path / "cv"
    source.mkdir()
    participants().to_parquet(source / "one.parquet")
    participants().to_parquet(source / "two.parquet")
    (source / "cv_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "runs": [
                    {"model_id": "gru", "fold": "fold_0", "predictions": "one.parquet"},
                    {"model_id": "gru", "fold": "fold_1", "predictions": "two.parquet"},
                ],
            }
        )
    )
    settings = config()
    settings["analysis"] = {
        "participant_predictions_dir": str(source),
        "output_dir": str(tmp_path / "analysis"),
    }
    assert len(run_motionage(settings)["runs"]) == 2
    assert (tmp_path / "analysis/gru/fold_0/mapping.json").exists()
    assert (tmp_path / "analysis/gru/fold_1/mapping.json").exists()


def test_metadata_conflict_is_not_silently_ignored(tmp_path):
    frame = participants()
    frame.to_parquet(tmp_path / "pred.parquet")
    metadata = frame[["SEQN", "RIDAGEYR"]].copy()
    metadata["RIDAGEYR"] += 1
    metadata.to_parquet(tmp_path / "meta.parquet")
    settings = config()
    settings["analysis"] = {
        "predictions_path": str(tmp_path / "pred.parquet"),
        "participants_path": str(tmp_path / "meta.parquet"),
        "output_dir": str(tmp_path / "analysis"),
    }
    with pytest.raises(ValueError, match="metadata conflict"):
        run_motionage(settings)
