import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cv", [False, True])
def test_documented_example_runs_real_commands_end_to_end(tmp_path, cv):
    output = tmp_path / "workflow"
    command = [
        sys.executable,
        str(ROOT / "examples/synthetic/run_workflow.py"),
        "--output-dir",
        str(output),
    ]
    if cv:
        command.append("--cv")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "example_manifest.json").read_text())
    assert manifest["synthetic"] and not manifest["contains_real_participants"]
    summary = json.loads((output / "analysis/analysis_summary.json").read_text())
    assert len(summary["runs"]) == (6 if cv else 1)
    paths = sorted((output / "analysis").glob("**/participant_scores.csv"))
    assert len(paths) == (6 if cv else 1)
    if not cv:
        prediction_command = [
            sys.executable,
            str(ROOT / "scripts/predict/predict.py"),
            "--config",
            str(ROOT / "configs/examples/prediction.yaml"),
            "--set",
            f"checkpoint_dir={output / 'training'}",
            "--set",
            f"data.parquet_path={output / 'processed/activity_mortstat/activity_mortstat_joined.parquet'}",
            "--set",
            f"data.participants_path={output / 'processed/participants.parquet'}",
            "--set",
            f"output_path={output / 'reused_predictions.parquet'}",
        ]
        subprocess.run(
            prediction_command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        original = pd.read_parquet(
            output / "training/participant_predictions.parquet"
        ).sort_values("SEQN")
        reused = pd.read_parquet(output / "reused_predictions.parquet").sort_values(
            "SEQN"
        )
        np.testing.assert_allclose(original.probability, reused.probability, atol=1e-6)
        evaluation_command = [
            sys.executable,
            str(ROOT / "scripts/analyze/evaluate_predictions.py"),
            "--config",
            str(ROOT / "configs/examples/evaluation.yaml"),
            "--set",
            f"predictions_path={output / 'training/participant_predictions.parquet'}",
            "--set",
            f"output_path={output / 'standalone_metrics.json'}",
            "--set",
            "bootstrap.n_resamples=20",
        ]
        subprocess.run(
            evaluation_command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        independent = json.loads((output / "standalone_metrics.json").read_text())
        assert independent["metrics"][0]["n"] == 16
    for path in paths:
        scores = pd.read_csv(path)
        assert len(scores) == 64
        assert np.isfinite(scores[["MotionAge", "MotionAgeAccel"]].to_numpy()).all()
        assert scores.groupby("SEQN").split.nunique().max() == 1
        parameters = pd.read_csv(path.parent / "mapping_parameters.csv")
        assert (parameters.beta > 0).all()
        assert parameters.n_participants.sum() == len(
            scores.loc[scores.split == "train"]
        )
        metrics = json.loads((path.parent / "evaluation/metrics.json").read_text())
        assert metrics["metrics"][0]["auroc_uncertainty"]["n_resamples_valid"] > 0
