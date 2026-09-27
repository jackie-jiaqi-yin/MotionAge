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
