from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import motionage.training as training


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_final_refit_config_normalizes_defaults_and_overrides() -> None:
    assert hasattr(training, "resolve_final_refit_config")

    defaults = training.resolve_final_refit_config({"final_refit": {"enabled": True}})
    assert defaults == {
        "enabled": True,
        "run_after_tuning": True,
        "epochs": 18,
        "output_subdir": "winner_refit_trainval",
        "combine_train_val": True,
        "use_winner_threshold": True,
    }

    configured = training.resolve_final_refit_config(
        {
            "final_refit": {
                "enabled": True,
                "run_after_tuning": False,
                "epochs": "7",
                "output_subdir": "custom_refit",
                "combine_train_val": False,
                "use_winner_threshold": False,
            }
        }
    )
    assert configured == {
        "enabled": True,
        "run_after_tuning": False,
        "epochs": 7,
        "output_subdir": "custom_refit",
        "combine_train_val": False,
        "use_winner_threshold": False,
    }


def test_resolve_final_refit_config_rejects_missing_disabled_or_invalid_configs() -> None:
    assert hasattr(training, "resolve_final_refit_config")

    assert training.resolve_final_refit_config({}, require_enabled=False) == {
        "enabled": False,
        "run_after_tuning": False,
    }
    with pytest.raises(ValueError, match="final_refit.enabled=true"):
        training.resolve_final_refit_config({})
    with pytest.raises(ValueError, match="final_refit.enabled=true"):
        training.resolve_final_refit_config({"final_refit": {"enabled": False}})
    with pytest.raises(ValueError, match="epochs"):
        training.resolve_final_refit_config({"final_refit": {"enabled": True, "epochs": 0}})


def test_build_trainval_refit_splits_combines_train_and_validation_ids() -> None:
    assert hasattr(training, "build_trainval_refit_splits")

    splits = training.build_trainval_refit_splits(
        {
            "train": np.array([3, 1, 1]),
            "val": np.array([2, 3]),
            "test": np.array([9, 8, 8]),
        }
    )

    np.testing.assert_array_equal(splits["train"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(splits["test"], np.array([8, 9]))
    with pytest.raises(KeyError, match="train, val, and test"):
        training.build_trainval_refit_splits({"train": np.array([1]), "test": np.array([2])})


def test_resolve_winner_trial_uses_explicit_or_ranked_summary_files(tmp_path: Path) -> None:
    assert hasattr(training, "resolve_winner_trial")

    assert training.resolve_winner_trial(tmp_path, explicit_trial=42) == 42

    binary_summary_dir = tmp_path / "binary"
    binary_summary_dir.mkdir()
    write_json(binary_summary_dir / "binary_model_selection_summary.json", {"analysis_trial": 7})
    assert training.resolve_winner_trial(binary_summary_dir, explicit_trial=None) == 7

    topk_dir = tmp_path / "topk"
    topk_dir.mkdir()
    write_json(
        topk_dir / "topk_eval_summary.json",
        [
            {"trial": 12, "rank": 2},
            {"trial": 5, "rank": 1},
        ],
    )
    assert training.resolve_winner_trial(topk_dir, explicit_trial=None) == 5

    tuning_summary_dir = tmp_path / "summary"
    tuning_summary_dir.mkdir()
    write_json(
        tuning_summary_dir / "tuning_summary.json",
        [
            {"trial": 1, "objective_value": 0.2},
            {"trial": 3, "objective_value": 0.5},
        ],
    )
    assert training.resolve_winner_trial(tuning_summary_dir, explicit_trial=None) == 3

    with pytest.raises(FileNotFoundError, match="winner trial"):
        training.resolve_winner_trial(tmp_path / "missing", explicit_trial=None)


def test_resolve_fixed_epoch_plan_uses_max_epochs_without_resume() -> None:
    assert hasattr(training, "resolve_fixed_epoch_plan")

    plan = training.resolve_fixed_epoch_plan(
        max_epochs=12,
        fixed_epochs=None,
        resume_from_checkpoint=False,
        resumed_from_epoch=5,
    )

    assert plan == {
        "epoch_budget": 12,
        "start_epoch": 1,
        "resumed_from_epoch": 0,
        "epochs_remaining": 12,
    }


def test_resolve_fixed_epoch_plan_uses_resume_start_epoch() -> None:
    assert hasattr(training, "resolve_fixed_epoch_plan")

    plan = training.resolve_fixed_epoch_plan(
        max_epochs=12,
        fixed_epochs="8",
        resume_from_checkpoint=True,
        resumed_from_epoch=3,
    )

    assert plan == {
        "epoch_budget": 8,
        "start_epoch": 4,
        "resumed_from_epoch": 3,
        "epochs_remaining": 5,
    }


def test_resolve_fixed_epoch_plan_handles_completed_resume_budget() -> None:
    assert hasattr(training, "resolve_fixed_epoch_plan")

    plan = training.resolve_fixed_epoch_plan(
        max_epochs=12,
        fixed_epochs=8,
        resume_from_checkpoint=True,
        resumed_from_epoch=10,
    )

    assert plan == {
        "epoch_budget": 8,
        "start_epoch": 11,
        "resumed_from_epoch": 10,
        "epochs_remaining": 0,
    }


def test_resolve_fixed_epoch_plan_rejects_invalid_epoch_budgets() -> None:
    assert hasattr(training, "resolve_fixed_epoch_plan")

    with pytest.raises(ValueError, match="fixed_epochs must be >= 1"):
        training.resolve_fixed_epoch_plan(
            max_epochs=12,
            fixed_epochs=0,
            resume_from_checkpoint=False,
            resumed_from_epoch=0,
        )
    with pytest.raises(ValueError, match="max_epochs must be >= 1"):
        training.resolve_fixed_epoch_plan(
            max_epochs=0,
            fixed_epochs=None,
            resume_from_checkpoint=False,
            resumed_from_epoch=0,
        )
