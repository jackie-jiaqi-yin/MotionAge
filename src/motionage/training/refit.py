"""Utilities for winner-trial final refit workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def resolve_final_refit_config(
    raw_config: dict[str, Any],
    *,
    require_enabled: bool = True,
) -> dict[str, Any]:
    """Normalize the top-level final-refit config block."""
    refit_cfg = raw_config.get("final_refit")
    if not isinstance(refit_cfg, dict):
        if require_enabled:
            raise ValueError("Config must define top-level final_refit.enabled=true.")
        return {"enabled": False, "run_after_tuning": False}

    enabled = bool(refit_cfg.get("enabled", False))
    if require_enabled and not enabled:
        raise ValueError("Config must define top-level final_refit.enabled=true.")
    if not enabled:
        return {"enabled": False, "run_after_tuning": False}

    epochs = int(refit_cfg.get("epochs", 18))
    if epochs < 1:
        raise ValueError(f"final_refit.epochs must be >= 1, got {epochs}.")

    return {
        "enabled": True,
        "run_after_tuning": bool(refit_cfg.get("run_after_tuning", True)),
        "epochs": epochs,
        "output_subdir": str(refit_cfg.get("output_subdir", "winner_refit_trainval")),
        "combine_train_val": bool(refit_cfg.get("combine_train_val", True)),
        "use_winner_threshold": bool(refit_cfg.get("use_winner_threshold", True)),
    }


def resolve_winner_trial(tuning_dir: Path, explicit_trial: int | None) -> int:
    """Resolve the selected tuning trial from explicit input or summary artifacts."""
    if explicit_trial is not None:
        return int(explicit_trial)

    binary_summary_path = tuning_dir / "binary_model_selection_summary.json"
    if binary_summary_path.exists():
        payload = json.loads(binary_summary_path.read_text(encoding="utf-8"))
        if "analysis_trial" in payload:
            return int(payload["analysis_trial"])

    topk_path = tuning_dir / "topk_eval_summary.json"
    if topk_path.exists():
        rows = json.loads(topk_path.read_text(encoding="utf-8"))
        if rows:
            ranked = sorted(rows, key=lambda row: int(row.get("rank", 999999)))
            return int(ranked[0]["trial"])

    summary_path = tuning_dir / "tuning_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if rows:
            best = max(rows, key=lambda row: float(row.get("objective_value", float("-inf"))))
            return int(best["trial"])

    raise FileNotFoundError(
        "Could not infer a winner trial. Provide explicit_trial or a supported tuning summary file."
    )


def build_trainval_refit_splits(id_splits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Combine train and validation participant IDs for final refit."""
    if "train" not in id_splits or "val" not in id_splits or "test" not in id_splits:
        raise KeyError("train, val, and test participant splits are required for train+val refit.")

    train_val_ids = np.sort(np.unique(np.concatenate([id_splits["train"], id_splits["val"]])))
    return {
        "train": train_val_ids,
        "test": np.sort(np.unique(np.asarray(id_splits["test"]))),
    }


def resolve_fixed_epoch_plan(
    *,
    max_epochs: int,
    fixed_epochs: int | str | None,
    resume_from_checkpoint: bool,
    resumed_from_epoch: int,
) -> dict[str, int]:
    """Resolve fixed-epoch refit start and remaining epoch counts."""
    epoch_budget = int(fixed_epochs if fixed_epochs is not None else max_epochs)
    if fixed_epochs is None and int(max_epochs) < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}.")
    if epoch_budget < 1:
        raise ValueError(f"fixed_epochs must be >= 1, got {epoch_budget}.")

    resumed_epoch = int(resumed_from_epoch) if resume_from_checkpoint else 0
    start_epoch = resumed_epoch + 1 if resumed_epoch > 0 else 1
    return {
        "epoch_budget": epoch_budget,
        "start_epoch": start_epoch,
        "resumed_from_epoch": resumed_epoch,
        "epochs_remaining": max(epoch_budget - start_epoch + 1, 0),
    }
