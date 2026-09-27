"""Local file and command-line conventions shared by executable workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from motionage.config import apply_overrides, load_yaml_config, resolve_config


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xpt", ".zip"}:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    if path.suffix.lower() == ".dat":
        # NCHS 2019 public-use NHANES linked mortality layout (zero-based bounds).
        return pd.read_fwf(
            path, colspecs=[(0, 6), (14, 15), (15, 16), (42, 45), (45, 48)],
            names=["SEQN", "eligstat", "mortstat", "permth_int", "permth_exm"],
            na_values=["."],
        )
    return pd.read_csv(path)


def write_json(path: str | Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, Path):
            return str(item)
        return item

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_ids(frame: pd.DataFrame, id_column: str, *, unique: bool = True) -> pd.DataFrame:
    require_columns(frame, [id_column])
    frame = frame.copy()
    ids = pd.to_numeric(frame[id_column], errors="raise")
    if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any():
        raise ValueError("Participant IDs must be finite, non-missing integers.")
    frame[id_column] = ids.astype("int64")
    if unique and frame[id_column].duplicated().any():
        raise ValueError("Expected one row per participant; duplicate IDs found.")
    return frame


def config_main(function: Callable[[dict], Any], description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = apply_overrides(resolve_config(load_yaml_config(args.config)), args.set)
    try:
        result = function(config)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, default=str))
