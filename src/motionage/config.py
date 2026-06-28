"""Configuration helpers for MotionAge experiments and replay scripts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


_PUBLIC_CONFIG_PATH_KEY_TERMS = (
    "checkpoint",
    "dir",
    "file",
    "path",
    "root",
    "uri",
)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, apply anchor inheritance, and resolve local references."""
    config_path = Path(path)
    raw = _load_raw_yaml(config_path)
    merged = _apply_anchor_inheritance(raw, config_path=config_path)
    return _resolve_experiment_references(merged, config_path=config_path)


def resolve_config(raw: dict[str, Any], trial: Any = None, _prefix: str = "") -> dict[str, Any]:
    """Resolve `{default, search}` leaves for single-run or tuning contexts."""
    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        full_key = f"{_prefix}.{key}" if _prefix else key
        if isinstance(value, dict):
            if "default" in value:
                resolved[key] = _resolve_leaf(full_key, value, trial)
            else:
                resolved[key] = resolve_config(value, trial, _prefix=full_key)
        else:
            resolved[key] = copy.deepcopy(value)
    return resolved


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-path CLI overrides such as `model.hidden_size=256`."""
    resolved = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.split(".")
        target = resolved
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = _parse_scalar(raw_value)
    return resolved


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a resolved config as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def public_config_snapshot(
    config: dict[str, Any],
    *,
    redact_paths: bool = True,
    redacted_value: str = "<redacted>",
) -> dict[str, Any]:
    """Return a public-safe config snapshot for reports and PR summaries."""
    snapshot = copy.deepcopy(config)
    if not redact_paths:
        return snapshot
    return _redact_path_like_values(snapshot, redacted_value=redacted_value)


def public_config_diff(
    reference_config: dict[str, Any],
    candidate_config: dict[str, Any],
    *,
    redact_paths: bool = True,
) -> list[dict[str, Any]]:
    """Return public-safe changed config leaves for report tables."""
    reference_snapshot = public_config_snapshot(reference_config, redact_paths=redact_paths)
    candidate_snapshot = public_config_snapshot(candidate_config, redact_paths=redact_paths)
    reference_leaves = _flatten_config_leaves(reference_snapshot)
    candidate_leaves = _flatten_config_leaves(candidate_snapshot)

    rows: list[dict[str, Any]] = []
    for key in sorted(set(reference_leaves) | set(candidate_leaves)):
        if redact_paths and _is_path_like_config_path(key):
            continue
        reference_value = reference_leaves.get(key, _MISSING_CONFIG_VALUE)
        candidate_value = candidate_leaves.get(key, _MISSING_CONFIG_VALUE)
        if reference_value == candidate_value:
            continue
        rows.append(
            {
                "key": key,
                "reference_value": _public_diff_value(reference_value),
                "candidate_value": _public_diff_value(candidate_value),
            }
        )
    return rows


def public_config_rows(
    config: dict[str, Any],
    *,
    redact_paths: bool = True,
) -> list[dict[str, Any]]:
    """Return public-safe flattened config rows for report tables."""
    snapshot = public_config_snapshot(config, redact_paths=redact_paths)
    leaves = _flatten_config_leaves(snapshot)

    rows: list[dict[str, Any]] = []
    for key in sorted(leaves):
        if redact_paths and _is_path_like_config_path(key):
            continue
        rows.append({"key": key, "value": copy.deepcopy(leaves[key])})
    return rows


_MISSING_CONFIG_VALUE = object()


def _resolve_leaf(full_key: str, node: dict[str, Any], trial: Any) -> Any:
    if trial is None or "search" not in node:
        return copy.deepcopy(node["default"])

    search = node["search"]
    search_type = search["type"]
    if search_type == "categorical":
        return trial.suggest_categorical(full_key, search["choices"])
    if search_type == "int":
        return trial.suggest_int(
            full_key,
            search["low"],
            search["high"],
            step=search.get("step", 1),
            log=search.get("log", False),
        )
    if search_type == "float":
        kwargs: dict[str, Any] = {
            "name": full_key,
            "low": search["low"],
            "high": search["high"],
        }
        if "step" in search:
            kwargs["step"] = search["step"]
        if search.get("log", False):
            kwargs["log"] = True
        return trial.suggest_float(**kwargs)
    raise ValueError(f"Unknown search type '{search_type}' for key '{full_key}'.")


def _parse_scalar(value: str) -> int | float | bool | str:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _redact_path_like_values(value: Any, *, redacted_value: str) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_path_like_config_key(str(key)):
                redacted[key] = redacted_value
            else:
                redacted[key] = _redact_path_like_values(nested, redacted_value=redacted_value)
        return redacted
    if isinstance(value, list):
        return [_redact_path_like_values(item, redacted_value=redacted_value) for item in value]
    return copy.deepcopy(value)


def _flatten_config_leaves(value: Any, *, _prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value and _prefix:
            return {_prefix: {}}
        leaves: dict[str, Any] = {}
        for key, nested in value.items():
            nested_prefix = f"{_prefix}.{key}" if _prefix else str(key)
            leaves.update(_flatten_config_leaves(nested, _prefix=nested_prefix))
        return leaves
    return {_prefix: copy.deepcopy(value)}


def _public_diff_value(value: Any) -> Any:
    if value is _MISSING_CONFIG_VALUE:
        return None
    return copy.deepcopy(value)


def _is_path_like_config_path(key_path: str) -> bool:
    return any(_is_path_like_config_key(part) for part in key_path.split("."))


def _is_path_like_config_key(key: str) -> bool:
    normalized = key.lower()
    parts = [part for part in normalized.replace("-", "_").split("_") if part]
    return any(part in _PUBLIC_CONFIG_PATH_KEY_TERMS for part in parts)


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _apply_anchor_inheritance(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    experiment_cfg = raw.get("experiment")
    if not isinstance(experiment_cfg, dict):
        return raw

    inherit_cfg = experiment_cfg.get("inherit_from_anchor")
    if inherit_cfg in (None, False):
        return raw
    if inherit_cfg is True:
        inherit_cfg = {}
    if not isinstance(inherit_cfg, dict):
        raise ValueError("experiment.inherit_from_anchor must be a mapping or boolean.")

    init_checkpoint = experiment_cfg.get("init_checkpoint")
    if init_checkpoint in (None, "", "null"):
        raise ValueError("experiment.inherit_from_anchor requires experiment.init_checkpoint.")

    anchor_config_ref = inherit_cfg.get("config_path", "auto")
    if anchor_config_ref in (None, "", "auto"):
        checkpoint_path = _resolve_reference_path(init_checkpoint, config_path=config_path)
        anchor_config_path = checkpoint_path.parent.parent / "config.yaml"
    else:
        anchor_config_path = _resolve_reference_path(anchor_config_ref, config_path=config_path)
    if not anchor_config_path.exists():
        raise FileNotFoundError(f"Anchor config not found: {anchor_config_path}")

    source = resolve_config(_load_raw_yaml(anchor_config_path), trial=None)
    merged = copy.deepcopy(raw)

    if inherit_cfg.get("windowing", False):
        _inherit_mapping_section(
            merged=merged,
            source=source,
            section="windowing",
            context=f"{config_path}: experiment.inherit_from_anchor.windowing",
        )

    model_keys = [str(key) for key in inherit_cfg.get("model_keys", [])]
    if model_keys:
        _inherit_mapping_keys(
            merged=merged,
            source=source,
            section="model",
            keys=model_keys,
            context=f"{config_path}: experiment.inherit_from_anchor.model_keys",
        )

    data_keys = [str(key) for key in inherit_cfg.get("data_keys", [])]
    if data_keys:
        _inherit_mapping_keys(
            merged=merged,
            source=source,
            section="data",
            keys=data_keys,
            context=f"{config_path}: experiment.inherit_from_anchor.data_keys",
        )

    return merged


def _resolve_experiment_references(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(raw)
    experiment_cfg = resolved.get("experiment")
    if not isinstance(experiment_cfg, dict):
        return resolved

    checkpoint_ref = experiment_cfg.get("init_checkpoint")
    if checkpoint_ref not in (None, "", "null"):
        experiment_cfg["init_checkpoint"] = str(
            _resolve_reference_path(checkpoint_ref, config_path=config_path)
        )
    return resolved


def _resolve_reference_path(reference: str | Path, *, config_path: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve()

    for base in (config_path.parent, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def _inherit_mapping_section(
    *,
    merged: dict[str, Any],
    source: dict[str, Any],
    section: str,
    context: str,
) -> None:
    source_section = source.get(section)
    if not isinstance(source_section, dict):
        raise ValueError(f"{context} could not find mapping section '{section}' in anchor config.")

    target_section = merged.get(section)
    if target_section is None:
        target_section = {}
        merged[section] = target_section
    if not isinstance(target_section, dict):
        raise ValueError(f"{context} expected '{section}' to be a mapping when present.")

    for key, value in source_section.items():
        if key in target_section:
            if target_section[key] != value:
                raise ValueError(
                    f"{context} conflict for {section}.{key}: "
                    f"target={target_section[key]!r} anchor={value!r}"
                )
            continue
        target_section[key] = copy.deepcopy(value)


def _inherit_mapping_keys(
    *,
    merged: dict[str, Any],
    source: dict[str, Any],
    section: str,
    keys: list[str],
    context: str,
) -> None:
    source_section = source.get(section)
    if not isinstance(source_section, dict):
        raise ValueError(f"{context} could not find mapping section '{section}' in anchor config.")

    target_section = merged.get(section)
    if target_section is None:
        target_section = {}
        merged[section] = target_section
    if not isinstance(target_section, dict):
        raise ValueError(f"{context} expected '{section}' to be a mapping when present.")

    for key in keys:
        if key not in source_section:
            raise ValueError(f"{context} could not find {section}.{key} in anchor config.")
        value = source_section[key]
        if key in target_section:
            if target_section[key] != value:
                raise ValueError(
                    f"{context} conflict for {section}.{key}: "
                    f"target={target_section[key]!r} anchor={value!r}"
                )
            continue
        target_section[key] = copy.deepcopy(value)
