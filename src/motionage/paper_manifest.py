"""Paper model manifest validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

REQUIRED_PAPER_MODEL_FAMILIES = ("gru", "lstm", "transformer")
ARCHITECTURE_FIELDS_BY_FAMILY = {
    "gru": ("hidden_size", "num_layers", "dropout", "hour_emb_dim", "day_emb_dim"),
    "lstm": ("hidden_size", "num_layers", "dropout", "hour_emb_dim", "day_emb_dim"),
    "transformer": (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "hour_emb_dim",
        "day_emb_dim",
        "intensity_proj_dim",
        "max_seq_len",
    ),
}


@dataclass(frozen=True)
class PaperModelManifestEntry:
    """Resolved metadata for one paper-visible model config."""

    model_id: str
    family: str
    source_config_path: str
    source_model_type: str
    task_type: str
    prediction_mode: str | None
    architecture: dict[str, int | float]
    covariates_enabled: bool
    covariate_levels: tuple[str, ...]
    num_numeric_features: int | None
    selection_metric: str
    seq_len: int
    stride_ratio: float
    max_epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class PaperStudyManifest:
    """Resolved metadata for one public paper study manifest."""

    study_id: str
    task: str
    n_folds: int
    fold_root: str
    training_seed: int
    analysis_template_path: str
    official_feature_set: str


@dataclass(frozen=True)
class PaperManifestReadiness:
    """Aggregate readiness counts for one public paper manifest."""

    ready_model_count: int
    not_ready_model_count: int
    total_model_count: int
    all_models_ready: bool


@dataclass(frozen=True)
class MotionAgeAnalysisTemplate:
    """Public metadata resolved from the MotionAge analysis template."""

    analysis_name: str
    output_dir: str
    participant_predictions_dir: str
    id_column: str
    age_column: str
    sex_column: str
    probability_column: str
    target_column: str
    probability_transform: str
    age_bin_column: str
    fit_partitions: tuple[str, ...]
    strata: tuple[str, ...]
    clip_eps: float
    weighted_fit: bool
    clamp_output_to_fit_age_range: bool
    min_stratum_participants: int
    outputs: dict[str, str]


def load_motionage_analysis_template(template_path: str | Path) -> MotionAgeAnalysisTemplate:
    """Load public metadata from a MotionAge analysis template."""
    path = Path(template_path)
    template = _load_yaml(path)
    analysis = _required_mapping(template, "analysis", context="MotionAge analysis template")
    mapping = _required_mapping(template, "mapping", context="MotionAge analysis template")
    outputs = _required_mapping(template, "outputs", context="MotionAge analysis template")

    output_dir = _required_relative_path_text(analysis, "output_dir", context="analysis")
    participant_predictions_dir = _required_relative_path_text(
        analysis,
        "participant_predictions_dir",
        context="analysis",
    )
    resolved_outputs = {
        key: _required_relative_path_text(outputs, key, context="outputs")
        for key in ("participant_scores", "mapping_parameters", "evaluation_tables")
    }
    fit_partitions = _normalize_template_partitions(
        _required_list(mapping, "fit_partitions", context="mapping")
    )
    strata = _required_text_tuple(mapping, "strata", context="mapping")
    probability_transform = _required_text(
        mapping,
        "probability_transform",
        context="mapping",
    ).lower()
    if probability_transform != "logit":
        raise ValueError("mapping.probability_transform must be logit.")

    clip_eps = _required_positive_float(mapping, "clip_eps", context="mapping")
    if clip_eps >= 0.5:
        raise ValueError("mapping.clip_eps must be between 0 and 0.5.")

    return MotionAgeAnalysisTemplate(
        analysis_name=_required_text(analysis, "name", context="analysis"),
        output_dir=output_dir,
        participant_predictions_dir=participant_predictions_dir,
        id_column=_required_text(analysis, "id_column", context="analysis"),
        age_column=_required_text(analysis, "age_column", context="analysis"),
        sex_column=_required_text(analysis, "sex_column", context="analysis"),
        probability_column=_required_text(analysis, "probability_column", context="analysis"),
        target_column=_required_text(analysis, "target_column", context="analysis"),
        probability_transform=probability_transform,
        age_bin_column=_required_text(mapping, "age_bin_column", context="mapping"),
        fit_partitions=fit_partitions,
        strata=strata,
        clip_eps=clip_eps,
        weighted_fit=_required_bool(mapping, "weighted_fit", context="mapping"),
        clamp_output_to_fit_age_range=_required_bool(
            mapping,
            "clamp_output_to_fit_age_range",
            context="mapping",
        ),
        min_stratum_participants=_required_positive_int(
            mapping,
            "min_stratum_participants",
            context="mapping",
        ),
        outputs=resolved_outputs,
    )


def load_paper_study_manifest(manifest_path: str | Path) -> PaperStudyManifest:
    """Load and validate public study metadata from a paper manifest."""
    path = Path(manifest_path)
    manifest = _load_yaml(path)
    study = manifest.get("study")
    if not isinstance(study, dict):
        raise ValueError("Paper model manifest must define a study mapping.")

    repo_root = _infer_repo_root(path)
    analysis_template_path = _required_text(
        study,
        "analysis_template_path",
        context="study",
    )
    resolved_analysis_template_path = _resolve_study_analysis_template_path(
        analysis_template_path,
        repo_root=repo_root,
    )
    load_motionage_analysis_template(resolved_analysis_template_path)

    return PaperStudyManifest(
        study_id=_required_text(study, "study_id", context="study"),
        task=_required_text(study, "task", context="study"),
        n_folds=_required_positive_int(study, "n_folds", context="study"),
        fold_root=_required_text(study, "fold_root", context="study"),
        training_seed=_required_int(study, "training_seed", context="study"),
        analysis_template_path=analysis_template_path,
        official_feature_set=_required_text(study, "official_feature_set", context="study"),
    )


def load_paper_manifest_readiness(manifest_path: str | Path) -> PaperManifestReadiness:
    """Load aggregate ready/not-ready model counts from a paper manifest."""
    path = Path(manifest_path)
    manifest = _load_yaml(path)
    models = _manifest_list(manifest, "models", required=True)
    not_ready_models = _manifest_list(manifest, "not_ready_models", required=False)
    return PaperManifestReadiness(
        ready_model_count=len(models),
        not_ready_model_count=len(not_ready_models),
        total_model_count=len(models) + len(not_ready_models),
        all_models_ready=len(not_ready_models) == 0,
    )


def load_paper_model_manifest(
    manifest_path: str | Path,
) -> tuple[PaperModelManifestEntry, ...]:
    """Load and resolve paper model entries from a mortality-CV manifest."""
    path = Path(manifest_path)
    manifest = _load_yaml(path)
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Paper model manifest must define a non-empty models list.")

    repo_root = _infer_repo_root(path)
    entries: list[PaperModelManifestEntry] = []
    for index, row in enumerate(models):
        if not isinstance(row, dict):
            raise ValueError(f"models[{index}] must be a mapping.")

        model_id = _required_text(row, "model_id", context=f"models[{index}]")
        family = _required_text(row, "family", context=f"models[{index}]").lower()
        source_config_path = _required_text(row, "source_config_path", context=f"models[{index}]")
        source_path = _resolve_public_config_path(source_config_path, repo_root=repo_root)
        source_config = _load_yaml(source_path)

        task_cfg = source_config.get("task")
        if not isinstance(task_cfg, dict):
            raise ValueError(f"{source_config_path} must define a task mapping.")
        task_type = str(task_cfg.get("type", "")).strip()
        if task_type != "binary_classification":
            raise ValueError(f"{source_config_path} task.type must be binary_classification.")
        selection_metric = _required_text(
            task_cfg,
            "selection_metric",
            context=f"{source_config_path}.task",
        )

        model_cfg = source_config.get("model")
        if not isinstance(model_cfg, dict):
            raise ValueError(f"{source_config_path} must define a model mapping.")
        source_model_type = str(model_cfg.get("type", "")).strip().lower()
        _validate_model_type_matches_family(
            family=family,
            source_model_type=source_model_type,
            source_config_path=source_config_path,
        )

        windowing_cfg = source_config.get("windowing")
        if not isinstance(windowing_cfg, dict):
            raise ValueError(f"{source_config_path} must define a windowing mapping.")
        training_cfg = source_config.get("training")
        if not isinstance(training_cfg, dict):
            raise ValueError(f"{source_config_path} must define a training mapping.")

        covariates_enabled, covariate_levels, num_numeric_features = _resolve_covariate_metadata(
            source_config,
            model_cfg,
            source_config_path=source_config_path,
        )
        prediction_mode = model_cfg.get("prediction_mode")
        entries.append(
            PaperModelManifestEntry(
                model_id=model_id,
                family=family,
                source_config_path=source_config_path,
                source_model_type=source_model_type,
                task_type=task_type,
                prediction_mode=str(prediction_mode) if prediction_mode is not None else None,
                architecture=_resolve_architecture_metadata(
                    family=family,
                    model_cfg=model_cfg,
                    source_config_path=source_config_path,
                ),
                covariates_enabled=covariates_enabled,
                covariate_levels=covariate_levels,
                num_numeric_features=num_numeric_features,
                selection_metric=selection_metric,
                seq_len=_required_positive_int(
                    windowing_cfg,
                    "seq_len",
                    context=f"{source_config_path}.windowing",
                ),
                stride_ratio=_required_positive_float(
                    windowing_cfg,
                    "stride_ratio",
                    context=f"{source_config_path}.windowing",
                ),
                max_epochs=_required_positive_int(
                    training_cfg,
                    "max_epochs",
                    context=f"{source_config_path}.training",
                ),
                batch_size=_required_positive_int(
                    training_cfg,
                    "batch_size",
                    context=f"{source_config_path}.training",
                ),
                learning_rate=_required_positive_float(
                    training_cfg,
                    "learning_rate",
                    context=f"{source_config_path}.training",
                ),
            )
        )

    _reject_duplicate_entry_values(entries, field_name="model_id")
    _reject_duplicate_entry_values(entries, field_name="source_config_path")
    return tuple(entries)


def validate_paper_model_manifest(
    manifest_path: str | Path,
    *,
    required_families: Sequence[str] = REQUIRED_PAPER_MODEL_FAMILIES,
) -> tuple[PaperModelManifestEntry, ...]:
    """Validate paper model manifest coverage and source-config consistency."""
    path = Path(manifest_path)
    manifest = _load_yaml(path)
    load_paper_study_manifest(path)
    readiness = load_paper_manifest_readiness(path)
    if not readiness.all_models_ready:
        not_ready_models = manifest.get("not_ready_models", [])
        raise ValueError(f"Paper model manifest still has not_ready_models: {not_ready_models}")

    entries = load_paper_model_manifest(path)
    observed_families = {entry.family for entry in entries}
    missing = sorted({str(family).lower() for family in required_families} - observed_families)
    if missing:
        raise ValueError(f"Missing required model families: {missing}")
    return entries


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _manifest_list(manifest: dict[str, Any], key: str, *, required: bool) -> list[Any]:
    value = manifest.get(key)
    if value is None:
        if required:
            raise ValueError(f"Paper model manifest must define a {key} list.")
        return []
    if not isinstance(value, list):
        raise ValueError(f"Paper model manifest {key} must be a list.")
    return value


def _required_mapping(row: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must define a {key} mapping.")
    return value


def _infer_repo_root(manifest_path: Path) -> Path:
    parts = manifest_path.parts
    if len(parts) >= 3 and parts[-3:-1] == ("configs", "paper"):
        return manifest_path.parents[2]
    return manifest_path.parent


def _resolve_public_config_path(source_config_path: str, *, repo_root: Path) -> Path:
    source_path = Path(source_config_path)
    if source_path.is_absolute():
        raise ValueError(f"source_config_path must be repository-relative: {source_config_path}")
    resolved = repo_root / source_path
    if not resolved.exists():
        raise FileNotFoundError(f"Model source config not found: {source_config_path}")
    return resolved


def _resolve_study_analysis_template_path(template_path: str, *, repo_root: Path) -> Path:
    source_path = Path(template_path)
    if source_path.is_absolute():
        raise ValueError(
            f"study.analysis_template_path must be repository-relative: {template_path}"
        )
    resolved = repo_root / source_path
    if not resolved.exists():
        raise FileNotFoundError(f"Study analysis template not found: {template_path}")
    return resolved


def _required_text(row: dict[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"{context} must define {key}.")
    return str(value).strip()


def _required_relative_path_text(row: dict[str, Any], key: str, *, context: str) -> str:
    value = _required_text(row, key, context=context)
    if Path(value).is_absolute():
        raise ValueError(f"{context}.{key} must be repository-relative: {value}")
    return value


def _required_list(row: dict[str, Any], key: str, *, context: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty list.")
    return value


def _required_text_tuple(row: dict[str, Any], key: str, *, context: str) -> tuple[str, ...]:
    values = _required_list(row, key, context=context)
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{context}.{key} must not contain blanks.")
    return normalized


def _required_bool(row: dict[str, Any], key: str, *, context: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean.")
    return value


def _required_int(row: dict[str, Any], key: str, *, context: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer.")
    return value


def _required_positive_int(row: dict[str, Any], key: str, *, context: str) -> int:
    value = _required_int(row, key, context=context)
    if value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer.")
    return value


def _required_positive_float(row: dict[str, Any], key: str, *, context: str) -> float:
    value = row.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a number.")
    numeric_value = float(value)
    if numeric_value <= 0:
        raise ValueError(f"{context}.{key} must be positive.")
    return numeric_value


def _normalize_template_partitions(partitions: list[Any]) -> tuple[str, ...]:
    aliases = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    normalized: list[str] = []
    for partition in partitions:
        key = str(partition).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unsupported mapping.fit_partitions value: {partition!r}")
        canonical = aliases[key]
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("mapping.fit_partitions must contain at least one partition.")
    return tuple(normalized)


def _resolve_covariate_metadata(
    source_config: dict[str, Any],
    model_cfg: dict[str, Any],
    *,
    source_config_path: str,
) -> tuple[bool, tuple[str, ...], int | None]:
    data_cfg = source_config.get("data")
    covariates_cfg = data_cfg.get("covariates") if isinstance(data_cfg, dict) else None
    if not isinstance(covariates_cfg, dict) or not bool(covariates_cfg.get("enabled")):
        return False, (), None

    levels = covariates_cfg.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"{source_config_path}.data.covariates.levels must be a non-empty list.")
    covariate_levels = tuple(str(level).strip() for level in levels)
    if any(not level for level in covariate_levels):
        raise ValueError(f"{source_config_path}.data.covariates.levels must not contain blanks.")

    return (
        True,
        covariate_levels,
        _required_positive_int(
            model_cfg,
            "num_numeric_features",
            context=f"{source_config_path}.model",
        ),
    )


def _resolve_architecture_metadata(
    *,
    family: str,
    model_cfg: dict[str, Any],
    source_config_path: str,
) -> dict[str, int | float]:
    fields = ARCHITECTURE_FIELDS_BY_FAMILY[family]
    return {
        field: _required_architecture_number(
            model_cfg,
            field,
            context=f"{source_config_path}.model",
        )
        for field in fields
    }


def _required_architecture_number(
    row: dict[str, Any],
    key: str,
    *,
    context: str,
) -> int | float:
    value = row.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a number.")
    if value < 0:
        raise ValueError(f"{context}.{key} must be non-negative.")
    return value


def _validate_model_type_matches_family(
    *,
    family: str,
    source_model_type: str,
    source_config_path: str,
) -> None:
    expected_types = {f"{family}_binary", f"{family}_covariates_binary"}
    if source_model_type not in expected_types:
        raise ValueError(
            f"{source_config_path} family '{family}' does not match "
            f"source model.type '{source_model_type}'."
        )


def _reject_duplicate_entry_values(
    entries: Sequence[PaperModelManifestEntry],
    *,
    field_name: str,
) -> None:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(getattr(entry, field_name))
        counts[value] = counts.get(value, 0) + 1

    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate paper {field_name} values: {duplicates}")
