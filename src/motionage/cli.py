"""Command-line helpers for public MotionAge checks."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

from motionage.paper_manifest import (
    PaperModelManifestEntry,
    PaperStudyManifest,
    REQUIRED_PAPER_MODEL_FAMILIES,
    load_paper_study_manifest,
    validate_paper_model_manifest,
)

DOCTOR_DEPENDENCIES = ("numpy", "pandas", "pyyaml", "scipy", "scikit-learn", "torch")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MotionAge command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-paper-models":
        return _validate_paper_models(args)
    if args.command == "doctor":
        return _doctor(args)

    parser.print_help()
    return 2


def validate_paper_models_main(argv: Sequence[str] | None = None) -> int:
    """Validate paper model manifest coverage from a console script."""
    parser = _build_validate_paper_models_parser(prog="motionage-validate-paper-models")
    args = parser.parse_args(argv)
    return _validate_paper_models(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motionage")
    parser.add_argument(
        "--version",
        action="version",
        version=f"motionage {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")

    _build_validate_paper_models_parser(
        prog="validate-paper-models",
        parser=subparsers.add_parser(
            "validate-paper-models",
            help="validate paper model manifest coverage and source config consistency",
        ),
    )
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="report MotionAge runtime and dependency versions",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="emit the environment report as JSON",
    )
    return parser


def _build_validate_paper_models_parser(
    *,
    prog: str,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog=prog,
        description="Validate paper model manifest coverage and source config consistency.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"motionage {_package_version()}",
    )
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument(
        "--required-family",
        action="append",
        dest="required_families",
        default=None,
        help="required paper-visible model family; may be provided more than once",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="output_families",
        default=None,
        help="only include this model family in the emitted output; may be provided more than once",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="output_model_ids",
        default=None,
        help="only include this paper model_id in the emitted output; may be provided more than once",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="emit a machine-readable JSON summary",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        dest="emit_markdown",
        help="emit a Markdown table of resolved paper model configs",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        dest="summary_only",
        help="omit per-model details from JSON/Markdown output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        dest="output_path",
        default=None,
        help="write the selected output to a file instead of stdout",
    )
    return parser


def _package_version() -> str:
    try:
        return version("motionage")
    except PackageNotFoundError:
        return "0+unknown"


def _doctor(args: argparse.Namespace) -> int:
    payload = _doctor_payload()
    if args.emit_json:
        print(json.dumps(payload, indent=2))
    else:
        print(_doctor_text(payload))
    return 0


def _doctor_payload() -> dict[str, object]:
    return {
        "motionage": _package_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            dependency: _distribution_version(dependency)
            for dependency in DOCTOR_DEPENDENCIES
        },
    }


def _distribution_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not installed"


def _doctor_text(payload: dict[str, object]) -> str:
    dependencies = payload["dependencies"]
    assert isinstance(dependencies, dict)
    lines = [
        "MotionAge environment report",
        f"motionage: {payload['motionage']}",
        f"python: {payload['python']}",
        f"platform: {payload['platform']}",
        "dependencies:",
    ]
    for name, dependency_version in dependencies.items():
        lines.append(f"  {name}: {dependency_version}")
    return "\n".join(lines)


def _validate_paper_models(args: argparse.Namespace) -> int:
    validation_requirements = _paper_model_validation_requirements(args.required_families)
    required_families = tuple(validation_requirements["required_families"])
    try:
        entries = validate_paper_model_manifest(
            args.manifest_path,
            required_families=required_families,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        output_entries = _filter_manifest_entries(
            entries,
            output_families=args.output_families,
            output_model_ids=args.output_model_ids,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_filters = _paper_model_output_filters(args.output_families, args.output_model_ids)
    family_counts = Counter(entry.family for entry in output_entries)
    if args.emit_json:
        study = load_paper_study_manifest(args.manifest_path)
        payload = _paper_model_manifest_payload(
            args.manifest_path,
            study,
            output_filters,
            validation_requirements,
            output_entries,
            family_counts,
            include_models=not args.summary_only,
        )
        output = f"{json.dumps(payload, indent=2)}\n"
    elif args.emit_markdown:
        study = load_paper_study_manifest(args.manifest_path)
        markdown = _paper_model_manifest_markdown(
            study,
            output_filters,
            validation_requirements,
            output_entries,
            include_models=not args.summary_only,
        )
        output = f"{markdown}\n"
    else:
        output = _paper_model_manifest_text(args.manifest_path, output_entries, family_counts)

    try:
        _emit_output(output, args.output_path)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _filter_manifest_entries(
    entries: Sequence[PaperModelManifestEntry],
    *,
    output_families: Sequence[str] | None,
    output_model_ids: Sequence[str] | None,
) -> tuple[PaperModelManifestEntry, ...]:
    selected = tuple(entries)
    output_filters = _paper_model_output_filters(output_families, output_model_ids)

    if output_filters["families"]:
        normalized_families = tuple(output_filters["families"])
        observed_families = {entry.family for entry in entries}
        unknown_families = sorted(set(normalized_families) - observed_families)
        if unknown_families:
            raise ValueError(f"Unknown paper model family filters: {unknown_families}")
        selected = tuple(entry for entry in selected if entry.family in normalized_families)

    if output_filters["model_ids"]:
        normalized_model_ids = tuple(output_filters["model_ids"])
        observed_model_ids = {entry.model_id for entry in entries}
        unknown_model_ids = sorted(set(normalized_model_ids) - observed_model_ids)
        if unknown_model_ids:
            raise ValueError(f"Unknown paper model_id filters: {unknown_model_ids}")
        selected = tuple(entry for entry in selected if entry.model_id in normalized_model_ids)

    if not selected:
        raise ValueError("No paper model entries matched output filters.")
    return selected


def _paper_model_output_filters(
    output_families: Sequence[str] | None,
    output_model_ids: Sequence[str] | None,
) -> dict[str, list[str]]:
    return {
        "families": list(
            dict.fromkeys(family.strip().lower() for family in output_families or ())
        ),
        "model_ids": list(
            dict.fromkeys(model_id.strip() for model_id in output_model_ids or ())
        ),
    }


def _paper_model_validation_requirements(
    required_families: Sequence[str] | None,
) -> dict[str, list[str]]:
    family_values = required_families or REQUIRED_PAPER_MODEL_FAMILIES
    return {
        "required_families": list(
            dict.fromkeys(family.strip().lower() for family in family_values)
        ),
    }


def _paper_model_manifest_text(
    manifest_path: Path,
    entries: Sequence[PaperModelManifestEntry],
    family_counts: Counter[str],
) -> str:
    lines = [f"Validated {len(entries)} paper model configs from {manifest_path}."]
    for family in sorted(family_counts):
        lines.append(f"{family}: {family_counts[family]}")
    return "\n".join(lines) + "\n"


def _emit_output(output: str, output_path: Path | None) -> None:
    if output_path is None:
        print(output, end="")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def _paper_model_manifest_payload(
    manifest_path: Path,
    study: PaperStudyManifest,
    output_filters: dict[str, list[str]],
    validation_requirements: dict[str, list[str]],
    entries: Sequence[PaperModelManifestEntry],
    family_counts: Counter[str],
    *,
    include_models: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "study": asdict(study),
        "output_filters": output_filters,
        "validation_requirements": validation_requirements,
        "model_summary": _paper_model_summary(entries),
        "model_count": len(entries),
        "family_counts": dict(sorted(family_counts.items())),
    }
    if include_models:
        model_ids_by_family: dict[str, list[str]] = {}
        for entry in entries:
            model_ids_by_family.setdefault(entry.family, []).append(entry.model_id)
        payload["model_ids_by_family"] = dict(sorted(model_ids_by_family.items()))
        payload["models"] = [asdict(entry) for entry in entries]
    return payload


def _paper_model_manifest_markdown(
    study: PaperStudyManifest,
    output_filters: dict[str, list[str]],
    validation_requirements: dict[str, list[str]],
    entries: Sequence[PaperModelManifestEntry],
    *,
    include_models: bool = True,
) -> str:
    sections = [
        _paper_study_manifest_markdown(study),
        _paper_model_output_filters_markdown(output_filters),
        _paper_model_validation_requirements_markdown(validation_requirements),
        _paper_model_summary_markdown(entries),
    ]
    if include_models:
        sections.append(_paper_model_entries_markdown(entries))
    return "\n\n".join(sections)


def _paper_study_manifest_markdown(study: PaperStudyManifest) -> str:
    lines = [
        "## Study",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for field, value in asdict(study).items():
        lines.append(f"| {field} | {value} |")
    return "\n".join(lines)


def _paper_model_output_filters_markdown(output_filters: dict[str, list[str]]) -> str:
    lines = [
        "## Output Filters",
        "",
        "| Filter | Values |",
        "| --- | --- |",
    ]
    for key in ("families", "model_ids"):
        values = ", ".join(output_filters[key]) if output_filters[key] else "-"
        lines.append(f"| {key} | {values} |")
    return "\n".join(lines)


def _paper_model_validation_requirements_markdown(
    validation_requirements: dict[str, list[str]],
) -> str:
    lines = [
        "## Validation Requirements",
        "",
        "| Requirement | Values |",
        "| --- | --- |",
    ]
    required_families = validation_requirements["required_families"]
    values = ", ".join(required_families) if required_families else "-"
    lines.append(f"| required_families | {values} |")
    return "\n".join(lines)


def _paper_model_summary(
    entries: Sequence[PaperModelManifestEntry],
) -> dict[str, dict[str, int]]:
    return {
        "families": _sorted_counter(entry.family for entry in entries),
        "prediction_modes": _sorted_counter(
            entry.prediction_mode or "none" for entry in entries
        ),
        "covariates": _sorted_counter(
            "enabled" if entry.covariates_enabled else "disabled" for entry in entries
        ),
        "covariate_levels": _sorted_counter(
            ", ".join(entry.covariate_levels) if entry.covariate_levels else "none"
            for entry in entries
        ),
    }


def _sorted_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _paper_model_summary_markdown(entries: Sequence[PaperModelManifestEntry]) -> str:
    summary = _paper_model_summary(entries)
    lines = [
        "## Model Summary",
        "",
        "| Metric | Value | Count |",
        "| --- | --- | --- |",
    ]
    metric_labels = (
        ("families", "family"),
        ("prediction_modes", "prediction_mode"),
        ("covariates", "covariates"),
        ("covariate_levels", "covariate_level"),
    )
    for key, label in metric_labels:
        for value, count in summary[key].items():
            lines.append(f"| {label} | {value} | {count} |")
    return "\n".join(lines)


def _paper_model_entries_markdown(entries: Sequence[PaperModelManifestEntry]) -> str:
    lines = [
        "## Models",
        "",
        "| Family | Model ID | Model type | Prediction mode | Architecture | Covariates | "
        "Levels | Numeric features | Seq len | Stride | Epochs | Batch | LR | "
        "Selection | Source config |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- |",
    ]
    for entry in entries:
        prediction_mode = entry.prediction_mode or "-"
        architecture = _paper_model_architecture_markdown(entry.architecture)
        covariates = "yes" if entry.covariates_enabled else "no"
        covariate_levels = ", ".join(entry.covariate_levels) if entry.covariate_levels else "-"
        num_numeric_features = (
            str(entry.num_numeric_features) if entry.num_numeric_features is not None else "-"
        )
        lines.append(
            "| "
            f"{entry.family} | "
            f"{entry.model_id} | "
            f"{entry.source_model_type} | "
            f"{prediction_mode} | "
            f"{architecture} | "
            f"{covariates} | "
            f"{covariate_levels} | "
            f"{num_numeric_features} | "
            f"{entry.seq_len} | "
            f"{entry.stride_ratio} | "
            f"{entry.max_epochs} | "
            f"{entry.batch_size} | "
            f"{entry.learning_rate} | "
            f"{entry.selection_metric} | "
            f"{entry.source_config_path} |"
        )
    return "\n".join(lines)


def _paper_model_architecture_markdown(architecture: dict[str, int | float]) -> str:
    return "; ".join(f"{key}={value}" for key, value in architecture.items())
