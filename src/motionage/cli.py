"""Command-line helpers for public MotionAge checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from motionage.paper_manifest import (
    PaperModelManifestEntry,
    PaperStudyManifest,
    REQUIRED_PAPER_MODEL_FAMILIES,
    load_paper_study_manifest,
    validate_paper_model_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MotionAge command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-paper-models":
        return _validate_paper_models(args)

    parser.print_help()
    return 2


def validate_paper_models_main(argv: Sequence[str] | None = None) -> int:
    """Validate paper model manifest coverage from a console script."""
    parser = _build_validate_paper_models_parser(prog="motionage-validate-paper-models")
    args = parser.parse_args(argv)
    return _validate_paper_models(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motionage")
    subparsers = parser.add_subparsers(dest="command")

    _build_validate_paper_models_parser(
        prog="validate-paper-models",
        parser=subparsers.add_parser(
            "validate-paper-models",
            help="validate paper model manifest coverage and source config consistency",
        ),
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
        "--output",
        type=Path,
        dest="output_path",
        default=None,
        help="write the selected output to a file instead of stdout",
    )
    return parser


def _validate_paper_models(args: argparse.Namespace) -> int:
    required_families = tuple(args.required_families or REQUIRED_PAPER_MODEL_FAMILIES)
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
            output_entries,
            family_counts,
        )
        output = f"{json.dumps(payload, indent=2)}\n"
    elif args.emit_markdown:
        study = load_paper_study_manifest(args.manifest_path)
        output = f"{_paper_model_manifest_markdown(study, output_filters, output_entries)}\n"
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
    entries: Sequence[PaperModelManifestEntry],
    family_counts: Counter[str],
) -> dict[str, object]:
    model_ids_by_family: dict[str, list[str]] = {}
    for entry in entries:
        model_ids_by_family.setdefault(entry.family, []).append(entry.model_id)

    return {
        "manifest_path": str(manifest_path),
        "study": asdict(study),
        "output_filters": output_filters,
        "model_count": len(entries),
        "family_counts": dict(sorted(family_counts.items())),
        "model_ids_by_family": dict(sorted(model_ids_by_family.items())),
        "models": [asdict(entry) for entry in entries],
    }


def _paper_model_manifest_markdown(
    study: PaperStudyManifest,
    output_filters: dict[str, list[str]],
    entries: Sequence[PaperModelManifestEntry],
) -> str:
    return "\n\n".join(
        [
            _paper_study_manifest_markdown(study),
            _paper_model_output_filters_markdown(output_filters),
            _paper_model_entries_markdown(entries),
        ]
    )


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


def _paper_model_entries_markdown(entries: Sequence[PaperModelManifestEntry]) -> str:
    lines = [
        "## Models",
        "",
        "| Family | Model ID | Model type | Prediction mode | Source config |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        prediction_mode = entry.prediction_mode or "-"
        lines.append(
            "| "
            f"{entry.family} | "
            f"{entry.model_id} | "
            f"{entry.source_model_type} | "
            f"{prediction_mode} | "
            f"{entry.source_config_path} |"
        )
    return "\n".join(lines)
