"""Command-line helpers for public MotionAge checks."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from motionage.paper_manifest import (
    REQUIRED_PAPER_MODEL_FAMILIES,
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

    family_counts = Counter(entry.family for entry in entries)
    print(f"Validated {len(entries)} paper model configs from {args.manifest_path}.")
    for family in sorted(family_counts):
        print(f"{family}: {family_counts[family]}")
    return 0
