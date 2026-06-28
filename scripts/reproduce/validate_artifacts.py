"""Validate a local MotionAge artifact manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from motionage.artifacts import load_artifact_manifest, validate_artifact_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a manifest and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to an artifact manifest YAML file.",
    )
    parser.add_argument(
        "--fail-on-optional-missing",
        action="store_true",
        help="Return a non-zero exit code when optional artifacts are missing.",
    )
    args = parser.parse_args(argv)

    manifest = load_artifact_manifest(args.manifest)
    report = validate_artifact_manifest(manifest)
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report["ok"]:
        return 1
    if args.fail_on_optional_missing and report["missing_optional"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
