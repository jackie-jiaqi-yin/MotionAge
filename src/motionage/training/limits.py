"""Training-loop limit parsing helpers."""

from __future__ import annotations

from typing import Any


def optional_positive_int(value: Any) -> int | None:
    """Return a positive integer limit, or ``None`` when the value means unlimited."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
