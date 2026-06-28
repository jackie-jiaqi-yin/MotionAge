"""Runtime helpers for process-level training configuration."""

from __future__ import annotations

import logging
from typing import Any

import torch


def configure_torch_cpu_threads(
    config: dict[str, Any],
    logger: logging.Logger | None = None,
    *,
    allow_interop_runtime_error: bool = False,
) -> None:
    """Apply optional process-level PyTorch CPU thread settings from config."""
    training_config = dict(config.get("training", {}))

    num_threads = training_config.get("torch_num_threads")
    if num_threads not in (None, "", "null"):
        requested_num_threads = int(num_threads)
        if torch.get_num_threads() != requested_num_threads:
            torch.set_num_threads(requested_num_threads)

    interop_threads = training_config.get("torch_num_interop_threads")
    if interop_threads not in (None, "", "null"):
        requested_interop_threads = int(interop_threads)
        if torch.get_num_interop_threads() != requested_interop_threads:
            try:
                torch.set_num_interop_threads(requested_interop_threads)
            except RuntimeError as exc:
                if not allow_interop_runtime_error:
                    raise
                if logger is not None:
                    logger.warning(
                        (
                            "Skipping torch.set_num_interop_threads(%d): %s "
                            "(current interop setting=%d)"
                        ),
                        requested_interop_threads,
                        exc,
                        torch.get_num_interop_threads(),
                    )

    if logger is not None:
        logger.info(
            "PyTorch CPU threads: intraop=%d interop=%d",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
        )
