"""Torch device resolution helpers."""

from __future__ import annotations

import logging
import platform
from typing import Any

import torch

logger = logging.getLogger(__name__)


def resolve_device(
    config: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> torch.device:
    """Resolve a torch device from config values such as ``auto``, ``cpu``, ``cuda``, or ``mps``."""
    active_logger = logger or globals()["logger"]
    device_str = str(config.get("device", "auto")).lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.backends.mps.is_built():
            active_logger.warning(
                "MPS backend is built but unavailable; falling back to CPU. %s",
                mps_diagnostics(),
            )
        return torch.device("cpu")

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available.")
        return torch.device("cuda")

    if device_str == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("Requested device=mps but this PyTorch build was not compiled with MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Requested device=mps but MPS is unavailable. "
                f"{mps_diagnostics()} "
                "Try upgrading torch and re-syncing the environment."
            )
        return torch.device("mps")

    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device(device_str)


def mps_diagnostics() -> str:
    """Return a compact summary of local MPS runtime state."""
    mac_ver = platform.mac_ver()[0] or "unknown"
    return (
        f"macOS={mac_ver}, torch={torch.__version__}, "
        f"mps_built={torch.backends.mps.is_built()}, "
        f"mps_available={torch.backends.mps.is_available()}"
    )
