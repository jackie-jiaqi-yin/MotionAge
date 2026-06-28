from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

import motionage.training as training


def set_accelerator_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool = False,
    mps_available: bool = False,
    mps_built: bool = False,
) -> None:
    monkeypatch.setattr("motionage.training.device.torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("motionage.training.device.torch.backends.mps.is_available", lambda: mps_available)
    monkeypatch.setattr("motionage.training.device.torch.backends.mps.is_built", lambda: mps_built)


def test_resolve_device_prefers_cuda_then_mps_then_cpu_for_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(training, "resolve_device")

    set_accelerator_state(monkeypatch, cuda_available=True, mps_available=True, mps_built=True)
    assert training.resolve_device({"device": "auto"}) == torch.device("cuda")

    set_accelerator_state(monkeypatch, cuda_available=False, mps_available=True, mps_built=True)
    assert training.resolve_device({"device": "auto"}) == torch.device("mps")

    set_accelerator_state(monkeypatch, cuda_available=False, mps_available=False, mps_built=False)
    assert training.resolve_device({"device": "auto"}) == torch.device("cpu")


def test_resolve_device_warns_when_mps_is_built_but_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(training, "resolve_device")
    logger = Mock()
    set_accelerator_state(monkeypatch, cuda_available=False, mps_available=False, mps_built=True)
    monkeypatch.setattr("motionage.training.device.mps_diagnostics", lambda: "mps diagnostic details")

    assert training.resolve_device({"device": "auto"}, logger=logger) == torch.device("cpu")

    logger.warning.assert_called_once()
    assert "mps diagnostic details" in logger.warning.call_args.args


def test_resolve_device_accepts_explicit_cpu_and_custom_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(training, "resolve_device")
    set_accelerator_state(monkeypatch)

    assert training.resolve_device({"device": "cpu"}) == torch.device("cpu")
    assert training.resolve_device({"device": "meta"}) == torch.device("meta")


def test_resolve_device_rejects_unavailable_cuda_or_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(training, "resolve_device")

    set_accelerator_state(monkeypatch, cuda_available=False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        training.resolve_device({"device": "cuda"})

    set_accelerator_state(monkeypatch, mps_built=False, mps_available=False)
    with pytest.raises(RuntimeError, match="not compiled with MPS"):
        training.resolve_device({"device": "mps"})

    set_accelerator_state(monkeypatch, mps_built=True, mps_available=False)
    monkeypatch.setattr("motionage.training.device.mps_diagnostics", lambda: "mps diagnostic details")
    with pytest.raises(RuntimeError, match="mps diagnostic details"):
        training.resolve_device({"device": "mps"})


def test_mps_diagnostics_reports_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(training, "mps_diagnostics")
    set_accelerator_state(monkeypatch, mps_built=True, mps_available=False)
    monkeypatch.setattr("motionage.training.device.platform.mac_ver", lambda: ("14.0", ("", "", ""), ""))
    monkeypatch.setattr("motionage.training.device.torch.__version__", "2.9.0")

    diagnostics = training.mps_diagnostics()

    assert "macOS=14.0" in diagnostics
    assert "torch=2.9.0" in diagnostics
    assert "mps_built=True" in diagnostics
    assert "mps_available=False" in diagnostics
