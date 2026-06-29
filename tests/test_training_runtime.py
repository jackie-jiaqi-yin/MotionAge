from __future__ import annotations

from unittest.mock import Mock

import pytest

from motionage.training.runtime import configure_torch_cpu_threads


def test_configure_torch_cpu_threads_skips_redundant_resets(monkeypatch) -> None:
    set_num_threads = Mock()
    set_num_interop_threads = Mock()

    monkeypatch.setattr("motionage.training.runtime.torch.get_num_threads", lambda: 4)
    monkeypatch.setattr("motionage.training.runtime.torch.get_num_interop_threads", lambda: 1)
    monkeypatch.setattr("motionage.training.runtime.torch.set_num_threads", set_num_threads)
    monkeypatch.setattr(
        "motionage.training.runtime.torch.set_num_interop_threads",
        set_num_interop_threads,
    )

    configure_torch_cpu_threads(
        {
            "training": {
                "torch_num_threads": 4,
                "torch_num_interop_threads": 1,
            }
        }
    )

    set_num_threads.assert_not_called()
    set_num_interop_threads.assert_not_called()


def test_configure_torch_cpu_threads_updates_changed_values(monkeypatch) -> None:
    set_num_threads = Mock()
    set_num_interop_threads = Mock()

    monkeypatch.setattr("motionage.training.runtime.torch.get_num_threads", lambda: 2)
    monkeypatch.setattr("motionage.training.runtime.torch.get_num_interop_threads", lambda: 1)
    monkeypatch.setattr("motionage.training.runtime.torch.set_num_threads", set_num_threads)
    monkeypatch.setattr(
        "motionage.training.runtime.torch.set_num_interop_threads",
        set_num_interop_threads,
    )

    configure_torch_cpu_threads(
        {
            "training": {
                "torch_num_threads": 4,
                "torch_num_interop_threads": 3,
            }
        }
    )

    set_num_threads.assert_called_once_with(4)
    set_num_interop_threads.assert_called_once_with(3)


def test_configure_torch_cpu_threads_can_warn_and_continue_on_interop_runtime_error(
    monkeypatch,
) -> None:
    logger = Mock()
    set_num_interop_threads = Mock(side_effect=RuntimeError("already initialized"))

    monkeypatch.setattr("motionage.training.runtime.torch.get_num_threads", lambda: 2)
    monkeypatch.setattr("motionage.training.runtime.torch.get_num_interop_threads", lambda: 1)
    monkeypatch.setattr("motionage.training.runtime.torch.set_num_interop_threads", set_num_interop_threads)

    configure_torch_cpu_threads(
        {"training": {"torch_num_interop_threads": 3}},
        logger=logger,
        allow_interop_runtime_error=True,
    )

    set_num_interop_threads.assert_called_once_with(3)
    logger.warning.assert_called_once()


def test_configure_torch_cpu_threads_raises_interop_runtime_error_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr("motionage.training.runtime.torch.get_num_threads", lambda: 2)
    monkeypatch.setattr("motionage.training.runtime.torch.get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(
        "motionage.training.runtime.torch.set_num_interop_threads",
        Mock(side_effect=RuntimeError("already initialized")),
    )

    with pytest.raises(RuntimeError, match="already initialized"):
        configure_torch_cpu_threads({"training": {"torch_num_interop_threads": 3}})
