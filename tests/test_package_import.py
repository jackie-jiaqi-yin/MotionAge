from __future__ import annotations

import motionage


def test_package_imports() -> None:
    assert isinstance(motionage.__version__, str)
    assert motionage.__version__
