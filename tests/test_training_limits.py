from __future__ import annotations

import motionage.training as training


def test_optional_positive_int_parses_positive_integer_values() -> None:
    assert hasattr(training, "optional_positive_int")

    assert training.optional_positive_int(1) == 1
    assert training.optional_positive_int("5") == 5
    assert training.optional_positive_int(3.0) == 3


def test_optional_positive_int_treats_missing_invalid_or_nonpositive_values_as_unlimited() -> None:
    assert hasattr(training, "optional_positive_int")

    for value in (None, "", "not-an-int", 0, "0", -1, "-3"):
        assert training.optional_positive_int(value) is None
