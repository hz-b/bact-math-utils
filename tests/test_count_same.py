import pytest
from bact_math_utils.misc import CountSame, enumerate_same_value_pairs


def test_count_same():
    values = [42] * 3 + [23] + [42] * 2
    step = CountSame()(values)
    assert (0, 42) == next(step)
    assert next(step) == (1, 42)
    assert next(step) == (2, 42)

    assert next(step) == (0, 23)

    assert next(step) == (0, 42)
    assert next(step) == (1, 42)


def test_enumerate_same_pairs():
    r = enumerate_same_value_pairs(
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 2, 2, 2]
    )
    assert r == [0, 1, 2] * 3

    r = enumerate_same_value_pairs(
        [42, 42, 355/113, 355/133, 2.78, 2.78],
        ["x", "x", "x", "y", "y", "y"]
    )
    assert r == [0, 1, 0, 0, 0, 1]
