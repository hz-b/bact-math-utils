from bact_math_utils import stats
import numpy as np
import pytest


def test10_zeros():
    A = np.zeros(10)
    mse = stats.mean_square_error(A)
    assert mse == pytest.approx(0, abs=1e-12)

    mae = stats.mean_absolute_error(A)
    assert mae == pytest.approx(0, abs=1e-12)


def test20_single_value():
    """one value should be returned as one value"""

    for cnt, val in enumerate([1, -1, 0, 2, -2, 355 / 113, -355 / 113]):
        N = 10 + 20 * cnt
        A = np.zeros(N)
        A[cnt] = val

        mse = stats.mean_square_error(A)
        assert mse == pytest.approx(val ** 2 / N, abs=1e-12)

        mae = stats.mean_absolute_error(A)
        assert mae == pytest.approx(np.absolute(val) / N, abs=1e-12)


def test30_cross_check_range():
    """values compared to prediction of Faulhaber's formula"""
    n = 23
    A = np.arange(1, n + 1)

    mae = stats.mean_absolute_error(A)
    # (1 / 2 * n * (n+1)) / n
    assert mae == pytest.approx((n + 1) / 2, abs=1e-12)

    mse = stats.mean_square_error(A)
    # 1 / 6 * (n * (n+1) * (2n + 1)) * 1 / N
    assert mse == pytest.approx((n + 1) * (2 * n + 1) / 6, abs=1e-12)
