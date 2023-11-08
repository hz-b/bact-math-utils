"""Simple stat utils not provided by numpy
"""
import numpy as np


def mean_square_error(dv: np.ndarray, **kwargs) -> np.ndarray:
    """means square error from difference

    Args:
        dv:   difference (user needs to calculate A - B)
        axis: to sum over. see :func:`numpy.mean` documentation
              for its usage

    Returns:
        mean square error
    """
    dv = np.asarray(dv)
    mse = np.mean(dv ** 2, **kwargs)
    return mse


def mean_absolute_error(dv: np.ndarray, **kwargs) -> np.ndarray:
    """means absolute error from difference

    Args:
        dv:   difference (user needs to calculate A - B)
        axis: to sum over. see :func:`numpy.mean` documentation
              for its usage

    Returns:
        mean absolute error
    """
    dv = np.asarray(dv)
    mae = np.mean(np.absolute(dv), **kwargs)
    return mae
