'''Helper functions for linear regression

Regression function is not given as typically one would use
:func:`scipy.linalg.lstsq`.

Similar functionality is provided by
:any:`numpy.polynomial.Polynomial`
'''
from scipy import linalg
import numpy as np

__all__ = ['x_to_cov', 'cov_to_std', 'linear_fit_1d']


def x_to_cov(X: np.ndarray, residuals: float, N: int, p: int) -> np.ndarray:
    '''compute the covariance for a linear regression

    The linear regression could be computed e.g. by :func:`scipy.linalg.lstsq`
    Mainly given here, so that it does not need to be repeated.

    Args:
        X: independent variables
        N: number of data points
        p: number of parameters (including the bias or constant part)

    See e.g. [ElemStatLearn] chapter 3.2
    '''

    # Literature would now take one value off for the bias
    # here contained within p
    devisor = N - p
    assert(devisor > 0)

    A = X.T.dot(X)
    A = linalg.inv(A)
    cov = A / devisor * residuals
    return cov


def cov_to_std(cov: np.ndarray) -> np.ndarray:
    '''Compute standard deviation from co variance matrix
    '''
    diag = cov.diagonal()
    std = np.sqrt(diag)

    return std


def linear_fit_1d(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    '''fit a line and estimate the accuracy of the parameters

    Args:
        x: independent
        y: dependent

    Fits slope and intercept to the data

    Returns: (p, dp)


    '''
    N = len(x)
    X = np.ones((N, 2), np.float_)
    X[:, 0] = x

    p, residues, rank, s = linalg.lstsq(X, y)

    cov = x_to_cov(X, residues, N, 2)
    std = cov_to_std(cov)

    return p, std
