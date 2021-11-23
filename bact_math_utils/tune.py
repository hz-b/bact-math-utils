import numpy as np
from typing import Sequence

_4pi = 1.0 / (4 * np.pi)


def working_point_change(dk: float, beta: float, *, length=1) -> float:
    """Use betatron function to predict tune change

    Args:
        dk:     the quadrupole strength change dk
        beta:   the betatron function. assumed to be constant accross the magnet
        length: magnet length

    .. math::

         \Delta Q = \frac{1}{4 \pi} \int_{s_0}^{s_0 + l} \Delta k \beta(s) d(s)

    """
    # Integral part: rectangular approximation
    idk = dk * length
    integ = idk * beta
    res = integ * _4pi
    return res


def tune_change(
    dk: float, beta: float, *, length: float = 1, f: float, nb: int
) -> float:
    """Use betatron function to predict tune change

    Args:
        dk:     quadrupole strength change (normalised to dipole strength)
        beta:   betatron function at magnet
        length: magnet length
        f:      frequency of the machine
        nb:     number of bunches

    Measured tune change is

    .. math::
         \Delta Q_m = \frac{\Delta T}{f_m \cdot n_b}


    with math:`\Delta T` the measured tune change, math:`f_m` the main RF frequency and  :math:`n_b`
    the number of bunches
    """

    dQ = working_point_change(dk, beta, length=length)
    main_frequency = f / nb
    dT = dQ * main_frequency
    return dT


def tune_frequency_to_fractional_tune(
    freq: Sequence, *, ring_frequency: float
) -> np.ndarray:
    freq = np.asarray(freq)
    return freq / ring_frequency


def to_momentum(freq, *, ref_freq, alpha):
    """Deduces momentum from frequency

    Args:
        alpha: alpha compaction factor
        freq:  current master clock frequency
        ref_freq: reference frequency of the master clock

    """
    df = freq - ref_freq
    tmp = df / (freq * alpha)
    delta = -tmp
    return delta


__all__ = ["tune_change", "tune_frequency_to_fractional_tune", "to_momentum"]
