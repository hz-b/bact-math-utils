'''

@author: Pierre Schnizer

Calculate Touschek and Gas life time applying noise on the beam
'''
from .exp_fit import scaled_exp
from scipy.optimize import curve_fit
import numpy as np


def lifetime_function_orig(u, a, b, c):
    """Lifetime function code directly implemented

    Warning:
        Do not use it! Use :func:`lifetime_function` instead
    """
    return 1/(1/np.sqrt(b**2 + c**2*u**2) + 1/a)


def _calculate_touschek2(u, b, c):
    r"""Calculate the *square* touschek lifetime

    .. math::
        \tau_{Touschek}^2 = b^2 + (c u)^2
    """
    b2 = b*b
    us = c * u
    tau_touschek2 = b2 + us**2
    return tau_touschek2


def calculate_touschek(*args):
    """Calculate the touschek lifetime

    Just applies :func:`np.sqrt` to :func:`_calculate_touschek2`
    """
    tau_touschek2 = _calculate_touschek2(*args)
    result = np.sqrt(tau_touschek2)
    return result


def _liftime_function_physics(tauG, tau_touschek):
    r"""Calculate lifetime from gas and touschek life time

    .. math::

        \frac{1}{\tau} = \frac{1}{\tau_{gas}} + \frac{1}{\tau_{Touschek}}

    reformulated to use scaled 1 / (1 + q)
    """
    # And even better
    tauGs = tauG / tau_touschek

    devisor = 1 + tauGs
    result = tauG / devisor
    return result


def lifetime_function(u, tauG, b, c):
    r"""Calculate the life time depending on

    Args:
        u : the noise level with which the beam was angeregt
        tauG : gas life
        b : constant contribution to Touschek lifetime
        c : linear scale factor for contribution to Touschek lifetime

    Returns:
        lifetime (float)

    Evaluates the equation

    .. math::
        \frac{1}{\tau} = \frac{1}{\tau_{gas}} + \frac{1}{\tau_{Touschek}}

    with

    .. math::

            \tau_{T} = \sqrt{b^2 + c u^2}

    Its derivatives are required for the fit function. These were calculated to

    .. math::

        \frac{df}{d \tau_{gas} } = \frac{f^2}{ \tau^2_{gas} }
        \quad
        \frac{df}{db} = b \frac{f^2}{\tau^3_{T}}
        \quad
        \frac{df}{dc} = c \frac{f^2}{\tau^3_{T}}

    with :math:`\tau_{T}` the Touschek life time

    The derivatives are implemented in :func:`calculate_lifetime_parameters_df`

    TODO:
        * Should this code go into ? bact/bact/core/eV/life_time.py
        * Please give a reference for the equations

    """
    # Personally I prefer to handle each term in a single line, in particular
    # the ones that can fail. Tracecback will then put our nose to it
    tau_touschek2 = _calculate_touschek2(u, b, c)
    tau_touschek = np.sqrt(tau_touschek2)
    result = _liftime_function_physics(tauG, tau_touschek)
    return result


def lifetime_function_fdf(u, tauG, b, c):
    """Derivatives of the lifetime function


    For details see :func:`calculate_lifetime_parameters` and
    :func:`lifetime_function`
    """
    tauG2 = tauG**2

    tau_touschek2 = _calculate_touschek2(u, b, c)
    tau_touschek = np.sqrt(tau_touschek2)
    tau = _liftime_function_physics(tauG, tau_touschek)
    tau2 = tau**2

    df_dtauG = tau2 / tauG2

    devisor = tau_touschek2 * tau_touschek
    df_tmp = tau2 / devisor

    df_db = df_tmp * b
    df_dc = df_tmp * c * u**2

    result = np.array((df_dtauG, df_db, df_dc), order="F")
    result = np.transpose(result)
    return tau, result


def lifetime_function_df(u, tauG, b, c):
    """Derivatives of the lifrtime function


    For details see :func:`calculate_lifetime_parameters` and
    :func:`lifetime_function`
    """
    f, dtau = lifetime_function_fdf(u, tauG, b, c)
    return dtau


def calculate_lifetime_parameters(noise_level, life_time,
                                  start_parameters=None):
    """Calculate the Touschek and Gas lifetime from noise measurement and t
    life time  measurements

    Uses :func:`scipy.optimize.curve_fit` to fit the
    function :func:`lifetime_function` to the set noise level and measured
    life time

    Args:
        noise_level : the noise level that was used on the beam (unit? )
        life_time   : the measured life time (i.e. the current decay?)
        start_parameters (optional, None) : the start parameters to use in the
                                            function

    Uses
        * :func:`scipy.optimize.curve_fit` for fit rotine
        * :func: `calculate_lifetime_parameters_df` for derivatives
    """
    if start_parameters is None:
        start_parameters = [8.0, 3.0, 5.0]
    start_parameters = np.asarray(start_parameters)

    npar = len(start_parameters)
    assert(npar == 3)

    c, cov = curve_fit(lifetime_function, noise_level, life_time,
                       p0=start_parameters, method="lm",
                       jac=lifetime_function_df)

    tmp = np.diag(cov)
    err = np.sqrt(tmp)
    return c, cov, err


def beam_decay(t, u, c0, tauG, b, c, offset):
    """Calculate beam decay

    Args:
        t      : time
        u      : noise excitation
        c0     : beam current at time t=0
        u      : the noise level with which the beam was excited
        tauG   : gas life
        b      : constant contribution to Touschek lifetime
        c      : linear scale factor for contribution to Touschek lifetime
        offset : offset of the current reading of the beam current monitor

    Returns:
        beam current
    """
    tau = lifetime_function(u, tauG, b, c)
    y = scaled_exp(t, c0, -tau) + offset
    return y


def beam_decay_fsdf(t, u, c0, tauG, b, c, offset):
    """Calculate beam decay current at its derivatives

    Args:
        See argument description of :func:`beam_decay_df`

    Returns:
           (fs , df)

    fs has to be multiplied with c0 to obtain f
    """
    tau2, dtau = lifetime_function_fdf(u, tauG, b, c)
    tau = np.sqrt(tau2)
    ys = scaled_exp(t, 1, tau)

    nu = len(u)
    df = np.zeros((nu, 5), np.float_)

    # Derviatives with respect to calculate_lifetime
    df_scale = -c0 * t / tau2 * ys
    tmp = dtau * df_scale[:, np.newaxis]
    df[:, 0] = ys
    df[:, 1:4] = tmp
    df[:, 4] = 1

    return ys, df


def beam_decay_fdf(t, u, c0, tauG, b, c, offset):
    fs, df = beam_decay_fsdf(t, u, c0, tauG, b, c, offset)
    f = fs * c
    return f, df


def beam_decay_df(t, u, c0, tauG, b, c, offset):
    """Calculate beam decay derivatives

    Args:
        See argument description of :func:`beam_decay`
    """
    fs, df = beam_decay_fsdf(t, u, c0, tauG, b, c, offset)
    return df
