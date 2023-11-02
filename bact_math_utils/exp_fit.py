"""Fit an exponential to an decay

with a twist towards real world measurement data
"""
import numpy as np
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger("bact_math_utils")


def estimate_tau_inv(indep, dep):
    """Estimate amplitude and tau_inv for an exponential function

    Args:
        indep : the independent (typically t)
        dep : the dependent variable  (typically Tau)

    Returns:
        c, tau_inv
    Fits a linear regression  to the logarithmic of the dependent

    Shallow wrapper of :func:`numpy.polyfit`
    """

    lndep = np.log(dep)
    pars = np.polyfit(indep, lndep, 1)
    ln_c = pars[-1]
    tau_inv = pars[-2]
    c = np.exp(ln_c)
    return c, tau_inv


def scaled_exp(t, c, tau, b=None, t0=None):
    r"""Calculate a scaled exponential

    Well, used for the fit functions ...

    .. math::
        y(t) = c * \exp{(t/tau)}
    """
    t = np.asarray(t)
    c = np.asarray(c)
    tau = np.asarray(tau)

    dt = t
    if t0 is not None:
        logger.debug("scale_exp: using t0 of %s", t0)
        dt = t - t0

    ts = dt / tau
    scale = np.exp(ts)
    y = c * scale
    if b is not None:
        logger.debug("scaled_exp: using b of %s", b)
        y = y + b
    return y


def scaled_exp_df(t, c0, tau, b=None, t0=None):
    r"""
    Derivatives:

    .. math::
        \frac{df}{d\tau}  = - \frac{c_{0} t}{\tau^{2}} e^{- \frac{t}{\tau}}
        \frac{df}{c_0}    =  e^{- \frac{t}{\tau}}
        \frac{df}{b}      =  1
        \frac{df}{d\tau_{inv}} = - \tau_{inv} c_{0} e^{\tau_{inv}
                                   \left(t - t_{0}\right)

    """
    logger.debug("scaled_exp_df: extra parameters b=%s t0=%s", b, t0)
    tau_inv = 1.0 / tau

    dt = t
    if t0 is not None:
        dt = t - t0

    dts = dt * tau_inv

    e_term = np.exp(dts)

    df_c0 = e_term
    c0_eterm = c0 * e_term

    df_dtau = c0_eterm * dt

    # Should it not be that
    df_dtau = -df_dtau * tau_inv ** 2

    dfs = [df_c0, df_dtau]

    if b is not None:
        df_db = np.ones(df_c0.shape, dtype=np.float)
        dfs.append(df_db)

    if t0 is not None:
        df_t0 = -tau_inv * c0_eterm
        dfs.append(df_t0)

    df = np.array(dfs)
    df = df.transpose()
    return df


def fit_scaled_exp(t, y, p0=None):
    r"""Fit a decay function to the measured current

    Args:
        t : independent variable (typically time)
        y : dependent variable
        p0 (optional, default=None) : start parameters

    Fits the function :func:`scaled_exp` to the data t, y

    .. math::

        y = c_0  e^{- \frac{t}{\tau}}

    using its derivatives.

    If p0 is not given it uses :func:`estimate_tau_inv` to estimate p0

    uses:
        * :func:`scipy.optimize.curve_fit` for fit routine
        * :func:`numpy.polfit` to estimate p0 using a linear fit
    """

    t = np.asarray(t)
    y = np.asarray(y)

    if p0 is None:
        c0, tau_inv = estimate_tau_inv(t, y)
        tau0 = 1.0 / tau_inv
    else:
        c0, tau0 = p0

    c, cov = curve_fit(scaled_exp, t, y, p0=(c0, tau0), jac=scaled_exp_df)
    tmp = np.diag(cov)
    err = np.sqrt(tmp)
    return c, cov, err


def fit_scaled_exp_offset(t, y, *, p0=None, b=None, t0=None):
    r"""Fit a decay function to the measured current

    This routine corrects for an offset (e.g. a calibration factor).
    Furthermore it substracts t0 from t if t0 is given.

    Args:
        t : independent variable (typically time)
        y : dependent variable
        b : offset
        p0 (optional, default=None) : start parameters c0, tau, b. This can
                                      contain the value b as last parameter.
                                      Will complain if the parameter b is
                                      defined too
        t0 : start point of decay

    Fits the function :func:`scaled_exp` to the data t, y

    .. math::

        y = c_0  e^{- \frac{t - t_0}{\tau}} + b

    using its derivatives. See documentation of :func:`scaled_exp_df` for its
    derivatives

    If p0 is not given it uses :func:`estimate_tau_inv` to estimate p0.

    Warning:
        If using an offset, the estimate will be less robust. In particular the
        rather provide a good guess for the starting parameters. The estimate
        based on the linear model tends to faill.

    uses:
        * :func:`scipy.optimize.curve_fit` for fit routine
        * :func:`numpy.polfit` to estimate p0 using a linear fit
    """

    t = np.asarray(t)
    y = np.asarray(y)

    # This routine tries dealing with data that should be corrected first

    # Correct data. If negative assume that an offset should be
    # contained in the data
    # The offset correction will then remove it afterwards
    yt = y
    _eps = 1e-8

    # Handle start parameters
    if p0:
        c0, tau0, *remaining = p0[:2]
        # If there are extra parameters these are expected to be
        # just describing a parameter b
        b = 0
        if len(remaining):
            (tmp,) = remaining
            if b is not None:
                txt = "parameter b specified in p0 and as argument"
                raise AssertionError(txt)
            b = tmp
    else:
        if b is None:
            b = 0
            y_min = y.min()
            if y_min < _eps:
                b = -y_min + _eps
                yt = y + b
        c0, tau_inv = estimate_tau_inv(t, yt)
        tau0 = 1.0 / tau_inv

    txt = "fit_scaled_exp_offset: starting with parameters"
    txt += " c0 = %s, tau0 = %s, b = %s"
    logger.debug(txt, c0, tau0, b)

    c, cov = curve_fit(scaled_exp, t, y, p0=(c0, tau0, b), jac=scaled_exp_df)
    tmp = np.diag(cov)
    err = np.sqrt(tmp)
    return c, cov, err


__all__ = ["fit_scaled_exp", "fit_scaled_exp_offset", "estimate_tau_inv", "scaled_exp"]
