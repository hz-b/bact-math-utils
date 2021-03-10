'''Fit an exponential to an decay


'''
import numpy as np
from scipy.optimize import curve_fit


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


def scaled_exp(t, c, tau):
    """Calculate a scaled exponential

    Well, used for the fit functions ...

    .. math::
        y(t) = c * \exp{(t/tau)}
    """
    t = np.asarray(t)
    c = np.asarray(c)
    tau = np.asarray(tau)

    ts = t / tau
    scale = np.exp(ts)
    y = c * scale
    return y


def scaled_exp_df(t, c0, tau):
    """
    Derivatives:

    .. math::
        \frac{df}{d\\tau} = '\\frac{c_{0} t}{\\tau^{2}} e^{- \\frac{t}{\\tau}}'
        \frac{df}{c_0} =  'e^{- \\frac{t}{\\tau}}'

    """
    tau_inv = 1./tau

    ts = t * tau_inv

    e_term = np.exp(ts)
    df_c0 = e_term
    df_dtau = c0 * t * e_term

    df = np.array((df_c0, df_dtau))
    df = df.transpose()
    return df


def fit_scaled_exp(t, y, p0=None):
    """Fit a decay function to the measured current

    Args:
        t : independent variable (typically time)
        y : dependent variable
        p0 (optional, default=None) : start parameters

    Fits the function :func:`scaled_exp` to the data t, y

    .. math::

        y = c_0  e^{- \\frac{t}{\\tau}}

    using its derivatives.

    .. math::
        \\frac{df}{d\\tau} =  \\frac{c_{0} t}{\\tau^{2}} e^{- \\frac{t}{\\tau}}
        \\frac{df}{c_0} =  e^{- \\frac{t}{\\tau}}

    If p0 is not given it uses :func:`estimate_tau_inv` to estimate p0

    uses:
        * :func:`scipy.optimize.curve_fit` for fit routine
        * :func:`numpy.polfit` to estimate p0 using a linear fit
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if p0 is None:
        c0, tau_inv = estimate_tau_inv(t, y)
        tau0 = 1./tau_inv
    else:
        c0, tau0 = p0

    c, cov = curve_fit(scaled_exp, t, y,  p0=(c0, tau0),
                       jac=scaled_exp_df)
    tmp = np.diag(cov)
    err = np.sqrt(tmp)
    return c, cov, err


def scaled_exp_df_offset(t, c0, tau, t0=None, b=None):
    """
    Derivatives:

    .. math::
        \frac{df}{d\\tau} = \\frac{c_{0} t}{\\tau^{2}} e^{- \\frac{t}{\\tau}}
        \frac{df}{c_0}    =  e^{- \\frac{t}{\\tau}}
        \frac{df}{b}      =  1

    """
    tau_inv = 1./tau

    dt = t
    if t0 is not None:
        dt = t - t0

    dts = dt * tau_inv

    e_term = np.exp(dts)

    df_c0 = e_term
    c0_eterm = c0 * e_term
    df_dtau = c0_eterm * dt

    dfs = [df_c0, df_dtau]

    if t0 is not None:
        df_t0 = - tau_inv * c0_eterm
        dfs.append(df_t0)

    if b is not None:
        df_db = np.ones(df_c0.shape, dtype=np.float)
        dfs.append(df_db)

    df = np.array(dfs)
    df = df.transpose()
    return df


def fit_scaled_exp_offset(t, y, p0=None):
    """Fit a decay function to the measured current

    This routine corrects for an offset (e.g. a calibration factor).
    Furthermore it
    """
    t = np.asarray(t)
    y = np.asarray(y)

    # Assume that the data start in the middle of this series
    t0 = t.mean()
    dt = t - t0
    yt = y
    b = 0

    # This routine
    # Correct data. If negative assume that an offset should be
    # contained in the data
    # The offset correction will then remove it afterwards
    y_min = y.min()

    _eps = 1e-8
    if y_min < _eps:
        b = - y_min + _eps
        yt = y + b

    if p0 is None:
        c0, tau_inv = estimate_tau_inv(dt, yt)
        tau0 = 1./tau_inv
    else:
        c0, tau0 = p0

    c, cov = curve_fit(scaled_exp, t, y,  p0=(c0, tau0, t0, b),
                       jac=scaled_exp_df_offset)
    tmp = np.diag(cov)
    err = np.sqrt(tmp)
    return c, cov, err


__all__ = ['fit_scaled_exp', 'fit_scaled_exp_offset']
