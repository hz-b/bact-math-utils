import unittest
import numpy as np
from bact_math_utils.exp_fit import estimate_tau_inv, fit_scaled_exp, scaled_exp


class Test0_ScaledExponent(unittest.TestCase):
    def test0(self):
        """0 parameters, c0, tau = 1
        """
        t = np.linspace(-10, 10)
        y = scaled_exp(t, 0, 1)
        flags = y == 0
        self.assertEqual(flags.all(), True)

    def test1(self):
        """ tau = 1, c = 1
        """
        c = 1
        tau = 1

        y = scaled_exp(0, c, tau)
        self.assertAlmostEqual(y, 1)

        y = scaled_exp(1, c, tau)
        self.assertAlmostEqual(y, np.exp(1))

        y = scaled_exp(10, c, tau)
        self.assertAlmostEqual(y, np.exp(10))

    def test2(self):
        """ tau = 1, c = 10
        """
        c = 10
        tau = 1
        y = scaled_exp(0, c, tau)
        self.assertAlmostEqual(y, 10)

        y = scaled_exp(1, c, tau)
        self.assertAlmostEqual(y, 10 * np.exp(1))

        y = scaled_exp(10, c, tau)
        self.assertAlmostEqual(y, 10 * np.exp(10))

    def test3(self):
        """ tau = 10, c = 1
        """
        c = 1
        tau = 1
        y = scaled_exp(0, c, tau)
        self.assertAlmostEqual(y, 1)

        t = 1
        y = scaled_exp(1, c, tau)
        self.assertAlmostEqual(y, np.exp(1/10.))

    def test3(self):
        """ tau = -1, c = 1
        """
        c = 1
        tau = -1
        y = scaled_exp(0, c, tau)
        self.assertAlmostEqual(y, 1)

        y = scaled_exp(1, c, tau)
        self.assertAlmostEqual(y, np.exp(-1))


class Test0_FitScaledExponent(unittest.TestCase):

    def evalFitFuncs(self, c, tau):
        """Calculate values and test if fit works
        """
        t = np.linspace(-10, 10)
        y = scaled_exp(t, c, tau)

        c_fe, tau_inv =  estimate_tau_inv(t, y)
        self.assertAlmostEqual(c_fe, c)

        tau_f = 1. / tau_inv
        self.assertAlmostEqual(tau_f, tau)

        pars, corr, err = fit_scaled_exp(t, y)
        c_f, tau_f = pars
        self.assertAlmostEqual(c_f, c)
        self.assertAlmostEqual(tau_f, tau)

    def test1(self):
        """ tau = 1, c = 1
        """
        c = 1
        tau = 1
        self.evalFitFuncs(c, tau)

    def test2(self):
        """ tau = 1, c = 10
        """
        c = 10
        tau = 1
        self.evalFitFuncs(c, tau)

    def test3(self):
        """ tau = 10, c = 1
        """
        c = 1
        tau = 1
        self.evalFitFuncs(c, tau)

    def test4(self):
        """ tau = -1, c = 1
        """
        c = 1
        tau = -1
        self.evalFitFuncs(c, tau)


if __name__ == '__main__':
    unittest.main()
