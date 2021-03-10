from __future__ import division
import unittest
from bact.applib.lifetime import lifetime_calculate
import numpy as np
import copy

lifetime_function = lifetime_calculate.lifetime_function
lifetime_function_orig = lifetime_calculate.lifetime_function_orig

class _TestLifetimeFunction(unittest.TestCase):
    """Test Lifetime basis
    """
    start_parameters = None
    noise_values = None

    
    def setUp(self):
        assert(self.start_parameters is not None)
        assert(self.noise_values is not None)
        
    def test0_gas_dominated_arraydim(self):
        """Fit to set parameters
        """
        tau  = lifetime_function(self.noise_values, *self.start_parameters)
        l = len(self.noise_values)
        self.assertEqual(len(tau.shape), 1)

    def test1_gas_dominated_array_inverse(self):
        """Fit to unsane parameters
        """
        params = copy.copy(self.start_parameters)
        params[0] = 0.2
        tau  = lifetime_function(self.noise_values, *params)
        l = len(self.noise_values)
        self.assertEqual(len(tau.shape), 1)

    def test2_result(self):
        """
        """
        tau  = lifetime_function(self.noise_values, *self.start_parameters)
        l = len(self.noise_values)
        self.assertEqual(len(tau.shape), 1)

    def test4_result(self):
        """Test at independent = 1
        """
        u = 1
        a, b, c = self.start_parameters
        tau  = lifetime_function(u, a, b, c)

        tau_ref = lifetime_function_orig(u, a, b, c)
        self.assertAlmostEqual(tau, tau_ref, 5)
        
    def test5_result(self):
        """Test at independent = 10
        """
        u = 10
        a, b, c = self.start_parameters
        tau  = lifetime_function(u, a, b, c)

        tau_ref = lifetime_function_orig(u, a, b, c)
        self.assertAlmostEqual(tau, tau_ref, 5)

class Test0_InsaneGasDomisnated(_TestLifetimeFunction):
    """Expected parameters
    """
    start_parameters = [1, 1e100, 1e100]
    n_points = 41
    noise_values  = np.linspace(0.0, 10.0, n_points)
    noise_values[0] += 1e-3

    def test3_result(self):
        tau  = lifetime_function(self.noise_values, *self.start_parameters)
        l = len(self.noise_values)

        for t in tau:
            self.assertAlmostEqual(t, self.start_parameters[0], 7)
        self.assertEqual(len(tau.shape), 1)

class Test2_LiftimeFuntionTouschekDominated(_TestLifetimeFunction):
    """Expected parameters
    """
    start_parameters = [.1e-6, 3.0, 5.0]
    n_points = 80
    noise_values  = np.linspace(0.0, 10.0, n_points)
    noise_values[0] += 1e-3

    
class Test3_LiftimeFuntion(_TestLifetimeFunction):
    """Expected parameters
    """
    start_parameters = [8.0, 3.0, 1.0]
    n_points = 41
    noise_values  = np.linspace(0.0, 10.0, n_points)
    noise_values[0] += 1e-3

    def test3_result(self):
        """Test value close to u = 0"""
        u = 1e-3
        tau  = lifetime_function(u, *self.start_parameters)
        self.assertAlmostEqual(tau, 2.18, 2)


del _TestLifetimeFunction

scaled_exp = lifetime_calculate.scaled_exp
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

estimate_tau_inv = lifetime_calculate.estimate_tau_inv
fit_scaled_exp = lifetime_calculate.fit_scaled_exp

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
