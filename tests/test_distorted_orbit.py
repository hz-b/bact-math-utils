import unittest
import numpy as np
from bact_math_utils import distorted_orbit

class TestOrbit(unittest.TestCase):
    """ """

    def setUp(self):
        N = 5
        self.beta = np.linspace(1, 2, num=N)
        self.mu = np.linspace(0, (np.pi / 2), num=N)
        self.eps = 1e-7

    def test10_UnitUnscaledKick(self):
        """negative side so to say ...."""
        tune = 1

        f = distorted_orbit.closed_orbit_kick_unscaled
        res = f(self.mu, tune=tune, mu_i=0)
        test = np.cos(tune * np.pi - self.mu)
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test11_UnitUnscaledKick(self):
        """positive side so to say ...."""
        tune = 1
        mu_i = tune * np.pi

        f = distorted_orbit.closed_orbit_kick_unscaled
        res = f(self.mu, tune=tune, mu_i=mu_i)
        test = np.cos(-self.mu)
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test20_UnitScaledKick(self):
        tune = 1

        f = distorted_orbit.closed_orbit_kick
        res = f(self.mu, tune=tune, mu_i=0, theta_i=1, beta_i=1)
        test = np.cos(tune * np.pi - self.mu)
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test20_UnitScaledKick2(self):
        tune = 1

        f = distorted_orbit.closed_orbit_kick
        res = f(self.mu, tune=tune, mu_i=0, theta_i=2, beta_i=4)
        test = np.cos(tune * np.pi - self.mu)
        # One two for the angle and one for the squared beta_i
        test *= 2 * 2
        self.assertTrue((np.absolute(res - test) < self.eps).all())

    def test30_noDistortion(self):
        res = distorted_orbit.closed_orbit_distortion(
            self.beta, self.mu, tune=self.mu.max() * 2, beta_i=0, theta_i=0, mu_i=0
        )
        self.assertTrue((np.absolute(res) < self.eps).all())

    def test31_UnitDistortionZero(self):
        tune = 1

        res = distorted_orbit.closed_orbit_distortion(
            self.beta, self.mu, tune=tune, beta_i=0, theta_i=1, mu_i=0
        )
        # Still 0: beta is 0
        self.assertTrue((np.absolute(res) < self.eps).all())

    def test32_UnitDistortion(self):
        tune = 1 + 1 / 2

        test = np.cos(self.mu)

        beta = np.ones(self.beta.shape)
        res = distorted_orbit.closed_orbit_distortion(
            beta, self.mu, tune=tune, beta_i=1, theta_i=1, mu_i=tune * np.pi
        )
        self.assertTrue((np.absolute(res + test / 2) < self.eps).all())

        beta_sq = np.arange(1, 6)
        beta = beta_sq ** 2
        res = distorted_orbit.closed_orbit_distortion(
            beta, self.mu, tune=tune, beta_i=1, theta_i=1, mu_i=tune * np.pi
        )
        self.assertTrue((np.absolute(res + beta_sq * test / 2) < self.eps).all())


if __name__ == "__main__":
    unittest.main()
