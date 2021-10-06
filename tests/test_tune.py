from bact_math_utils import tune
import numpy as np
import unittest


class testTune(unittest.TestCase):
    def test00_dQ(self):
        """zero element"""
        dQ = tune.working_point_change(0, 0, length=0)
        self.assertAlmostEqual(dQ, 0)

    def test01_dQ(self):
        """Neutral element"""
        dQ = tune.working_point_change(1, 1, length=2)
        self.assertAlmostEqual(dQ, 1 / (2 * np.pi))

    def test02_dQ(self):
        """dk * length close to pi"""
        dQ = tune.working_point_change(1 / 113, 4, length=355)
        self.assertAlmostEqual(dQ, 1, places=6)

    def test03_dQ(self):
        """beta close to pi, k * l close to 4"""
        dQ = tune.working_point_change(1.0 / 2.0, 355.0 / 113.0, length=8)
        self.assertAlmostEqual(dQ, 1, places=6)

    def test04_dQ(self):
        """BESSY II: muxer applied to Q1M1D1R"""
        dQ = tune.working_point_change(0.01 * 2, 10, length=0.25)
        self.assertAlmostEqual(dQ, 0.1 /(8 * np.pi), places=6)

    def test10_dT(self):
        """Close to BESSY 2 real usage: Q1M1D1R x
        """
        dT = tune.tune_change(0.01 * 2, beta=15, length=0.25, f=500e6, nb=400)
        self.assertAlmostEqual(dT, 3/2 * 2/ 4 * 0.1 /(4 * np.pi) * 500e6/400)

    def test11_dT(self):
        """Close to BESSY 2 real usage: Q1M1D1R y
        """
        dT = tune.tune_change(0.01 * 2, beta=10, length=0.25, f=500e6, nb=400)
        self.assertAlmostEqual(dT, 2/4 * 0.1 /(4 * np.pi) * 500e6/400)


if __name__ == "__main__":
    unitest.main()
