import unittest
from bact_math_utils import linear_fit
import numpy as np


def compute_cov(x):
    '''Compute covariance for linear fit with independent x
    '''
    N = len(x)
    X = np.ones((N, 2), np.float_)
    X[:, 0] = x

    residues = 1
    # p includes bias
    p = 2
    cov = linear_fit.x_to_cov(X, residues, N, p)
    return cov


class Test00_Trivia(unittest.TestCase):
    '''But still good to check
    '''
    def test00_cov_to_std(self):
        '''Covariance : null matrix
        '''
        cov = np.zeros((2, 2))
        std = linear_fit.cov_to_std(cov)
        self.assertAlmostEqual(std[0], 0)
        self.assertAlmostEqual(std[1], 0)

    def test01_cov_to_std(self):
        '''Covariance : diagonal 1
        '''
        cov = np.zeros((2, 2))
        idx = np.arange(2)
        cov[idx, idx] = 1

        std = linear_fit.cov_to_std(cov)
        self.assertAlmostEqual(std[0], 1)
        self.assertAlmostEqual(std[1], 1)

    def test02_cov_to_std(self):
        '''Covariance : diagonal 4, 9
        '''
        cov = np.zeros((2, 2))
        cov[0, 0] = 4
        cov[1, 1] = 9

        std = linear_fit.cov_to_std(cov)
        self.assertAlmostEqual(std[0], 2)
        self.assertAlmostEqual(std[1], 3)

    def test03_cov_to_std(self):
        '''Covariance: off diagonal elements set
        '''
        cov = np.zeros((2, 2))
        cov[0, 1] = 4
        cov[1, 0] = 9

        std = linear_fit.cov_to_std(cov)
        self.assertAlmostEqual(std[0], 0)
        self.assertAlmostEqual(std[1], 0)


class _TestLineFit(unittest.TestCase):
    '''Commons for testing line fit
    '''
    x = None
    y = None

    cov00 = None
    cov01 = None
    cov11 = None

    p = None
    p_err = None

    def setUp(self):
        assert(self.x is not None)
        assert(self.y is not None)

        assert(self.cov00 is not None)
        assert(self.cov01 is not None)
        assert(self.cov11 is not None)

        assert(self.p is not None)
        assert(self.p_err is not None)

    def computeCov(self):
        return compute_cov(self.x)

    def checkCov(self, cov00, cov01, cov11, cov):
        self.assertAlmostEqual(cov[0, 0],  cov00)
        self.assertAlmostEqual(cov[0, 1],  cov01)
        self.assertAlmostEqual(cov[1, 0],  cov01)
        self.assertAlmostEqual(cov[1, 1],  cov11)

    def computeCheckCov(self, cov00, cov01, cov11):
        cov = self.computeCov()
        self.checkCov(cov00, cov01, cov11, cov)

    def computeCheckFit(self, p, p_err):
        pc, pc_err = linear_fit.linear_fit_1d(self.x, self.y)
        a, b = p
        a_err, b_err = p_err
        self.assertAlmostEqual(pc[0], a)
        self.assertAlmostEqual(pc[1], b)
        self.assertAlmostEqual(pc_err[0], a_err)
        self.assertAlmostEqual(pc_err[1], b_err)

    def test00_x_to_cov(self):
        self.computeCheckCov(self.cov00, self.cov01, self.cov11)

    def test01_fit(self):
        self.computeCheckFit(self.p, self.p_err)


class Test01_ZeroSlope(_TestLineFit):
    '''Line identical to x axis
    '''

    x = [0, 1, 2]
    y = [0, 0, 0]

    cov00 = 1/2
    cov01 = -1/2
    cov11 = 1 - 1/(2*3)

    p = [0, 0]
    p_err = [0, 0]


class Test02_Slope1(Test01_ZeroSlope):
    '''Line with slope one
    '''
    y = [0, 1, 2]
    p = [1, 0]


class Test03_IdealLine(_TestLineFit):
    '''Line with slope 3 and intercept 7
    '''
    x = [-1, 5, 3]
    y = np.array([-3, 15, 9]) + 7

    cov00 = 3 / (7 * 8)
    cov01 = - 1 / 8
    cov11 = 5**4 / 1000

    p = [3, 7]
    p_err = [0, 0]


class Test03_IdealLineWithNoise(Test03_IdealLine):
    '''Line with slope 3 and intercept 7
    '''
    p_err = [0.02, 0.068]

    def setUp(self):
        '''with determenistic noise

        Double points so that fit parameter are the smae as for
        the central line
        '''
        super().setUp()
        x = self.x
        y = self.y
        x = np.concatenate([x, x])
        y = np.concatenate([y, y])
        self.x = x

        # Add determenistic noise
        offset = .1
        off = np.ones(len(self.y)) * offset
        off = np.concatenate([off, -off])
        self.y = y + off

        self.cov00 = 1/8 * self.cov00
        self.cov01 = 1/8 * self.cov01
        self.cov11 = 1/8 * self.cov11

    def test01_fit(self):
        p, p_err = linear_fit.linear_fit_1d(self.x, self.y)
        self.assertAlmostEqual(p[0], self.p[0])
        self.assertAlmostEqual(p[1], self.p[1])
        self.assertAlmostEqual(p_err[0], self.p_err[0], places=3)
        self.assertAlmostEqual(p_err[1], self.p_err[1], places=3)


del _TestLineFit

if __name__ == '__main__':
    unittest.main()
