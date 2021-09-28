import unittest
import itertools
import logging
from bact_math_utils import misc

logger = logging.getLogger("bact_math_utils")


class _TestSeen(unittest.TestCase):
    N = 5
    offset = 7

    def setUp(self):
        self.range = list(range(self.offset, self.N + self.offset))

    def checkReturnedEnum(self, enum, payload, f):
        """
        Args:
            enum:    as returned by UniqueEverSeen
            payload: originally handled by iterator
            f:       function returning approprate count for payload value
        """
        # correct length returned
        self.assertEqual(len(enum), len(payload))

        # correct values returned
        for (step, tmp) in enumerate(zip(payload, enum)):
            t_r, t_e = tmp
            cnt, t_r2 = t_e
            # correct value returned
            self.assertEqual(t_r, t_r2)

            # correctly counted ... simple case
            # starting from zero
            expected = f(t_r2)
            try:
                self.assertEqual(cnt, expected)
            except Exception as exc:
                logger.error("Loop step %s: exception %s", step, exc)
                raise exc


class TestUniqueEverSeen(_TestSeen):
    def test00(self):
        """cross check test range length"""
        self.assertEqual(len(self.range), self.N)

    def test10(self):
        """test that range is properly enumerted"""

        eu = misc.EnumerateUniqueEverSeen()
        enum = list(eu(self.range))

        self.checkReturnedEnum(enum, self.range, lambda x: x - self.offset)

    def test11(self):
        """Test that a chained range is correctly numerated"""

        t_range = self.range
        r = list(itertools.chain(t_range, t_range, t_range))

        eu = misc.EnumerateUniqueEverSeen()
        enum = list(eu(r))
        self.checkReturnedEnum(enum, r, lambda x: x - self.offset)

    def test12(self):
        r = [1, 2, 1]
        eu = misc.EnumerateUniqueEverSeen()
        enum = list(eu(r))
        self.checkReturnedEnum(enum, r, lambda x: x - 1)

    def test13(self):
        """Test string variations

        as typically occurying during measurements
        """
        vals = ["Q1M1D1R"] * 3 + ["Q1M1D2R"] * 3 + ["Q1M1D1R"] * 3
        expected = [0] * 3 + [1] * 3 + [0] * 3

        eu = misc.EnumerateUniqueEverSeen()
        enum = list(eu(vals))

        gen = itertools.chain(expected)
        self.checkReturnedEnum(enum, vals, lambda x: next(gen))


class TestUniqueJustSeen(_TestSeen):

    def checkReturn(self, vals, f):
        eu = misc.EnumerateUniqueJustSeen()
        enum = list(eu(vals))

        self.checkReturnedEnum(enum, vals, f)
        r2 = misc.enumerate_changed_value(vals)

        for r, tmp in zip(r2, enum):
            self.assertEqual(r, tmp[0])

    def test01(self):
        """correctly counting classical range"""
        self.checkReturn(self.range, lambda x: x - self.offset)

    def test10(self):
        """correctly counting if number appears again"""
        r = [1, 2, 1]
        counter = itertools.count()

        self.checkReturn(r, lambda x: next(counter))

    def test11(self):
        """Test that a chained range is correctly numerated"""

        t_range = self.range
        r = list(itertools.chain(t_range, t_range, t_range))
        counter = itertools.count()

        self.checkReturn(r, lambda x: next(counter))

    def test12(self):
        """Correctly counting double occurance"""

        test = [1, 2, 2, 1]
        expected = [0, 1, 1, 2]

        gen = itertools.chain(expected)
        self.checkReturn(test, lambda x: next(gen))

    def test13(self):
        """Test string variations

        as typically occurying during measurements
        """
        vals = ["Q1M1D1R"] * 3 + ["Q1M1D2R"] * 3 + ["Q1M1D1R"] * 3
        expected = [0] * 3 + [1] * 3 + [2] * 3

        gen = itertools.chain(expected)
        self.checkReturn(vals, lambda x: next(gen))


class TestEumeratePairs(unittest.TestCase):
    def test20(self):
        '''Typical quadrupole arangement
        '''
        vals = ["Q1M1D1R"] * 3 + ["Q1M1D2R"] * 3 + ["Q1M1D1R"] * 3
        currents = [0, 1, 2] * 3

        tmp = misc.enumerate_changed_value_pairs(vals, currents)
        self.assertEqual(len(tmp), len(vals))

        for check, val in zip(tmp, range(len(tmp))):
            self.assertEqual(check, val)

    def test21(self):
        '''Real quadrupole measurement
        '''
        vals = ["Q1M1D1R"] * 9 + ["Q1M1D2R"] * 9 + ["Q1M1D1R"] * 9
        currents = list(itertools.chain([-1]*3, [0]*3, [1]*3)) * 3

        tmp = misc.enumerate_changed_value_pairs(vals, currents)

        def f():
            counter = itertools.count()
            for cnt in counter:
                for i in range(3):
                    # print(cnt)
                    yield cnt

        gen = f()
        for check, val in zip(tmp, gen):
            self.assertEqual(check, val)


del _TestSeen

if __name__ == "__main__":
    unittest.main()
