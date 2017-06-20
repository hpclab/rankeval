import os
import unittest
import logging

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from rankeval.analysis.statistical import randomization

class StatisticalSignificanceTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
 
    def test_randomization(self):
        A = np.array([1,1,1,1,1,1,1,0,0,0])
        B = np.array([0,0,0,0,0,0,0,1,1,1])
        n_perm = 10000


        p1,p2 = randomization( A, B, n_perm)

        assert_almost_equal(p2, .34195, decimal=2) # compute with https://github.com/searchivarius/PermTest


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
