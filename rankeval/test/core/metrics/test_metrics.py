import numpy as np
import unittest


from numpy.testing import assert_equal
from rankeval.core.metrics.precision import Precision


scores = np.array([2.3, 1.2, 0.0, 5.5, 1.0])
labels = np.array([2, 3, 0, 1, 0])
qid_offsets = [0,3,6,9,12]

class MetricsTestCase(unittest.TestCase):

    def test_precision(self):
        p = Precision(cutoff=1)
        result =  p.eval_per_query(labels, scores)
        print p
        assert_equal(result, 1.)

        p = Precision(cutoff=2)
        result =  p.eval_per_query(labels, scores)
        print p
        assert_equal(result, 1.)

        p = Precision(cutoff=3)
        result = p.eval_per_query(labels, scores)
        print p
        assert_equal(result, 1.)
        
        p = Precision(cutoff=5)
        result = p.eval_per_query(labels, scores)
        print p
        assert_equal(result, 3./5)