import numpy as np
import os
import unittest
import logging

from numpy.testing import assert_equal, assert_array_equal
from nose.tools import raises
from rankeval.core.metrics.precision import precision_at_k



# qid scores: {1:[1,0,0], 2:[0,1,0], 3:[0,0,1], 4:[0,0,0]}
scores = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
qid_offsets = [0,3,6,9,12]

class SVMLightLoaderTestCase(unittest.TestCase):

    def test_precision_at_k(self):
        p = precision_at_k(scores, labels, qid_offsets, 1)
        print p
        assert_equal(p, 3./4)
