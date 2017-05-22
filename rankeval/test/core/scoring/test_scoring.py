import os
import unittest
import logging

from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal

from rankeval.core.dataset.svmlight_format import load_svmlight_file
from rankeval.core.model.proxy_quickrank import ProxyQuickRank
from rankeval.core.scoring.Scoring import Scoring
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "quickrank.model.xml")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class ScoringTestCase(unittest.TestCase):

    def setUp(self):
        self.model = ProxyQuickRank.load(model_file)
        self.X, self.y, self.qids = load_svmlight_file(data_file, query_id=True)
        self.scorer = Scoring(self.model, self.X)

    def tearDown(self):
        del self.model
        self.model = None
        del self.X
        self.X = None
        del self.y
        self.y = None
        del self.qids
        self.qids = None

    def test_basic_scoring(self):
        self.y = self.scorer.score(detailed=False)
        print self.y
        assert_equal(True, False)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
