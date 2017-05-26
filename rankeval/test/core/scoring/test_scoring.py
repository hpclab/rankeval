import os
import unittest
import logging

from numpy.testing import assert_equal, assert_array_almost_equal, assert_almost_equal

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

    def test_basic_scoring_values(self):
        self.scorer.score(detailed=False)
        assert_array_almost_equal(self.scorer.get_predicted_scores()[:3],
                                  [0.16549695, 0.07126279, 0.10397919])
        assert_array_almost_equal(self.scorer.get_predicted_scores()[-3:],
                                  [0.19496895, 0.13345119, 0.07126279])

    def test_basic_scoring_sum(self):
        self.scorer.score(detailed=False)
        assert_almost_equal(self.scorer.get_predicted_scores().sum(), 621.9671631)

    def test_detailed_scoring_values(self):
        self.scorer.score(detailed=True)
        assert_array_almost_equal(self.scorer.get_partial_scores()[:3],
                                  [[0.06684528,  0.09865167],
                                   [0.03412888,  0.03713391],
                                   [0.06684528,  0.03713391]])
        assert_array_almost_equal(self.scorer.get_partial_scores()[-3:],
                                  [[0.09631728,  0.09865167],
                                   [0.09631728,  0.03713391],
                                   [0.03412888,  0.03713391]])

    def test_detailed_scoring_sum(self):
        self.scorer.score(detailed=True)
        assert_almost_equal(self.scorer.get_partial_scores().sum(), 621.9671631)
        assert_array_almost_equal(self.scorer.get_partial_scores().sum(axis=0), [312.43994141, 309.53387451])
        assert_array_almost_equal(self.scorer.get_partial_scores().sum(axis=1)[:3],
                                  [0.16549695, 0.07126279, 0.10397919])
        assert_array_almost_equal(self.scorer.get_partial_scores().sum(axis=1)[-3:],
                                  [0.19496895, 0.13345119, 0.07126279])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
