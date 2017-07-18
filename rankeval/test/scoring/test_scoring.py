import logging
import os
import unittest

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from rankeval.dataset import Dataset
from rankeval.model import RTEnsemble
from rankeval.scoring.scorer import Scorer
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "quickrank.model.xml")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class ScoringTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = RTEnsemble(model_file, format="QuickRank")
        cls.dataset = Dataset.load(data_file, format="svmlight")
        cls.scorer = Scorer(cls.model, cls.dataset)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        del cls.dataset
        cls.dataset = None
        del cls.scorer
        cls.scorer = None

    def test_basic_scoring_values(self):
        self.scorer.score(detailed=False)
        assert_array_almost_equal(self.scorer.get_predicted_scores()[:3],
                                  [0.16549695, 0.07126279, 0.10397919])
        assert_array_almost_equal(self.scorer.get_predicted_scores()[-3:],
                                  [0.13345119, 0.13345119, 0.07126279])

    def test_basic_scoring_sum(self):
        self.scorer.score(detailed=False)
        assert_almost_equal(self.scorer.get_predicted_scores().sum(),
                            598.72852, decimal=5)

    def test_detailed_scoring_values(self):
        self.scorer.score(detailed=True)
        assert_array_almost_equal(
            self.scorer.get_partial_predicted_scores()[:3],
            [[0.06684528,  0.09865167],
            [0.03412888,  0.03713391],
            [0.06684528,  0.03713391]])
        assert_array_almost_equal(
            self.scorer.get_partial_predicted_scores()[-3:],
            [[0.09631728,  0.03713391],
            [0.09631728,  0.03713391],
            [0.03412888,  0.03713391]])

    def test_basic_and_detailed_scoring(self):
        self.scorer.score(detailed=False)
        y_pred_basic = self.scorer.y_pred
        self.scorer.score(detailed=True)
        y_pred_detailed = self.scorer.y_pred
        assert_array_almost_equal(y_pred_basic, y_pred_detailed)

    def test_detailed_scoring_sum(self):
        self.scorer.score(detailed=True)
        assert_almost_equal(self.scorer.get_partial_predicted_scores().sum(),
                            598.72852, decimal=5)
        assert_array_almost_equal(
            self.scorer.get_partial_predicted_scores().sum(axis=0),
            [312.43994141, 286.2948])
        assert_array_almost_equal(
            self.scorer.get_partial_predicted_scores().sum(axis=1)[:3],
            [0.16549695, 0.07126279, 0.10397919])
        assert_array_almost_equal(
            self.scorer.get_partial_predicted_scores().sum(axis=1)[-3:],
            [0.13345119, 0.13345119, 0.07126279])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    unittest.main()
