import numpy as np
import unittest


from numpy.testing import assert_equal, assert_almost_equal
# import os
# from rankeval.core.dataset import Dataset
from rankeval.core.metrics.ndcg import NDCG

from rankeval.core.metrics.dcg import DCG
from rankeval.core.metrics.precision import Precision
# from rankeval.core.model import ProxyQuickRank
# from rankeval.core.scoring import Scorer
# from rankeval.test.base import data_dir
#
# model_file = os.path.join(data_dir, "quickrank.model.xml")
# data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")
from rankeval.core.metrics.recall import Recall


class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        # self.model = ProxyQuickRank.load(model_file)
        # self.dataset = Dataset(data_file, format="svmlight")
        # self.scorer = Scorer(self.model, self.dataset)

        self.query_scores = np.array([2.3, 0.0, 0.1, 5.5, 1.0])
        self.query_labels = np.array([2, 3, 0, 1, 0])


    # Precision

    def test_precision_eval_per_query_cutoff(self):
        p = Precision(cutoff=1)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 1.)

        p = Precision(cutoff=2)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 1.)

        p = Precision(cutoff=3)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 2./3)

    def test_precision_eval_per_query_threshold(self):
        p = Precision(threshold=2)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 1./5)

        p = Precision(threshold=1)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 2./5)

    def test_precision_eval_per_query(self):
        p = Precision()
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 3./5)

    def test_precision_eval(self):
        pass


    # Recall

    def test_recall_eval_per_query_cutoff(self):
        p = Recall(cutoff=1)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 1./3)

        p = Recall(cutoff=2)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 2./3)

        p = Recall(cutoff=3)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 2./3)


    def test_recall_eval_per_query_threshold(self):
        p = Recall(threshold=2)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 1./1)

        p = Recall(threshold=1)
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 2./2)

    def test_recall_eval_per_query(self):
        p = Recall()
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_equal(result, 3./3)

    def test_recall_eval(self):
        pass


    # DCG

    def test_DCG_eval_per_query(self):
        p = DCG()
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_almost_equal(result, 3.422, decimal=3)

    def test_DCG_eval_per_query_IDCG(self):
        p = DCG()
        result = p.eval_per_query(self.query_labels, self.query_labels)
        assert_almost_equal(result, 4.761, decimal=3)


    # NDCG

    def test_NDCG_eval_per_query(self):
        p = NDCG()
        result = p.eval_per_query(self.query_labels, self.query_scores)
        assert_almost_equal(result, 0.718, decimal=3)