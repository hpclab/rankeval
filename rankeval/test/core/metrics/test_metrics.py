
import numpy as np
import unittest


from numpy.testing import assert_equal, assert_almost_equal
import os

from rankeval.core.metrics.err import ERR

from rankeval.core.metrics.mrr import MRR

from rankeval.core.dataset import Dataset
from rankeval.test.base import data_dir

from rankeval.core.metrics.ndcg import NDCG

from rankeval.core.metrics.dcg import DCG
from rankeval.core.metrics.precision import Precision
from rankeval.core.metrics.recall import Recall
# from rankeval.core.model import ProxyQuickRank
# from rankeval.core.scoring import Scorer
# from rankeval.test.base import data_dir
#
# model_file = os.path.join(data_dir, "quickrank.model.xml")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        # self.model = ProxyQuickRank.load(model_file)
        # self.dataset = Dataset(data_file, format="svmlight")
        # self.scorer = Scorer(self.model, self.dataset)

        # setup document scores and predicted scored
        self.y_pred_query1 = np.array([2.3, 0.0, 0.1, 5.5, 1.0])
        self.y_pred_query2 = np.array([2.6, 0.2, 0.1, 1.5, 1.0])
        self.y_query1 = np.array([2, 3, 0, 1, 0])
        self.y_query2 = np.array([0, 1, 0, 3, 0])

        # setup a Dataset object
        self.dataset = Dataset(data_file, name="TestDataset", format="svmlight")
        self.dataset.y = np.concatenate((self.y_query1, self.y_query2))
        self.dataset.query_ids = np.array([0,5,10])
        self.dataset.n_queries = 2
        self.y_pred = np.concatenate((self.y_pred_query1, self.y_pred_query2))


    # Precision
    def test_precision_eval_per_query_cutoff(self):
        p = Precision(cutoff=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 1.)

        p = Precision(cutoff=2)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 1.)

        p = Precision(cutoff=3)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 2./3)

    def test_precision_eval_per_query_threshold(self):
        p = Precision(threshold=2)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 1./5)

        p = Precision(threshold=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 2./5)

    def test_precision_eval_per_query(self):
        p = Precision()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 3./5)
        result = p.eval_per_query(self.y_query2, self.y_pred_query2)
        assert_equal(result, 2./5)

    def test_precision_eval(self):
        p = Precision()
        precision_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        precision_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        precision_dataset = p.eval(self.dataset,self.y_pred)
        assert_almost_equal(precision_dataset[0], np.mean([precision_q1, precision_q2]), decimal=9)



    # Recall
    def test_recall_eval_per_query_cutoff(self):
        p = Recall(cutoff=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 1./3)

        p = Recall(cutoff=2)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 2./3)

        p = Recall(cutoff=3)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 2./3)


    def test_recall_eval_per_query_threshold(self):
        p = Recall(threshold=2)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 1./1)

        p = Recall(threshold=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 2./2)

    def test_recall_eval_per_query(self):
        p = Recall()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 3./3)

    def test_recall_eval(self):
        p = Recall()
        recall_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        recall_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        recall_dataset = p.eval(self.dataset, self.y_pred)
        assert_almost_equal(recall_dataset[0], np.mean([recall_q1, recall_q2]), decimal=9)


    # DCG
    def test_DCG_eval_per_query(self):
        p = DCG()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 3.422, decimal=3)

    def test_DCG_eval_per_query_IDCG(self):
        p = DCG()
        result = p.eval_per_query(self.y_query1, self.y_query1)
        assert_almost_equal(result, 4.761, decimal=3)


    def test_DCG_eval(self):
        p = DCG()
        dcg_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        dcg_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        dcg_dataset = p.eval(self.dataset, self.y_pred)
        assert_almost_equal(dcg_dataset[0], np.mean([dcg_q1, dcg_q2]), decimal=3)


    # NDCG
    def test_NDCG_eval_per_query(self):
        p = NDCG()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 0.718, decimal=3)

    def test_NDCG_eval(self):
        p = NDCG()
        ndcg_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        ndcg_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        ndcg_dataset = p.eval(self.dataset, self.y_pred)
        assert_almost_equal(ndcg_dataset[0], np.mean([ndcg_q1, ndcg_q2]), decimal=3)


    # MRR
    def test_MRR_eval_per_query_threshold(self):
        p = MRR(threshold=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 1. / 1, decimal=3)


    def test_MRR_eval_per_query(self):
        p = MRR()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 1. / 4, decimal=3)

        result = p.eval_per_query(self.y_query2, self.y_pred_query2)
        assert_almost_equal(result, 1. / 4, decimal=3)

    def test_MRR_eval(self):
        p = MRR()
        mrr_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        mrr_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        mrr_dataset = p.eval(self.dataset, self.y_pred)
        assert_almost_equal(mrr_dataset[0], np.mean([mrr_q1, mrr_q2]), decimal=3)


    # ERR
    def test_ERR_eval_per_query(self):
        p = ERR()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 1.3, decimal=3)


    # RBP