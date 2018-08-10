
import os
import unittest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from rankeval.dataset import Dataset
from rankeval.metrics import MAP, Precision, Recall, NDCG, DCG, RBP, MRR, ERR
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "quickrank.model.xml")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class MetricsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # setup document scores and predicted scored
        cls.y_pred_query1 = np.array([2.3, 0.0, 0.1, 5.5, 1.0])
        cls.y_pred_query2 = np.array([2.6, 0.2, 0.1, 1.5, 1.0])
        cls.y_query1 = np.array([2, 3, 0, 1, 0])
        cls.y_query2 = np.array([0, 1, 0, 3, 0])

        # setup a Dataset object
        cls.dataset = Dataset.load(data_file, name="TestDataset", format="svmlight")
        cls.dataset.y = np.concatenate((cls.y_query1, cls.y_query2))
        cls.dataset.query_ids = np.array([0, 1])
        cls.dataset.query_offsets = np.array([0,5,10])
        cls.dataset.n_queries = 2
        cls.y_pred = np.concatenate((cls.y_pred_query1, cls.y_pred_query2))

    @classmethod
    def tearDownClass(cls):
        del cls.y_pred_query1
        cls.y_pred_query1 = None
        del cls.y_pred_query2
        cls.y_pred_query2 = None
        del cls.y_query1
        cls.y_query1 = None
        del cls.y_query2
        cls.y_query2 = None
        del cls.dataset
        cls.dataset = None
        del cls.y_pred
        cls.y_pred = None

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
        assert_equal(result, 2./5)

        p = Precision(threshold=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_equal(result, 3./5)

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
        assert_almost_equal(recall_dataset[0], np.mean([recall_q1, recall_q2]),
                            decimal=9)

    def test_DCG_eval_per_query_flat(self):
        idx_y_pred_sorted = np.argsort(self.y_pred_query1)[::-1]

        # with cutoff
        p = DCG(cutoff=3)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result,
                            self.y_query1[idx_y_pred_sorted[0]]/np.log2(2.)
                            + self.y_query1[idx_y_pred_sorted[1]]/np.log2(3.)
                            + self.y_query1[idx_y_pred_sorted[2]]/np.log2(4.)
                            , decimal=3)


        # without cutoff
        # result: correct value is 3.422
        p = DCG()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result,
                            self.y_query1[idx_y_pred_sorted[0]] / np.log2(2.)
                            + self.y_query1[idx_y_pred_sorted[1]] / np.log2(3.)
                            + self.y_query1[idx_y_pred_sorted[2]] / np.log2(4.)
                            + self.y_query1[idx_y_pred_sorted[3]] / np.log2(5.)
                            + self.y_query1[idx_y_pred_sorted[4]] / np.log2(6.)
                            , decimal=3)


    def test_DCG_eval_per_query_exp(self):
        idx_y_pred_sorted = np.argsort(self.y_pred_query1)[::-1]

        # with cutoff
        p = DCG(cutoff=3)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result,
                            self.y_query1[idx_y_pred_sorted[0]]/np.log2(2.)
                            + self.y_query1[idx_y_pred_sorted[1]]/np.log2(3.)
                            + self.y_query1[idx_y_pred_sorted[2]]/np.log2(4.)
                            , decimal=3)


        # without cutoff
        # result: correct value is 3.422
        p = DCG()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result,
                            self.y_query1[idx_y_pred_sorted[0]] / np.log2(2.)
                            + self.y_query1[idx_y_pred_sorted[1]] / np.log2(3.)
                            + self.y_query1[idx_y_pred_sorted[2]] / np.log2(4.)
                            + self.y_query1[idx_y_pred_sorted[3]] / np.log2(5.)
                            + self.y_query1[idx_y_pred_sorted[4]] / np.log2(6.)
                            , decimal=3)


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

    def test_NDCG_eval_per_query(self):
        p = NDCG()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 0.596, decimal=3)

    def test_NDCG_eval(self):
        p = NDCG()
        ndcg_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        ndcg_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        ndcg_dataset = p.eval(self.dataset, self.y_pred)
        assert_almost_equal(ndcg_dataset[0], np.mean([ndcg_q1, ndcg_q2]), decimal=3)

    def test_MRR_eval_per_query_threshold(self):
        p = MRR(threshold=1)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 1. / 1, decimal=3)

        p = MRR(threshold=2)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 1. / 2, decimal=3)


    def test_MRR_eval_per_query(self):
        p = MRR()
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result, 1. / 1, decimal=3)

        result = p.eval_per_query(self.y_query2, self.y_pred_query2)
        assert_almost_equal(result, 1. / 2, decimal=3)

    def test_MRR_eval(self):
        p = MRR()
        mrr_q1 = p.eval_per_query(self.y_query1, self.y_pred_query1)
        mrr_q2 = p.eval_per_query(self.y_query2, self.y_pred_query2)
        mrr_dataset = p.eval(self.dataset, self.y_pred)
        assert_almost_equal(mrr_dataset[0], np.mean([mrr_q1, mrr_q2]), decimal=3)

    def test_ERR_eval_per_query(self):
        idx_y_pred_sorted = np.argsort(self.y_pred_query1)[::-1]
        max_y = self.y_query1.max()

        # with cutoff
        p = ERR(cutoff=3)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        utility1 = (2. ** self.y_query1[idx_y_pred_sorted[0]] - 1.) / (2. ** max_y)
        utility2 = (2. ** self.y_query1[idx_y_pred_sorted[1]] - 1.) / (2. ** max_y)
        utility3 = (2. ** self.y_query1[idx_y_pred_sorted[2]] - 1.) /(2. ** max_y)
        assert_almost_equal(result,
                            utility1 / 1. * 1.
                            + utility2 / 2. * (1. - utility1)
                            + utility3 / 3. * (1. - utility1) * (1. - utility2)
                            , decimal=3)

    def test_RBP_eval_per_query(self):
        idx_y_pred_sorted = np.argsort(self.y_pred_query1)[::-1]

        # with cutoff
        p = RBP(cutoff=3, p=0.5)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result,
                            (1.0-p.p) * ((self.y_query1[idx_y_pred_sorted[0]] > 0) * (p.p**0.)
                            + (self.y_query1[idx_y_pred_sorted[1]] > 0) * (p.p**1.)
                            + (self.y_query1[idx_y_pred_sorted[2]] > 0) * (p.p**2.))
                            , decimal=3)

        # without cutoff
        # result: correct value is 3.422
        p = RBP(p=0.5)
        result = p.eval_per_query(self.y_query1, self.y_pred_query1)
        assert_almost_equal(result,
                            (1.0 - p.p) * ((self.y_query1[idx_y_pred_sorted[0]] > 0) * (p.p ** 0.)
                            + (self.y_query1[idx_y_pred_sorted[1]] > 0) * (p.p ** 1.)
                            + (self.y_query1[idx_y_pred_sorted[2]] > 0) * (p.p ** 2.)
                            + (self.y_query1[idx_y_pred_sorted[3]] > 0) * (p.p ** 3.)
                            + (self.y_query1[idx_y_pred_sorted[4]] > 0) * (p.p ** 4.))
                            , decimal=3)

    def test_MAP_eval_per_query(self):
        # without cutoff
        p = MAP()
        result = p.eval_per_query(self.y_query2, self.y_pred_query2)
        assert_almost_equal(result, 0.5, decimal=2)

        # without cutoff
        p = MAP(cutoff=3)
        result = p.eval_per_query(self.y_query2, self.y_pred_query2)
        assert_almost_equal(result, 0.25, decimal=3)

    def test_MAP_eval_per_query_countercheck(self):
        idx_y_pred_sorted = np.argsort(self.y_pred_query1)[::-1]
        sorted_y = self.y_query1[idx_y_pred_sorted]

        precision_at_i = np.zeros(self.y_query1.size, dtype=np.float32)
        recall_at_i = np.zeros(self.y_query1.size, dtype=np.float32)
        n_relevant = np.count_nonzero(self.y_query1)

        for i in np.arange(self.y_query1.size):
            relevants_at_i = float(np.count_nonzero(sorted_y[:i+1]))
            precision_at_i[i] = relevants_at_i / (i+1)
            recall_at_i[i] = relevants_at_i / n_relevant

        change_recall_at_i = np.ediff1d(recall_at_i, to_begin=recall_at_i[0])

        map_wo_cut = MAP()
        score1 = map_wo_cut.eval_per_query(self.y_query1, self.y_pred_query1)
        score2 = np.dot(precision_at_i, change_recall_at_i)
        assert_almost_equal(score1, score2, decimal=3)

        for i in np.arange(self.y_query1.size):
            idx_pos_k = np.argwhere(sorted_y[:i+1]).flatten()
            map_at_k = MAP(cutoff=i+1)
            score1 = map_at_k.eval_per_query(self.y_query1, self.y_pred_query1)
            score2 = np.sum(precision_at_i[idx_pos_k]) / min(i+1, n_relevant)
            assert_almost_equal(score1, score2, decimal=3)