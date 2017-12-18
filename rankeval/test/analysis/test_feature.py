import logging
import os
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, \
    assert_array_equal

from rankeval.analysis.feature import feature_importance, \
    _feature_importance_tree
from rankeval.dataset import Dataset
from rankeval.metrics import MSE
from rankeval.model import RTEnsemble
from ..base import data_dir


class FeatureImportanceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = RTEnsemble(
            os.path.join(data_dir, "quickrank.model.xml"),
            format="QuickRank")
        cls.dataset = Dataset.load(
            os.path.join(data_dir, "msn1.fold1.train.5k.txt"),
            format="svmlight")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        del cls.dataset
        cls.dataset = None
 
    def test_feature_importance(self):
        feature_imp, feature_cnt = feature_importance(self.model, self.dataset)

        features = [7, 105, 107, 114]
        assert_allclose(feature_imp[features],
                        [0.0405271754093, 0.0215954124466,
                         0.0478155618964, 0.018661751695],
                        atol=1e-6)

        assert_array_equal(feature_cnt[features],
                           [1, 1, 1, 1])

    def test_scoring_feature_importance(self):

        # default scores on the root node of the first tree
        y_pred = np.zeros(self.dataset.n_instances, dtype=np.float32)

        # initialize features importance
        feature_imp = np.zeros(self.dataset.n_features, dtype=np.float32)

        # initialize features count
        feature_count = np.zeros(self.dataset.n_features, dtype=np.uint16)

        y_pred_m, partial_y_pred, y_leaves = \
            self.model.score(self.dataset, detailed=True, cache=True)

        metric = MSE()

        for tree_id in np.arange(self.model.n_trees):
            y_pred_tree = _feature_importance_tree(self.model, self.dataset,
                                                   tree_id, y_pred, metric,
                                                   feature_imp, feature_count)
            # y_pred_tree *= self.model.trees_weight[tree_id]

            # Check the partial scores of each tree are compatible with
            # traditional scoring
            assert_allclose(y_pred_tree,
                            partial_y_pred[:, tree_id],
                            atol=1e-6)

        # Check the usual scoring and the scoring performed by analyzing also
        # the feature importance compute the same predictions
        assert_array_almost_equal(y_pred, y_pred_m)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
