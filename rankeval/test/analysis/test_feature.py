import os
import unittest
import logging

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from rankeval.analysis.feature import feature_importance, _feature_tree_imp

from rankeval.test.base import data_dir

from rankeval.core.model import RTEnsemble
from rankeval.core.dataset import Dataset


class FeatureImportanceTestCase(unittest.TestCase):

    def setUp(self):
        self.model = RTEnsemble(
            os.path.join(data_dir, "quickrank.model.xml"),
            format="quickrank")
        self.model = RTEnsemble(
            os.path.join(data_dir, "quickrank.model.xml"),
            format="quickrank")
        self.dataset = Dataset.load(
            os.path.join(data_dir, "msn1.fold1.train.5k.txt"),
            format="svmlight")

    def tearDown(self):
        del self.model
        self.model = None
        del self.dataset
        self.dataset = None
 
    def test_feature_importance(self):
        feature_imp = feature_importance(self.model, self.dataset)

        assert_allclose(feature_imp[[7, 105, 107, 114]],
                        [0.0405271754093, 0.0215954124466,
                         0.0478155618964, 0.018661751695])

    def test_scoring_feature_importance(self):

        # default scores on the root node of the first tree
        y_pred = np.zeros(self.dataset.n_instances)

        scorer = self.model.score(self.dataset, detailed=True)

        # initialize features importance
        feature_imp = np.zeros(self.dataset.n_features)

        for tree_id in np.arange(self.model.n_trees):
            y_pred_tree = _feature_tree_imp(self.model, self.dataset, tree_id,
                                            y_pred, feature_imp)
            y_pred_tree *= self.model.trees_weight[tree_id]

            # Check the partial scores of each tree are compatible with
            # traditional scoring
            assert_allclose(y_pred_tree, scorer.partial_y_pred[:, tree_id])

            y_pred += y_pred_tree

        # Check the usual scoring and the scoring performed by analyzing also
        # the feature importance compute the same predictions
        assert_array_almost_equal(scorer.y_pred, y_pred)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
