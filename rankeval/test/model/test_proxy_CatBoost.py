import logging
import os
import unittest

import coremltools
from numpy.testing import assert_equal, assert_array_equal, \
    assert_array_almost_equal

from rankeval.dataset import Dataset
from rankeval.model import ProxyCatBoost
from rankeval.model import RTEnsemble
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "CatBoost.model.coreml")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class ProxyCatBoostTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = RTEnsemble(model_file, format="CatBoost")
        cls.dataset = Dataset.load(data_file, format="svmlight")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        del cls.dataset
        cls.dataset = None

    def test_count_nodes(self):
        coreml_model = coremltools.models.model.MLModel(model_file)
        n_trees, n_nodes = ProxyCatBoost._count_nodes(coreml_model)
        # print "Num Trees: %d\nNum Nodes: %d" % (n_trees, n_nodes),
        assert_equal(n_trees, 2)
        assert_equal(n_nodes, 14)
        assert_equal(n_trees, self.model.trees_root.size)
        assert_equal(n_nodes, self.model.trees_nodes_value.size)

    def test_root_nodes(self):
        assert_equal((self.model.trees_root > -1).all(), True,
                     err_msg="Root nodes not set correctly")

    def test_root_nodes_adv(self):
        assert_array_equal(self.model.trees_root, [0, 7],
                           err_msg="Root nodes are not correct")

    def test_split_features(self):
        assert_array_equal(self.model.trees_nodes_feature,
                           [124, 62, 62, -1, -1, -1, -1,
                            112, 107, 107, -1, -1, -1, -1])
    #
    def test_tree_values(self):
        assert_array_almost_equal(self.model.trees_nodes_value,
            [-6.988956e+00,  6.712700e-02,  6.712700e-02,  8.421655e-03,
             1.095791e-03,  8.926381e-03, -1.645530e-02,
             -1.208052e+01, 1.148206e+01,  1.148206e+01,  1.408405e-02,
             -9.354122e-04, 6.002808e-03, -1.578260e-02],
            decimal=5,
            err_msg="Split thresholds or leaf outputs value are not correct")

    def test_left_children(self):
        assert_array_equal(self.model.trees_left_child,
                           [2, 4, 6, -1, -1, -1, -1,
                            9, 11, 13, -1, -1, -1, -1])

    def test_right_children(self):
        assert_array_equal(self.model.trees_right_child,
                           [1, 3, 5, -1, -1, -1, -1,
                            8, 10, 12, -1, -1, -1, -1])

    def test_leaf_correctness(self):
        for idx, feature in enumerate(self.model.trees_nodes_feature):
            if feature == -1:
                assert_equal(self.model.trees_left_child[idx], -1,
                             "Left child of a leaf node is not empty (-1)")
                assert_equal(self.model.trees_right_child[idx], -1,
                             "Right child of a leaf node is not empty (-1)")
                assert_equal(self.model.is_leaf_node(idx), True,
                             "Leaf node not detected as a leaf")

    def test_prediction(self):
        y_pred = self.model.score(self.dataset, cache=False)
        assert_array_almost_equal(y_pred[:5],
                                  [0.00748624, -0.03223789, -0.01468681,
                                   0.02301043, -0.03223789])
        assert_array_almost_equal(y_pred[-5:],
                                  [0.02301043, -0.03223789, 0.02301043,
                                   0.02301043, -0.03223789])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    unittest.main()
