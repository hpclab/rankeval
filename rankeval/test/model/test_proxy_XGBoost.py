import logging
import os
import unittest

from numpy.testing import assert_equal, assert_array_equal, \
    assert_array_almost_equal

from rankeval.dataset import Dataset
from rankeval.model import ProxyXGBoost
from rankeval.model import RTEnsemble
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "XGBoost.model.txt")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class ProxyXGBoostTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = RTEnsemble(model_file, format="XGBoost")
        cls.dataset = Dataset.load(data_file, format="svmlight")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        del cls.dataset
        cls.dataset = None

    def test_count_nodes(self):
        n_trees, n_nodes = ProxyXGBoost._count_nodes(model_file)
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
                           [52, 14, -1, -1, 17, -1, -1,
                            54, 10, -1, -1, 52, -1, -1])

    def test_tree_values(self):
        assert_array_almost_equal(
            self.model.trees_nodes_value,
            [4.4466496e-02, 2.6499998e+01, 3.3069301e-02, -2.7455300e-02,
             2.3198698e+01, 2.8918700e-02, 6.7713000e-02, 3.5544997e-03,
             2.0949998e+02, -1.8294000e-03, -4.2518899e-02, 4.4466496e-02,
             2.0927801e-03, 3.0617099e-02],
            decimal=5,
            err_msg="Split thresholds or leaf outputs value are not correct")

    def test_left_children(self):
        assert_array_equal(self.model.trees_left_child,
                           [1, 2, -1, -1, 5, -1, -1, 8, 9, -1, -1, 12, -1, -1])

    def test_right_children(self):
        assert_array_equal(self.model.trees_right_child,
                           [4, 3, -1, -1, 6, -1, -1, 11, 10, -1, -1, 13, -1, -1])

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
        y_pred = self.model.score(self.dataset, cache=True)
        assert_array_almost_equal(y_pred[:5],
                                  [0.55953574, 0.47463751, 0.47463751,
                                   0.48639977, 0.47071534])
        assert_array_almost_equal(y_pred[-5:],
                                  [0.43002582, 0.43002582, 0.43002582,
                                   0.47071534, 0.43002582])

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    unittest.main()
