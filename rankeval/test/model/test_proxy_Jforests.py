import logging
import os
import unittest

from numpy.testing import assert_equal, assert_array_equal, \
    assert_array_almost_equal

from rankeval.dataset import Dataset
from rankeval.model import ProxyJforests
from rankeval.model import RTEnsemble
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "Jforests.model.xml")
data_file = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class ProxyJforestsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = RTEnsemble(model_file, format="Jforests")
        cls.dataset = Dataset.load(data_file, format="svmlight")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cls.model = None
        del cls.dataset
        cls.dataset = None

    def test_count_nodes(self):
        n_trees, n_nodes = ProxyJforests._count_nodes(model_file)
        assert_equal(n_trees, 2)
        assert_equal(n_nodes, 26)
        assert_equal(n_trees, self.model.trees_root.size)
        assert_equal(n_nodes, self.model.trees_nodes_value.size)

    def test_root_nodes(self):
        assert_equal((self.model.trees_root > -1).all(), True,
                     "Root nodes not set correctly")

    def test_root_nodes_adv(self):
        assert_array_equal(self.model.trees_root, [0, 13],
                           "Root nodes are not correct")

    def test_tree_weights(self):
        assert_array_almost_equal(self.model.trees_weight,
                                  [1.0, 1.0],
                                  err_msg="Tree Weights are not correct")

    def test_split_features(self):
        assert_array_equal(self.model.trees_nodes_feature,
                           [129, 129, 107, 72, 55, 54,
                            -1, -1, -1, -1, -1, -1, -1,
                            133, 72, 105, 130, 62, 121,
                            -1, -1, -1, -1, -1, -1, -1])

    def test_tree_values(self):
        assert_array_almost_equal(self.model.trees_nodes_value,
            [268.0079, 265.0144, 13.9174, 19.1123, 0.00976, 0.0185,
             -1.2156, -0.2370, -1.9329, 0.8030, -0.01019, -1.9395, 0.5840,
             0.0, 21.3979, 13.2636, 181.0142, 0.3333, -1.5976, -0.1443,
             1.3819, 1.7707, 1.7353, 0.2240, -0.3769, -1.7937],
            decimal=4,
            err_msg="Split threshold values or leaf outputs are not correct")

    def test_left_children(self):
        assert_array_equal(self.model.trees_left_child,
                           [1,  3,  5,  6,  8,  7, -1, -1, -1, -1, -1, -1, -1,
                            14, 17, 21, 22, 19, 24, -1, -1, -1, -1, -1, -1, -1])

    def test_right_children(self):
        assert_array_equal(self.model.trees_right_child,
                           [2, 4, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1, -1,
                            20, 15, 16, 23, 18, 25, -1, -1, -1, -1, -1, -1, -1])

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
        y_pred = self.model.score(self.dataset)
        assert_array_almost_equal(y_pred[:5],
                                  [-2.083870, -1.359969, -1.359969,
                                   0.426128, -0.381351])
        assert_array_almost_equal(y_pred[-5:],
                                  [1.027176, -0.381351, -2.077223,
                                   0.658770, -0.381351])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    unittest.main()
