import logging
import os
import unittest

from numpy.testing import assert_equal, assert_array_equal, \
    assert_array_almost_equal

from rankeval.model import ProxyQuickRank
from rankeval.model import RTEnsemble
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "quickrank.model.xml")


class ProxyQuickRankTestCase(unittest.TestCase):

    def setUp(self):
        self.model = RTEnsemble(model_file, format="QuickRank")

    def tearDown(self):
        del self.model
        self.model = None

    def test_count_nodes(self):
        n_trees, n_nodes = ProxyQuickRank._count_nodes(model_file)
        assert_equal(n_trees, 2)
        assert_equal(n_nodes, 10)
        assert_equal(n_trees, self.model.trees_root.size)
        assert_equal(n_nodes, self.model.trees_nodes_value.size)

    def test_root_nodes(self):
        assert_equal((self.model.trees_root > -1).all(), True,
                     "Root nodes not set correctly")

    def test_root_nodes_adv(self):
        assert_array_equal(self.model.trees_root, [0, 5],
                           "Root nodes are not correct")

    def test_tree_weights(self):
        assert_array_almost_equal(self.model.trees_weight,
                                  [0.10000000149011612, 0.10000000149011612],
                                  err_msg="Tree Weights are not correct")

    def test_split_features(self):
        assert_array_equal(self.model.trees_nodes_feature,
                           [107, 114, -1, -1, -1, 7, -1, 105, -1, -1])

    def test_tree_values(self):
        assert_array_almost_equal(self.model.trees_nodes_value,
            [14.895151138305664, -8.0245580673217773, 0.3412887828162291,
             0.66845277963831218, 0.96317280453257792, 0.66666698455810547,
             0.37133907932286642, 17.135160446166992, 0.54762687170967062,
             0.98651670670179537],
            err_msg="Split threshold values or leaf outputs are not correct")

    def test_left_children(self):
        assert_array_equal(self.model.trees_left_child,
                           [1, 2, -1, -1, -1, 6, -1, 8, -1, -1])

    def test_right_children(self):
        assert_array_equal(self.model.trees_right_child,
                           [4, 3, -1, -1, -1, 7, -1, 9, -1, -1])

    def test_leaf_correctness(self):
        for idx, feature in enumerate(self.model.trees_nodes_feature):
            if feature == -1:
                assert_equal(self.model.trees_left_child[idx], -1,
                             "Left child of a leaf node is not empty (-1)")
                assert_equal(self.model.trees_right_child[idx], -1,
                             "Right child of a leaf node is not empty (-1)")
                assert_equal(self.model.is_leaf_node(idx), True,
                             "Leaf node not detected as a leaf")

    def test_load_save_quickrank_model(self):
        # save the model
        saved_model_file = model_file + ".saved.xml"
        saved = self.model.save(saved_model_file, format="QuickRank")
        assert_equal(saved, True, "File not save correctly")

        # reload the model
        model_reloaded = RTEnsemble(saved_model_file, format="QuickRank")

        os.remove(saved_model_file)

        assert_array_almost_equal(self.model.trees_root, model_reloaded.trees_root,
                                  err_msg="Tree roots are incorrect")
        assert_array_almost_equal(self.model.trees_weight, model_reloaded.trees_weight,
                                  err_msg="Tree weights are incorrect")
        assert_array_almost_equal(self.model.trees_nodes_value, model_reloaded.trees_nodes_value,
                                  err_msg="Node thresholds are incorrect")
        assert_array_almost_equal(self.model.trees_nodes_feature, model_reloaded.trees_nodes_feature,
                                  err_msg="Node features are incorrect")
        assert_array_almost_equal(self.model.trees_left_child, model_reloaded.trees_left_child,
                                  err_msg="Left children are incorrect")
        assert_array_almost_equal(self.model.trees_right_child, model_reloaded.trees_right_child,
                                  err_msg="Right children are incorrect")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    unittest.main()
