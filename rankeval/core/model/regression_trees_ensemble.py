"""Class for efficient modelling of an ensemble-based model composed of binary regression trees."""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>

import numpy as np


class RegressionTreesEnsemble(object):
    """
    Class for efficient modelling of an ensemble-based model composed of binary regression trees.

    Notes
    ----------
    This class only provides the sketch of the data structure to use for storing the model.
    The responsibility to correctly fill these data structures is delegated to the various model's loaders.

    Warning
    ----------
    This class should not be used directly. Use model's loaders instead.

    Parameters
    ----------
    n_trees : integer
        The number of regression trees in the ensemble.
    n_nodes : integer
        The total number of nodes (splitting nodes and leaves) in the ensemble

    Attributes
    ----------
    n_trees : integer
        The number of regression trees in the ensemble.
    n_nodes : integer
        The total number of nodes (splitting nodes and leaves) in the ensemble
    trees_root: list of integers
        Numpy array modelling the indexes of the root nodes of the regression trees composing the ensemble. The indexes
        refer to the following data structures:
        * trees_left_child
        * trees_right_child
        * trees_nodes_value
        * trees_nodes_feature
    trees_left_child: list of integers
        Numpy array modelling the structure (shape) of the regression trees, considering only the left children.
        Given a node of a regression tree (a single cell in this array), the value identify the index of the left
        children. If the node is a leaf, the children assumes -1 value.
    trees_right_child: list of integers
        Numpy array modelling the structure (shape) of the regression trees, considering only the right children.
        Given a node of a regression tree (a single cell in this array), the value identify the index of the right
        children. If the node is a leaf, the children assumes -1 value.
    trees_nodes_value: list of integers
        Numpy array modelling either the output of a leaf node (whether the node is a leaf, in accordance with the
        trees_structure data structure) or the splitting value of the node in the regression trees (with respect to the
        feature identified by the trees_nodes_feature data structure    ).
    trees_nodes_feature: list of integers
        Numpy array modelling the feature-id used by the selected splitting node (or -1 if the node is a leaf).
    """
    def __init__(self, n_trees, n_nodes):
        self.n_trees = n_trees
        self.n_nodes = n_nodes

        self.trees_root = np.full(shape=n_trees, fill_value=-1, dtype=np.int16)
        self.trees_weight = np.zeros(shape=n_trees, dtype=np.float32)
        self.trees_left_child = np.full(shape=n_nodes, fill_value=-1, dtype=np.int16)
        self.trees_right_child = np.full(shape=n_nodes, fill_value=-1, dtype=np.int16)
        self.trees_nodes_value = np.full(shape=n_nodes, fill_value=-1, dtype=np.float32)
        self.trees_nodes_feature = np.full(shape=n_nodes, fill_value=-1, dtype=np.int16)

    def is_leaf_node(self, index):
        """
        This method returns true if the node identified by the given index is a leaf node, false otherwise

        Parameters
        ----------
        index : integer
            The index of the node to test
        """
        return self.trees_left_child[index] == -1 and self.trees_right_child[index] == -1

