# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Class for efficient modelling of an ensemble-based model of binary regression
trees.
"""

import copy

import numpy as np

from ..scoring.scorer import Scorer


class RTEnsemble(object):
    """
    Class for efficient modelling of an ensemble-based model composed of binary
    regression trees.

    Notes
    ----------
    This class only provides the sketch of the data structure to use for storing
    the model. The responsibility to correctly fill these data structures is
    delegated to the various proxies model.
    """

    def __init__(self, file_path, name=None, format="QuickRank",
                 base_score=None, learning_rate=1, n_trees=None):
        """
        Load the model from the file identified by file_path using the given
        format.

        Parameters
        ----------
        file_path : str
            The fpath to the filename where the model has been saved
        name : str
            The name to be given to the current model
        format : ['QuickRank', 'ScikitLearn', 'XGBoost', 'LightGBM']
            The format of the model to load.
        base_score : None or float
            The initial prediction score of all instances, global bias.
            If None, it uses default value used by each software
            (0.5 XGBoost, 0.0 all the others).
        learning_rate : None or float
            The learning rate used by the model to shrinks the contribution of
             each tree. By default it is set to 1 (no shrinking at all).
        n_trees : None or int
            The maximum number of trees to load from the model. By default it is
            set to None, meaning the method will load all the trees.

        Attributes
        ----------
        file : str
            The path to the filename where the model has been saved
        name : str
            The name to be given to the current model
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
        trees_weight: list of floats
            Numpy array modelling the weights of the regression trees composing the ensemble.
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
            feature identified by the trees_nodes_feature data structure).
        trees_nodes_feature: list of integers
            Numpy array modelling the feature-id used by the selected splitting node (or -1 if the node is a leaf).

        Returns
        -------
        model : RegressionTreeEnsemble
            The loaded model as a RTEnsemble object
        """
        self.file = file_path
        self.name = "RTEnsemble: " + file_path
        if name is not None:
            self.name = name
        self.learning_rate = learning_rate

        self.base_score = base_score
        if self.base_score is None and format == "XGBoost":
            self.base_score = 0.5

        self.n_trees = None
        self.n_nodes = None

        self.trees_root = None
        self.trees_weight = None
        self.trees_left_child = None
        self.trees_right_child = None
        self.trees_nodes_value = None
        self.trees_nodes_feature = None

        self._cache_scorer = dict()

        if format == "QuickRank":
            from rankeval.model import ProxyQuickRank
            ProxyQuickRank.load(file_path, self)
        elif format == "LightGBM":
            from rankeval.model import ProxyLightGBM
            ProxyLightGBM.load(file_path, self)
        elif format == "XGBoost":
            from rankeval.model import ProxyXGBoost
            ProxyXGBoost.load(file_path, self)
        elif format == "ScikitLearn":
            from rankeval.model import ProxyScikitLearn
            ProxyScikitLearn.load(file_path, self)
        elif format == "Jforests":
            from rankeval.model import ProxyJforests
            ProxyJforests.load(file_path, self)
        elif format == "CatBoost":
            from rankeval.model import ProxyCatBoost
            ProxyCatBoost.load(file_path, self)
        else:
            raise TypeError("Model format %s not yet supported!" % format)

        if n_trees is not None and n_trees < self.n_trees:
            self._prune_model(n_trees)

    def initialize(self, n_trees, n_nodes):
        """
        Initialize the internal data structures in order to reflect the given
        shape and size of the ensemble. This method should be called only by
        the Proxy Models (the specific format-based loader/saver)

        Parameters
        ----------
        n_trees : integer
            The number of regression trees in the ensemble.
        n_nodes : integer
            The total number of nodes (splitting nodes and leaves) in the
            ensemble
        """
        self.n_trees = n_trees
        self.n_nodes = n_nodes

        self.trees_root = np.full(shape=n_trees, fill_value=-1, dtype=np.int32)
        self.trees_weight = \
            np.zeros(shape=n_trees, dtype=np.float32)
        self.trees_left_child = \
            np.full(shape=n_nodes, fill_value=-1, dtype=np.int32)
        self.trees_right_child = \
            np.full(shape=n_nodes, fill_value=-1, dtype=np.int32)
        self.trees_nodes_value = \
            np.full(shape=n_nodes, fill_value=-1, dtype=np.float32)
        self.trees_nodes_feature = \
            np.full(shape=n_nodes, fill_value=-1, dtype=np.int16)

    def is_leaf_node(self, index):
        """
        This method returns true if the node identified by the given index is a
        leaf node, false otherwise

        Parameters
        ----------
        index : integer
            The index of the node to test
        """
        return self.trees_left_child[index] == -1 and \
               self.trees_right_child[index] == -1

    def max_leaves(self):
        """
        Computes the maximum number of leaves across the trees of the model.

        Returns
        -------
        max_leaves : int
            Maximum number of leaves
        """
        return self.num_leaves().max()

    def num_leaves(self):
        """
        Computes the number of leaves for each tree of the model.

        Returns
        -------
        n_leaves : numpy 1d array (n_trees)
            Number of leaves
        """

        n_leaves = np.empty(shape=self.n_trees, dtype=np.int32)
        for idx_tree in np.arange(self.n_trees):
            root_node = self.trees_root[idx_tree]
            if idx_tree+1 == self.n_trees:
                next_root_node = self.n_nodes
            else:
                next_root_node = self.trees_root[idx_tree + 1]
            n_leaves[idx_tree] = (self.trees_left_child[root_node:next_root_node] == -1).sum()

        return n_leaves

    def num_nodes(self):
        """
        Computes the number of nodes for each tree of the model.

        Returns
        -------
        n_nodes : numpy 1d array (n_trees)
            Number of nodes
        """

        n_nodes = np.empty(shape=self.n_trees, dtype=np.int32)
        for idx_tree in np.arange(self.n_trees):
            root_node = self.trees_root[idx_tree]
            if idx_tree+1 == self.n_trees:
                next_root_node = self.n_nodes
            else:
                next_root_node = self.trees_root[idx_tree + 1]
            n_nodes[idx_tree] = next_root_node - root_node

        return n_nodes

    def height_trees(self):
        """
        Computes the height of each tree of the model.

        Returns
        -------
        heights : numpy 1d array (n_trees)
            Height of each tree
        """
        def height_node(idx_node, depth):
            if self.trees_left_child[idx_node] == -1:
                return depth
            else:
                return max(
                    height_node(self.trees_left_child[idx_node],
                               depth + 1),
                    height_node(self.trees_right_child[idx_node],
                               depth + 1)
                )

        height_trees = np.empty(shape=self.n_trees, dtype=np.int32)
        for idx_tree in np.arange(self.n_trees):
            root_node = self.trees_root[idx_tree]
            height_trees[idx_tree] = height_node(root_node, 0)
        return height_trees

    def save(self, f, format="QuickRank"):
        """
        Save the model onto the file identified by file_path, using the given
        model format.

        Parameters
        ----------
        f : str
            The path to the filename where the model has to be saved
        format : str
            The format to use for saving the model

        Returns
        -------
        status : bool
            Returns true if the save is successful, false otherwise
        """
        if format == "QuickRank":
            from rankeval.model import ProxyQuickRank
            return ProxyQuickRank.save(f, self)
        elif format == "LightGBM":
            from rankeval.model import ProxyLightGBM
            return ProxyLightGBM.save(f, self)
        elif format == "XGBoost":
            from rankeval.model import ProxyXGBoost
            return ProxyXGBoost.save(f, self)
        elif format == "ScikitLearn":
            from rankeval.model import ProxyScikitLearn
            return ProxyScikitLearn.save(f, self)
        else:
            raise TypeError("Model format %s not yet supported!" % format)

    def score(self, dataset, detailed=False, cache=False):
        """
        Score the given model on the given dataset. Depending on the detailed
        parameter, the scoring will be either basic (i.e., compute only the
        document scores) or detailed (i.e., besides computing the document
        scores analyze also several characteristics of the model. The scorer is
        cached until existance of the model instance.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be scored
        detailed : bool
            True if the model has to be scored in a detailed fashion, false
            otherwise
        cache : bool
            True if the scoring results has to be cached, False otherwise. Take
            into account that caching the results could need quite a lot of
            memory, and this negative effect is particularly evident when
            scoring multiple models on huge dataset. Default is False.

        Returns
        -------
        y_pred : numpy 1d array (n_instances)
            The predictions made by scoring the model on the given dataset
        partial_y_pred : numpy 2d array (n_instances x n_trees)
            The predictions made by scoring the model on the given dataset, on a
            tree basis (i.e., tree by tree and instance by instance)
        y_leaves : numpy 2d array (n_instances x n_trees)
            The leaf nodes predicted by scoring the model on the given dataset,
            on a tree basis (i.e., tree by tree and instance by instance)
        """

        # check that the features used by the model are "compatible" with the
        # features in the dataset (at least, in terms of their number)
        if np.max(self.trees_nodes_feature) + 1 > dataset.n_features:
            raise RuntimeError("Dataset features are not compatible with "
                               "model features")

        if dataset not in self._cache_scorer or \
                detailed and self._cache_scorer[dataset].partial_y_pred is None:

            scorer = Scorer(self, dataset)
            if cache:
                self._cache_scorer[dataset] = scorer
            # The scoring is performed only if it has not been done before...
            scorer.score(detailed)

            if self.learning_rate != 1:
                scorer.y_pred *= self.learning_rate
                if detailed:
                    scorer.partial_y_pred *= self.learning_rate

            if self.base_score:
                scorer.y_pred += self.base_score
                if detailed:
                    scorer.partial_y_pred[:, :1] += self.base_score
        else:
            scorer = self._cache_scorer[dataset]

        if detailed:
            return scorer.y_pred, scorer.partial_y_pred, scorer.out_leaves
        else:
            return scorer.y_pred

    def clear_cache(self):
        """
        This method is used to clear the internal cache of the model from the
        scoring objects. Call this method at the end of the analysis of the
        current model (the memory otherwise will be automatically be freed on
        object deletion)
        """
        self._cache_scorer.clear()

    def copy(self, n_trees=None):
        """
        Create a copy of this model, with all the trees up to the given number.
        By default n_trees is set to None, meaning to copy all the trees

        Parameters
        ----------
        n_trees : None or int
            The number of trees the model will have after calling this method.

        Returns
        -------
        model : RTEnsemble
            The copied model, pruned from all the trees exceeding the given
            number of trees chosen
        """
        new_model = copy.deepcopy(self)
        if n_trees is not None:
            new_model._prune_model(n_trees=n_trees)
        return new_model

    def _prune_model(self, n_trees):
        """
        This method prunes the ensemble of trees up to the given number of trees
        in order to reduce the size of the model. Useful for creating smaller
        models starting from a bigger model.

        Parameters
        ----------
        n_trees : int
            The number of trees the model will have after calling this method.
        """
        # skip the pruning if the model already contains less or equals trees
        # than expected
        if n_trees >= self.n_trees:
            return

        start_idx_prune = self.trees_root[n_trees]

        self.trees_root = self.trees_root[:n_trees]
        self.trees_weight = self.trees_weight[:n_trees]

        self.trees_nodes_feature = self.trees_nodes_feature[:start_idx_prune]
        self.trees_nodes_value = self.trees_nodes_value[:start_idx_prune]
        self.trees_right_child = self.trees_right_child[:start_idx_prune]
        self.trees_left_child = self.trees_left_child[:start_idx_prune]

        self.n_trees = n_trees
        self.n_nodes = start_idx_prune

        # Reset cache scorer
        self._cache_scorer = dict()

    def __str__(self):
        return self.name
