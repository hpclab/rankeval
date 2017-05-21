"""Class for efficient scoring of an ensemble-based model composed of binary regression trees on a given dataset."""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>

import numpy as np

from rankeval.core.model.regression_trees_ensemble import RegressionTreesEnsemble


class Scoring(object):
    """
    Class for efficient scoring of an ensemble-based model composed of binary regression trees on a given dataset.

    Notes
    ----------
    This class can be used for a simple or detailed scoring, depending on the mode selected at initialization time

    Parameters
    ----------
    model: RegressionTreesEnsemble
        The model to use for scoring
    X: numpy array of float
        The dense numpy matrix of shape (n_samples, n_features)

    Attributes
    ----------

    """
    def __init__(self, model, X):
        self.model = model
        self.X = X
        self.basic_scoring = False
        self.adv_scoring = False
        self.y = np.zeros(X.shape[0])
        # used to save the partial scores of each tree
        self.partial_y = np.zeros(model.n_trees)

    def score(self, detailed):
        """

        Parameters
        ----------
        detailed : bool
            True if the class has to performs a detailed scoring, false otherwise

        Returns
        -------
        y : numpy array of float
            the predicted score produced by the given model for each samples of the given dataset X
        """
        if self.basic_scoring:
            return self.y

        for idx, sample_features in enumerate(self.X):
            self.y[idx] = self._score_single_sample(sample_features)

        self.basic_scoring = True
        self.adv_scoring = detailed

        return self.y

    def _score_single_sample(self, sample_features):
        for idx_tree in np.arange(self.model.n_trees):
            self.partial_y[idx_tree] = self._score_single_sample_single_tree(sample_features, idx_tree)
        return self.partial_y.sum()

    def _score_single_sample_single_tree(self, sample_features, idx_tree):
        cur_node = self.model.trees_root[idx_tree]
        while not self.model.is_leaf_node(cur_node):
            feature_idx = self.model.trees_nodes_feature[cur_node]
            feature_threshold = self.model.trees_nodes_value[cur_node]
            if sample_features[feature_idx] < feature_threshold:
                cur_node = self.model.trees_left_child[cur_node]
            else:
                cur_node = self.model.trees_right_child[cur_node]
        return self.model.trees_nodes_value[cur_node]