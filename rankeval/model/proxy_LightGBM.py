# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Class providing the implementation for loading/storing a LightGBM model
from/to file.

The LightGBM project is described here:
    https://github.com/Microsoft/LightGBM

The LightGBM format adopts a textual representation using arrays for storing
split nodes (both features and thresholds), leaf values and tree structure.
Not all the information reported in the model are useful for the different
analysis, thus only the relevant parts are parsed.

NOTE: the leaves output of the regression trees already take into account the
weight of the tree (i.e., the learning rate or shrinkage factor). In order to
maintain the scoring made by rankeval (that multiply the leaf output by the tree
weight), the weight of the trees have been set equals to 1.

NOTE: currently rankeval does not support missing features. This features is not
strictly required because all the most popular open-domain Learning to Rank
datasets do not have missing values. To this regard, take thus into account that
you may experience an unexpected behaviour if the LightGBM model you are trying
to load is trained on a dataset with missing values (especially in the case you
use the setting zero_as_missing=true).
"""

import re

import numpy as np

from .rt_ensemble import RTEnsemble

tree_reg = re.compile("^Tree=(\d+)")
num_leaves_reg = re.compile("^num_leaves=(\d+)")
split_feature_reg = re.compile("^split_feature=(.*)")
threshold_reg = re.compile("^threshold=(.*)")
decision_type_reg = re.compile("^decision_type=(.*)")
default_value_reg = re.compile("^default_value=(.*)")
left_child_reg = re.compile("^left_child=(.*)")
right_child_reg = re.compile("^right_child=(.*)")
leaf_parent_reg = re.compile("^leaf_parent=(.*)")
leaf_value_reg = re.compile("^leaf_value=(.*)")
shrinkage_reg = re.compile("^shrinkage=(.*)")
has_categorical_reg = re.compile("^has_categorical=(.*)")


class ProxyLightGBM(object):
    """
    Class providing the implementation for loading/storing a LightGBM model
    from/to file.
    """

    @staticmethod
    def load(file_path, model):
        """
        Load the model from the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved
        model : RTEnsemble
            The model instance to fill
        """
        n_trees, n_nodes = ProxyLightGBM._count_nodes(file_path)
        # Initialize the model and allocate the needed space
        # given the shape and size of the ensemble
        model.initialize(n_trees, n_nodes)

        curr_tree = root_node = 0
        num_leaves = num_splits = 0
        with open(file_path, 'r') as f:
            for line in f:

                match = tree_reg.match(line)
                if match:
                    curr_tree = int(match.group(1))
                    root_node += num_leaves + num_splits
                    model.trees_root[curr_tree] = root_node
                    continue

                match = split_feature_reg.match(line)
                if match:
                    split_features = list( map(int, match.group(1).strip().split()) )
                    for pos, feature in enumerate(split_features):
                        model.trees_nodes_feature[root_node + pos] = feature
                    num_splits = len(split_features)
                    continue

                match = threshold_reg.match(line)
                if match:
                    thresholds = list( map(float, match.group(1).strip().split()) )
                    for pos, threshold in enumerate(thresholds):
                        model.trees_nodes_value[root_node + pos] = threshold
                    continue

                match = left_child_reg.match(line)
                if match:
                    left_children = list( map(int, match.group(1).strip().split()) )
                    for pos, child in enumerate(left_children):
                        if child >= 0:
                            model.trees_left_child[root_node + pos] = \
                                root_node + child
                        else:
                            model.trees_left_child[root_node + pos] = \
                                root_node + num_splits + abs(child) - 1
                    continue

                match = right_child_reg.match(line)
                if match:
                    right_children = list( map(int, match.group(1).strip().split()) )
                    for pos, child in enumerate(right_children):
                        if child >= 0:
                            model.trees_right_child[root_node + pos] = \
                                root_node + child
                        else:
                            model.trees_right_child[root_node + pos] = \
                                root_node + num_splits + abs(child) - 1
                    continue

                match = shrinkage_reg.match(line)
                if match:
                    # weight should be the shrinkage but it is set to 1 because
                    # leaves output already take into account the shrinkage
                    # shrinkage = float(match.group(1))
                    model.trees_weight[curr_tree] = 1.0
                    continue

                match = leaf_value_reg.match(line)
                if match:
                    leaf_values = list( map(float, match.group(1).strip().split()) )
                    num_leaves = len(leaf_values)
                    for pos, leaf_value in enumerate(leaf_values):
                        model.trees_nodes_value[root_node + num_splits + pos] \
                            = leaf_value
                    num_splits = len(split_features)
                    continue

                match = decision_type_reg.match(line)
                if match:
                    types = np.array(match.group(1).strip().split(), dtype=int)
                    if (types != 2).any():
                        raise AssertionError("Decision Tree not supported. RankEval does not "
                                             "support categorical features and missing values.")
                    continue

                match = default_value_reg.match(line)
                if match:
                    values = np.array(match.group(1).strip().split(),
                                      dtype=np.float64)
                    if values.any():
                        raise AssertionError("Missing Values not supported!")
                    continue

                match = has_categorical_reg.match(line)
                if match:
                    categorical = bool(int(match.group(1)))
                    if categorical:
                        raise AssertionError("Decision Tree not supported")
                    continue


    @staticmethod
    def save(file_path, model):
        """
        Save the model onto the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has to be saved
        model : RTEnsemble
            The model RTEnsemble model to save on file

        Returns
        -------
        status : bool
            Returns true if the save is successful, false otherwise
        """
        raise NotImplementedError("Feature not implemented!")

    @staticmethod
    def _count_nodes(file_path):
        """
        Count the total number of nodes (both split and leaf nodes)
        in the model identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved

        Returns
        -------
        tuple(n_trees, n_nodes) : tuple(int, int)
            The total number of trees and nodes (both split and leaf nodes)
            in the model identified by file_path.
        """

        n_nodes = 0
        n_trees = 0

        with open(file_path, 'r') as f:
            for line in f:
                match = num_leaves_reg.match(line)
                if match:
                    n_nodes += int(match.group(1))
                    continue
                match = split_feature_reg.match(line)
                if match:
                    n_nodes += len(match.group(1).strip().split())
                    continue
                match = tree_reg.match(line)
                if match:
                    n_trees += 1
                    continue

        return n_trees, n_nodes
