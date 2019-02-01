# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Class providing the implementation for loading/storing a XGBoost model
from/to file. The model has to be saved using textual representation, i.e., by
using the following method:
.. code-block:: python
    import xgboost as xgb
    ...
    bst = xgb.train(param, dtrain, num_round)
    bst.dump_model('xgboost.model')

The XGBoost project is described here:
    https://github.com/dmlc/xgboost

The XGBoost format adopts a textual representation where each line of the file
represent a single split node or a leaf node, with several attributes describing
the feature and the threshold involved (in case of a split node) or the output
(in case of a leaf). Each node is identified by a unique integer as well as
additional information not usefull for rankeval and thus ignored.

NOTE: the XGBoost version 0.6 does not properly dump the model. Indeed, as
reported in the issue here:

- https://github.com/dmlc/xgboost/issues/2077

The precision of the dumping is not sufficient and cause inconsistencies with
the XGBoost model. This inconsistencies cause rankeval scoring to return
different predictions with respect to the original model. Without a fix by
XGBoost authors, DO NOT USE this proxy.
"""

import re
import sys
import numpy as np

from .rt_ensemble import RTEnsemble

tree_reg = re.compile("^booster\[(\d+)\]")
node_reg = re.compile("(\d+):\[f(\d+)<(.*)\]")
leaf_reg = re.compile("(\d+):leaf=(.+?)(,.*)?$")


class ProxyXGBoost(object):
    """
    Class providing the implementation for loading/storing a XGBoost model
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
        n_trees, n_nodes = ProxyXGBoost._count_nodes(file_path)
        # Initialize the model and allocate the needed space
        # given the shape and size of the ensemble
        model.initialize(n_trees, n_nodes)

        root_node = 0
        num_nodes = 0
        queue = list()
        with open(file_path, 'r') as f:
            for line in f:

                match_tree = tree_reg.match(line)
                if match_tree:
                    assert(len(queue) == 0)
                    curr_tree = int(match_tree.group(1))
                    root_node += num_nodes
                    num_nodes = 0
                    model.trees_root[curr_tree] = root_node
                    model.trees_weight[curr_tree] = 1
                    continue

                match_node = node_reg.search(line)
                if match_node:
                    node_id = num_nodes + root_node
                    feature_id = int(match_node.group(2).strip())
                    threshold = np.float32(match_node.group(3).strip())

                    # Needed because XGBoost use as split condition
                    # < in place of <=
                    threshold = np.nextafter(
                        threshold, threshold - 1,
                        dtype=model.trees_nodes_value.dtype)

                    model.trees_nodes_feature[node_id] = feature_id
                    model.trees_nodes_value[node_id] = threshold

                match_leaf = leaf_reg.search(line)
                if match_leaf:
                    node_id = num_nodes + root_node
                    leaf_value = float(match_leaf.group(2).strip())
                    model.trees_nodes_value[node_id] = leaf_value

                if match_node or match_leaf:
                    num_nodes += 1
                    if len(queue) > 0:
                        parent_id, child = queue.pop()
                        if child == 'L':
                            model.trees_left_child[parent_id] = node_id
                        else:
                            model.trees_right_child[parent_id] = node_id

                if match_node:
                    # two elements in the queue for the left and right children
                    # Each element is identified by a node_id and the indication
                    # of being the left or right child.
                    queue.extend([(node_id, 'R'), (node_id, 'L')])

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

                match = tree_reg.match(line)
                if match:
                    n_trees += 1
                    continue

                match_node = node_reg.search(line)
                if match_node:
                    n_nodes += 1

                match_leaf = leaf_reg.search(line)
                if match_leaf:
                    n_nodes += 1

        return n_trees, n_nodes
