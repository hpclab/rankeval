# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Class providing the implementation for loading/storing a Scikit-Learn model
from/to file. The model has to be saved using the export_scikit_model method:
.. code-block:: python
    from sklearn import ensemble
    from rankeval.model import ProxyScikitLearn
    ...
    gbrt = ensemble.GradientBoostingRegressor(...)
    gbrt.fit(X, y)
    ProxyScikitLearn.export_scikit_model(model=gbrt, file_path=model_path)
    gbrt_rankeval = RTEnsemble(file_path=model_path, format='ScikitLearn')

The Scikit-Learn project is described here:
    http://scikit-learn.org/
"""

import re

import numpy as np

from .rt_ensemble import RTEnsemble

base_score_reg = re.compile("^base_score=(.+)$")
learning_rate_reg = re.compile("^learning_rate=(.+)$")
tree_reg = re.compile("^booster\[(\d+)\]")
node_reg = re.compile("(\d+):\[f(\d+)<=(.*)\]")
leaf_reg = re.compile("(\d+):leaf=(.+)$")


class ProxyScikitLearn(object):
    """
    Class providing the implementation for loading/storing a Scikit-Learn model
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
        n_trees, n_nodes = ProxyScikitLearn._count_nodes(file_path)
        # Initialize the model and allocate the needed space
        # given the shape and size of the ensemble
        model.initialize(n_trees, n_nodes)

        root_node = 0
        num_nodes = 0
        learning_rate = 1
        queue = list()
        with open(file_path, 'r') as f:
            for line in f:

                match = base_score_reg.match(line)
                if match:
                    model.base_score = float(match.group(1))
                    continue

                match = learning_rate_reg.match(line)
                if match:
                    model.learning_rate = float(match.group(1))
                    continue

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
                    node_id = int(match_node.group(1).strip()) + root_node
                    feature_id = int(match_node.group(2).strip())
                    threshold = float(match_node.group(3).strip())

                    model.trees_nodes_feature[node_id] = feature_id
                    model.trees_nodes_value[node_id] = threshold

                match_leaf = leaf_reg.search(line)
                if match_leaf:
                    node_id = int(match_leaf.group(1).strip()) + root_node
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

    @staticmethod
    def export_scikit_model(model, file_path):
        if not hasattr(model, 'estimators_'):
            raise TypeError("Only ensemble-based models are supported!")

        if not hasattr(model, 'init_'):
            raise TypeError("Base score missing!")
        if hasattr(model.init_, "quantile"):
            base_score = model.init_.quantile
        elif hasattr(model.init_, "mean"):
            base_score = model.init_.mean
        else:
            raise TypeError("Base score unknown!")


        with open(file_path, 'w') as writer:
            if not hasattr(model, 'init_'):
                raise TypeError("Base score missing!")
            writer.write("base_score=%f\n" % base_score)
            writer.write("learning_rate=%f\n" % model.learning_rate)
            for tree_id, tree in enumerate(model.estimators_.flatten()):
                ProxyScikitLearn._export_tree(writer, tree, tree_id)

    @staticmethod
    def _export_tree(writer, tree, tree_id=0):
        from sklearn.tree import _tree

        if not hasattr(tree, 'tree_'):
            raise TypeError("Only tree-based models are supported!")

        tree_ = tree.tree_
        feature_name = ["f%d" % i for i in np.unique(tree_.feature) if
                        i != _tree.TREE_UNDEFINED]
        writer.write("booster[%d] [%s]:\n" % (tree_id, ' '.join(feature_name)))

        def recurse(node, depth):
            indent = '\t' * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = "f%d" % tree_.feature[node]
                threshold = tree_.threshold[node]
                writer.write("%s%d:[%s<=%f]\n" % (indent, node, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                # print "%s%d:[%s>%f]" % (indent, node, name, threshold)
                recurse(tree_.children_right[node], depth + 1)
            else:
                if tree_.value[node].size > 1:
                    leaf_value = "c%d" % np.argmax(tree_.value[node])
                else:
                    leaf_value = tree_.value[node].flatten()[0]
                writer.write("%s%d:leaf=%s\n" % (indent, node, leaf_value))

        recurse(0, 1)
