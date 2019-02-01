# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Class providing the implementation for loading/storing a CatBoost model
from/to file.

The CatBoost project is described here:
    https://github.com/catboost/catboost

CatBoost allows to save the learned model in several formats (binary, coreml,
etc). Among them, we chose to adopts the Apple CoreML format for reading and
converting a model into the rankeval representation. It s possible to read  the
coreML representation using the coremltools python package. Once read it
provides all the structured information of the ensemble, with split nodes (both
features and thresholds), leaf values and tree structure. Not all the
information reported in the model are useful for the different analysis, thus
only the relevant parts are parsed.

NOTE: CatBoost trains oblivious trees, i.e., trees where at each level a single
condition is checked, independently from the which node we are currently working
on. Rankeval does not exploit oblivious trees, but instead it represent them
as normal decision trees Thus the same condition will appear on all the nodes
of a single level of a tree. The reason behind this choice is to fasten the
development of the CatBoost proxy, allowing to analyze it without focusing too
much on prediction time (that is not currently measured by rankeval).
"""

import numpy as np
import logging

from .rt_ensemble import RTEnsemble


class ProxyCatBoost(object):
    """
    Class providing the implementation for loading/storing a ProxyCatBoost model
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

        try:
            import coremltools
        except ImportError:
            logging.error('Missing coremltools package!.')
            return

        coreml_model = coremltools.models.model.MLModel(file_path)

        n_trees, n_nodes = ProxyCatBoost._count_nodes(coreml_model)
        # Initialize the model and allocate the needed space
        # given the shape and size of the ensemble
        model.initialize(n_trees, n_nodes)

        n_nodes_per_tree = int(n_nodes / n_trees)

        nodes = coreml_model.get_spec().treeEnsembleRegressor.treeEnsemble.nodes
        behaviors = coremltools.proto.TreeEnsemble_pb2.TreeEnsembleParameters.\
            TreeNode.TreeNodeBehavior

        for node in nodes:
            tree_offset = node.treeId * n_nodes_per_tree
            node_id_remap = ProxyCatBoost.remap_nodeId(node.nodeId,
                                                       n_nodes_per_tree)
            node_id_off = node_id_remap + tree_offset

            if node_id_remap == 0:  # this is the root of a tree
                model.trees_root[node.treeId] = tree_offset
                model.trees_weight[node.treeId] = 1

            if node.nodeBehavior == behaviors.Value('LeafNode'):
                model.trees_nodes_value[node_id_off] = \
                    node.evaluationInfo[0].evaluationValue
            else:
                if node.nodeBehavior == behaviors.Value('BranchOnValueGreaterThan'):
                    # we need to flip the condition given we use "<="
                    left = node.falseChildNodeId
                    right = node.trueChildNodeId
                elif node.nodeBehavior == behaviors.Value('BranchOnValueLessThanEqual'):
                    right = node.falseChildNodeId
                    left = node.trueChildNodeId
                else:
                    raise AssertionError(
                        "Branching condition not supported. RankEval does not "
                        "support branching conditions different from "
                        "BranchOnValueGreaterThan or BranchOnValueLessThanEqual.")

                model.trees_nodes_value[node_id_off] = node.branchFeatureValue
                model.trees_nodes_feature[node_id_off] = node.branchFeatureIndex
                model.trees_left_child[node_id_off] = tree_offset +\
                    ProxyCatBoost.remap_nodeId(left, n_nodes_per_tree)
                model.trees_right_child[node_id_off] = tree_offset + \
                    ProxyCatBoost.remap_nodeId(right, n_nodes_per_tree)

    @staticmethod
    def remap_nodeId(nodeId, n_nodes_per_tree):
        return n_nodes_per_tree - 1 - nodeId

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
    def _count_nodes(coreml_model):
        """
        Count the total number of nodes (both split and leaf nodes)
        in the CoreML model.

        Parameters
        ----------
        coreml_model : CoreML model
            The CoreML model to load from

        Returns
        -------
        tuple(n_trees, n_nodes) : tuple(int, int)
            The total number of trees and nodes (both split and leaf nodes)
            in the model identified by file_path.
        """

        nodes = coreml_model.get_spec().treeEnsembleRegressor.treeEnsemble.nodes

        n_trees = np.max([node.treeId for node in nodes]) + 1

        n_nodes_trees = np.empty(n_trees, dtype=np.uint16)
        for node in nodes:
            n_nodes_trees[node.treeId] = node.nodeId

        # node_Id starts from 0, thus + 1
        n_nodes = np.sum(n_nodes_trees + 1)

        return n_trees, n_nodes