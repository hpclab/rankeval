# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>, 
#          Claudio Lucchese <claudio.lucchese@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This package implements feature importance analysis.
"""

import numpy as np
from collections import deque
import xarray as xr

from ..model import RTEnsemble
from ..metrics import MSE, RMSE

from ._efficient_feature import eff_feature_importance, \
    eff_feature_importance_tree


def feature_importance(model, dataset, metric=None):
    """
    This method computes the feature importance relative to the given model
    and dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset used to evaluate the model (typically the one used to train
        the model).
    model : RTEnsemble
        The model whose features we want to evaluate.
    metric : rankeval.metrics.Metric
        The metric to use for compute the feature gain at each split node.
        The default metric is the Root Mean Squared Error (MSE).

    Returns
    -------
    feature_importance : xarray.DataArray
        A DataArray containing the feature importance scores, one for each
        feature of the given model scored on the given dataset. Two main
        information are stored in the DataArray:
            - feature_importance: A vector of importance values, one for each
                feature in the given model. The importance values reported are
                the sum of the improvements, in terms of MSE, of each feature,
                evaluated on the given dataset. The improvements are computed as
                the delta MSE before a split node and after, evaluating how much
                the MSE is improved as a result of the split.
            - feature_count: A vector of count values, one for each feature in
                the given model. The count values reported highlights the number
                of times each feature is used in a split node, i.e., to improve
                the MSE.
    """
    if metric is None:
        metric = MSE()

    if isinstance(metric, RMSE) or isinstance(metric, MSE):
        feature_imp, feature_count = eff_feature_importance(model, dataset)
        if isinstance(metric, RMSE):
            feature_imp[0] = np.sqrt(feature_imp[0])
    else:
        # initialize features importance
        feature_imp = np.zeros(dataset.n_features, dtype=np.float32)

        # initialize features count
        feature_count = np.zeros(dataset.n_features, dtype=np.uint16)

        # default scores on the root node of the first tree
        y_pred = np.zeros(dataset.n_instances)

        # iterate trees of the model
        for tree_id in np.arange(model.n_trees):
            _feature_importance_tree(model, dataset, tree_id, y_pred, metric,
                                     feature_imp, feature_count)

    performance = xr.DataArray([feature_imp, feature_count],
                               name='Feature Importance Analysis',
                               coords=[['importance', 'count'],
                                       np.arange(dataset.n_features, dtype=np.uint16)],
                               dims=['type', 'feature'])

    return performance


def _feature_importance_tree(model, dataset, tree_id, y_pred, metric,
                             feature_imp, feature_count):
    """
    This method computes the feature importance relative to a single tree of
    the given model.

    Parameters
    ----------
    dataset : Dataset
        The dataset used to evaluate the model (typically the one used to train
        the model).
    model : RTEnsemble
        The model whose features we want to evaluate.
    tree_id: int
        Id of the root of the tree to be evaluated.
    y_pred : numpy.array
        Current instance predictions.
    metric : rankeval.metrics.Metric
        The metric to use for compute the feature gain at each split node.
        The default metric is the Root Mean Squared Error (MSE).
    feature_imp : numpy.array
        The feature importance array, to be updated with the evaluation of the
        current tree.
    feature_ : numpy.array
        The feature count array, to be updated with the evaluation of the
        current tree.

    Returns
    -------
    y_pred_tree : numpy.array
        A vector of delta instance scores relative to the current tree.
    """
    if isinstance(metric, RMSE) or isinstance(metric, MSE):
        y_pred_tree = eff_feature_importance_tree(
            model, dataset, tree_id, y_pred, feature_imp, feature_count)
        if isinstance(metric, RMSE):
            map(np.math.sqrt, feature_imp)
        return y_pred_tree

    # The residual scores to fit
    y_target = dataset.y - y_pred

    # Set the default y_pred vector to the mean residual score
    y_pred_tree = np.full(dataset.n_instances, fill_value=y_target.mean())

    # queue with node_id to be visited and list of documents in the node
    root_node_id = model.trees_root[tree_id]
    deq = deque([(root_node_id, 0, None)])

    # we score the tree on a level basis, so that the feature_gain of each split
    # node is computed accordingly to the predictions of the parent level
    # (i.e., all the nodes in the tree with a depth < node_depth)

    # depth 0 prediction correspond to the mean residual score
    y_pred_tree_depth = [y_pred_tree]

    # support array for storing the local modification made by a split node
    y_pred_mod = np.empty(dataset.n_instances, dtype=np.float32)

    while len(deq) > 0:
        # current node info
        node_id, depth, doc_list = deq.popleft()
        feature_id = model.trees_nodes_feature[node_id]
        threshold = model.trees_nodes_value[node_id]

        feature_count[feature_id] += 1

        # this is good only for the internal nodes (non root)
        if doc_list is not None:
            left_docs = \
                doc_list[dataset.X[doc_list, feature_id] <= threshold]
            right_docs = \
                doc_list[dataset.X[doc_list, feature_id] > threshold]
        else:
            left_docs = np.where(dataset.X[:, feature_id] <= threshold)[0]
            right_docs = np.where(dataset.X[:, feature_id] > threshold)[0]

        # copy parent predictions inside y_pred_mod
        y_pred_mod[:] = y_pred_tree_depth[depth]

        # before updating leaves
        # dataset_y = y_target + y_pred
        pre_slit_metric, _ = metric.eval(dataset, y_pred_mod + y_pred)
        # ((y_target[doc_list] - y_pred_mod[doc_list]) ** 2.0).sum()

        # update y_pred including the weight of the current tree
        if left_docs.size > 0:
            y_pred_mod[left_docs] = y_target[left_docs].mean()
        if right_docs.size > 0:
            y_pred_mod[right_docs] = y_target[right_docs].mean()

        # update the y_pred of the next level
        if depth + 1 > len(y_pred_tree_depth) - 1:
            y_pred_tree_depth.append(y_pred_tree_depth[depth].copy())
        y_pred_tree_depth[depth + 1][doc_list] = y_pred_mod[doc_list]

        # after updating leaves
        post_split_metric, _ = metric.eval(dataset, y_pred_mod + y_pred)

        # compute the new metric score
        delta_metric = pre_slit_metric - post_split_metric

        # update feature importance
        feature_imp[feature_id] += delta_metric

        # if children are not leaves, add in the queue of the nodes to visit
        if not model.is_leaf_node(model.trees_left_child[node_id]):
            deq.append((model.trees_left_child[node_id], depth+1, left_docs))
        if not model.is_leaf_node(model.trees_right_child[node_id]):
            deq.append((model.trees_right_child[node_id], depth+1, right_docs))

    # update the leaves output including the tree weight
    y_pred_tree_depth[-1] *= model.trees_weight[tree_id]

    # update y_pred with the predictions made by the last level of the tree
    y_pred += y_pred_tree_depth[-1]

    return y_pred_tree_depth[-1]
