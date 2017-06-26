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
import xarray as xr
from collections import deque


def feature_importance(dataset, model):
    """
    This method computes the feature importance relative to the given model and dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset used to evaluate the model (typically the one used to train the model).
    model : RTEnsemble
        The model whose features we want to evaluate.

    Returns
    -------
    feature_importance : numpy.array
        A vector of importance values, one for each feature in the given model.
    """

    # initialize features importance
    fx_imp = np.zeros(dataset.X.shape[1])

    # default scores
    y_pred = np.full( dataset.X.shape[0], fill_value=dataset.y.mean() )

    # iterate trees of the model
    for tree_id in xrange( len(model.trees_root) ):
    	fx_imp_tree, y_pred_new = _fx_tree_imp(dataset, model, tree_id, y_pred)
    	y_pred = model.trees_weight[tree_id] * y_pred_new
    	fx_imp += fx_imp_tree

    	print fx_imp

    

    print model.score(dataset).y_pred
    print y_pred

    return fx_imp

def _fx_tree_imp(dataset, model, tree_id, y_pred):
    """
    This method computes the feature importance relative to a single tree of the given model.

    Parameters
    ----------
    dataset : Dataset
        The dataset used to evaluate the model (typically the one used to train the model).
    model : RTEnsemble
        The model whose features we want to evaluate.
    tree_id: int
    	Id of the root of the tree to be evaluated.
    y_pred : numpy.array
    	Current instance predictions.

    Returns
    -------
    feature_importance : numpy.array
        A vector of importance values, one for each feature in the given model.
    y_pred_new : numpy.array
        A vector of updated instance scores.
    """
    # init feature importances
    fx_imp = np.zeros(dataset.X.shape[1])

    # compute initial MSE
    previous_mse = ( (dataset.y - y_pred)**2.0 ).mean()

    # queue with node_id to be visited and list of documents in the node
    root_id = model.trees_root[tree_id]
    nodes = deque( [(root_id,None)] )

    while len(nodes)>0:
    	# current node info
    	node_id, doc_list = nodes.popleft()
    	feature_id = model.trees_nodes_feature[node_id]
    	threshold = model.trees_nodes_value[node_id]

    	# this is good only for the root
    	if doc_list is not None:
    		left_docs  = doc_list[ dataset.X[doc_list,feature_id] <= threshold ]
    		right_docs = doc_list[ dataset.X[doc_list,feature_id] > threshold ]
    	else:
    		left_docs  = np.where( dataset.X[:,feature_id] <= threshold ) [0]
    		right_docs = np.where( dataset.X[:,feature_id] > threshold ) [0]

    	left_pred  = dataset.y[left_docs].mean()
    	right_pred = dataset.y[right_docs].mean()

	    # update y_pred
    	y_pred[left_docs]  = left_pred
    	y_pred[right_docs] = right_pred

    	# compute new_mse
    	new_mse = ( (dataset.y - y_pred)**2.0 ).mean()

    	# update feature importance
    	fx_imp[feature_id] += previous_mse - new_mse

    	# if children are not leaves
    	if not model.is_leaf_node( model.trees_left_child[node_id] ):
    		nodes.append( (model.trees_left_child[node_id],left_docs) )
    	if not model.is_leaf_node( model.trees_right_child[node_id] ):
    		nodes.append( (model.trees_right_child[node_id],right_docs) )

    	previous_mse = new_mse


    return fx_imp, y_pred
