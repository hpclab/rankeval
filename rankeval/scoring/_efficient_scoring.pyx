# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Optimized scoring of RankEval.
"""

import cython
cimport cython

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
def basic_scoring(model, X):

    cdef np.intp_t n_instances = X.shape[0]
    cdef np.intp_t n_trees = model.n_trees

    cdef float[:, :] X_view = X
    y = np.zeros(n_instances, dtype=np.float32)
    cdef float[:] y_view = y

    cdef int[:] trees_root = model.trees_root
    cdef float[:] trees_weight = model.trees_weight
    cdef short[:] trees_nodes_feature = model.trees_nodes_feature
    cdef float[:] trees_nodes_value = model.trees_nodes_value
    cdef int[:] trees_left_child = model.trees_left_child
    cdef int[:] trees_right_child  = model.trees_right_child

    cdef int leaf_node
    cdef np.intp_t idx_tree, idx_instance
    with nogil, parallel():
        for idx_instance in prange(n_instances):
            for idx_tree in xrange(n_trees):
                leaf_node = _score_single_instance_single_tree(
                    X_view,
                    idx_instance,
                    idx_tree,
                    trees_root,
                    trees_weight,
                    trees_nodes_feature,
                    trees_nodes_value,
                    trees_left_child,
                    trees_right_child
                )

                y_view[idx_instance] += \
                    trees_nodes_value[leaf_node] * trees_weight[idx_tree]
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def detailed_scoring(model, X):

    cdef np.intp_t n_instances = X.shape[0]
    cdef np.intp_t n_trees = model.n_trees

    cdef float[:, :] X_view = X
    y_leaves = np.zeros((X.shape[0], model.n_trees), dtype=np.int32)
    cdef int[:, :] y_leaves_view = y_leaves

    partial_y = np.zeros((X.shape[0], model.n_trees), dtype=np.float32)
    cdef float[:, :] partial_y_view = partial_y

    cdef int[:] trees_root = model.trees_root
    cdef float[:] trees_weight = model.trees_weight
    cdef short[:] trees_nodes_feature = model.trees_nodes_feature
    cdef float[:] trees_nodes_value = model.trees_nodes_value
    cdef int[:] trees_left_child = model.trees_left_child
    cdef int[:] trees_right_child  = model.trees_right_child

    cdef int leaf_node
    cdef np.intp_t idx_tree, idx_instance
    with nogil, parallel():
        for idx_tree in prange(n_trees):
            for idx_instance in xrange(n_instances):
                leaf_node = _score_single_instance_single_tree(
                    X_view,
                    idx_instance,
                    idx_tree,
                    trees_root,
                    trees_weight,
                    trees_nodes_feature,
                    trees_nodes_value,
                    trees_left_child,
                    trees_right_child
                )

                y_leaves_view[idx_instance, idx_tree] = leaf_node
                partial_y_view[idx_instance, idx_tree] = \
                    trees_nodes_value[leaf_node] * trees_weight[idx_tree]

    return np.asarray(y_leaves), np.asarray(partial_y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _score_single_instance_single_tree(float[:,:] X,
                                            np.intp_t idx_instance,
                                            np.intp_t idx_tree,
                                            int[:] trees_root,
                                            float[:] trees_weight,
                                            short[:] trees_nodes_feature,
                                            float[:] trees_nodes_value,
                                            int[:] trees_left_child,
                                            int[:] trees_right_child) nogil:

    # Check the usage of np.intp_t in plave of np.int16_t
    cdef int cur_node = trees_root[idx_tree]
    cdef short feature_idx
    cdef float feature_threshold
    while trees_left_child[cur_node] != -1 and trees_right_child[cur_node] != -1:
        feature_idx = trees_nodes_feature[cur_node]
        feature_threshold = trees_nodes_value[cur_node]
        if X[idx_instance, feature_idx] <= feature_threshold:
            cur_node = trees_left_child[cur_node]
        else:
            cur_node = trees_right_child[cur_node]
    return cur_node