"""
Optimized scoring of RankEval.

Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
License: <TO DEFINE>
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
@cython.nonecheck(False)
def basic_scoring(model, X):

    cdef np.intp_t n_instances = X.shape[0]
    cdef np.intp_t n_trees = model.n_trees

    cdef float[:, :] X_view = X
    cdef float[:] y = np.zeros(n_instances, dtype=np.float32)

    cdef short[:] trees_root = model.trees_root
    cdef float[:] trees_weight = model.trees_weight
    cdef short[:] trees_nodes_feature = model.trees_nodes_feature
    cdef float[:] trees_nodes_value = model.trees_nodes_value
    cdef short[:] trees_left_child = model.trees_left_child
    cdef short[:] trees_right_child  = model.trees_right_child

    cdef float predicted_score
    cdef np.intp_t idx_tree, idx_instance
    with nogil, parallel():
        for idx_instance in prange(n_instances):
            for idx_tree in xrange(n_trees):
                predicted_score = _score_single_instance_single_tree(
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

                y[idx_instance] += predicted_score * trees_weight[idx_tree]
    return np.asarray(y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def detailed_scoring(model, X):

    cdef np.intp_t n_instances = X.shape[0]
    cdef np.intp_t n_trees = model.n_trees

    cdef float[:, :] X_view = X
    cdef float[:, :] partial_y = np.zeros((X.shape[0], model.n_trees), dtype=np.float32)

    cdef short[:] trees_root = model.trees_root
    cdef float[:] trees_weight = model.trees_weight
    cdef short[:] trees_nodes_feature = model.trees_nodes_feature
    cdef float[:] trees_nodes_value = model.trees_nodes_value
    cdef short[:] trees_left_child = model.trees_left_child
    cdef short[:] trees_right_child  = model.trees_right_child

    cdef float predicted_score
    cdef np.intp_t idx_tree, idx_instance
    with nogil, parallel():
        for idx_tree in prange(n_trees):
            for idx_instance in xrange(n_instances):
                predicted_score = _score_single_instance_single_tree(
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

                partial_y[idx_instance, idx_tree] = predicted_score * trees_weight[idx_tree]
    return np.asarray(partial_y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float _score_single_instance_single_tree(float[:,:] X,
                                              np.intp_t idx_instance,
                                              np.intp_t idx_tree,
                                              short[:] trees_root,
                                              float[:] trees_weight,
                                              short[:] trees_nodes_feature,
                                              float[:] trees_nodes_value,
                                              short[:] trees_left_child,
                                              short[:] trees_right_child) nogil:

    # Check the usage of np.intp_t in plave of np.int16_t
    cdef short cur_node = trees_root[idx_tree]
    cdef short feature_idx
    cdef float feature_threshold
    while trees_left_child[cur_node] != -1 and trees_right_child[cur_node] != -1:
        feature_idx = trees_nodes_feature[cur_node]
        feature_threshold = trees_nodes_value[cur_node]
        if X[idx_instance, feature_idx] < feature_threshold:
            cur_node = trees_left_child[cur_node]
        else:
            cur_node = trees_right_child[cur_node]
    return trees_nodes_value[cur_node]