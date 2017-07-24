"""
This file implements the feature importance analysis in an efficient way. The
limit is that the metric used to compute the gain for each split is hardcoded
in the source code and is the MSE.
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

cdef extern from "_efficient_feature_impl.h":
    void c_feature_importance(
        const float* X,
        const float* y,
        const int* trees_root,
        const float* trees_weight,
        const short* trees_nodes_feature,
        const float* trees_nodes_value,
        const int* trees_left_child,
        const int* trees_right_child,
        float* feature_imp,
        short* feature_count,
        const int n_instances,
        const int n_features,
        const int n_trees);

    void c_feature_importance_tree(
        const float* X,
        const float* y,
        const int* trees_root,
        const float* trees_weight,
        const short* trees_nodes_feature,
        const float* trees_nodes_value,
        const int* trees_left_child,
        const int* trees_right_child,
        const int tree_id,
        float* feature_imp,
        short* feature_count,
        const int n_instances,
        const int n_features,
        float* y_pred,
        float* y_pred_tree);

@cython.boundscheck(False)
@cython.wraparound(False)
def eff_feature_importance(model, dataset):

    # initialize features importance
    feature_imp = np.zeros(dataset.n_features, dtype=np.float32)

    # initialize features importance
    feature_count = np.zeros(dataset.n_features, dtype=np.uint16)

    c_feature_importance(
        <float*> np.PyArray_DATA(dataset.X),
        <float*> np.PyArray_DATA(dataset.y),
        <int*> np.PyArray_DATA(model.trees_root),
        <float*> np.PyArray_DATA(model.trees_weight),
        <short*> np.PyArray_DATA(model.trees_nodes_feature),
        <float*> np.PyArray_DATA(model.trees_nodes_value),
        <int*> np.PyArray_DATA(model.trees_left_child),
        <int*> np.PyArray_DATA(model.trees_right_child),
        <float*> np.PyArray_DATA(feature_imp),
        <short*> np.PyArray_DATA(feature_count),
        dataset.X.shape[0],
        dataset.X.shape[1],
        model.n_trees);

    return np.asarray(feature_imp, dtype=np.float32), \
           np.asarray(feature_count, dtype=np.uint16)

@cython.boundscheck(False)
@cython.wraparound(False)
def eff_feature_importance_tree(model, dataset, tree_id, y_pred,
                             feature_imp, feature_count):

    y_pred_tree = np.zeros(dataset.n_instances, dtype=np.float32);

    c_feature_importance_tree(
        <float*> np.PyArray_DATA(dataset.X),
        <float*> np.PyArray_DATA(dataset.y),
        <int*> np.PyArray_DATA(model.trees_root),
        <float*> np.PyArray_DATA(model.trees_weight),
        <short*> np.PyArray_DATA(model.trees_nodes_feature),
        <float*> np.PyArray_DATA(model.trees_nodes_value),
        <int*> np.PyArray_DATA(model.trees_left_child),
        <int*> np.PyArray_DATA(model.trees_right_child),
        tree_id,
        <float*> np.PyArray_DATA(feature_imp),
        <short*> np.PyArray_DATA(feature_count),
        dataset.X.shape[0],
        dataset.X.shape[1],
        <float*> np.PyArray_DATA(y_pred),
        <float*> np.PyArray_DATA(y_pred_tree));

    return np.asarray(y_pred_tree, dtype=np.float32)