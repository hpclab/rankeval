import cython
cimport cython

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np

import scipy as sc
import scipy.sparse

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
def efficient_topological_analysis(model, include_leaves=True):

    cdef np.intp_t n_trees = model.n_trees
    cdef np.intp_t n_nodes = model.n_nodes

    cdef int[:] trees_root = model.trees_root
    cdef int[:] trees_left_child = model.trees_left_child
    cdef int[:] trees_right_child  = model.trees_right_child

    node_indices = np.zeros(model.n_nodes, dtype=np.uint64)
    cdef unsigned long long[:] node_indices_view = node_indices
    cdef unsigned int[:] height_trees = np.zeros(model.n_trees, dtype=np.uint32)

    cdef bint c_include_leaves = include_leaves

    cdef np.intp_t idx_tree
    cdef int idx_last_node
    with nogil, parallel():
        for idx_tree in prange(n_trees):
            idx_last_node = trees_root[idx_tree+1] if idx_tree < n_trees-1 else n_nodes
            height_trees[idx_tree] = _compute_node_indices(idx_tree,
                                                           trees_root,
                                                           trees_left_child,
                                                           trees_right_child,
                                                           node_indices_view,
                                                           idx_last_node,
                                                           c_include_leaves)

    # Computes unique indices and counts the occurrences of each index (aggregate)
    unique_counts = np.unique(node_indices, return_counts=True)

    cdef unsigned long long[:] data_indices_view = unique_counts[0]
    cdef long[:] counts_view = unique_counts[1]

    # overwrite counts of 0-values since they should identify only the
    # root nodes but include also the leaves when include_leaves=False)
    counts_view[0] = n_trees

    cdef np.intp_t data_indices_size = data_indices_view.size

    # indices in a sparse matrix representation
    cdef unsigned long long[:] row_ind = np.zeros(data_indices_size, dtype=np.uint64)
    cdef unsigned long long[:] col_ind = np.zeros(data_indices_size, dtype=np.uint64)

    cdef np.intp_t idx_data
    cdef int exp
    with nogil, parallel():
        for idx_data in prange(data_indices_size):
            row_ind[idx_data] = most_significant_bit(data_indices_view[idx_data] + 1)
            col_ind[idx_data] = data_indices_view[idx_data] + 1 - 2**row_ind[idx_data]

    return sc.sparse.csr_matrix((counts_view, (row_ind, col_ind)), dtype=np.float32), np.asarray(height_trees)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _compute_node_indices(np.intp_t idx_tree,
                               int[:] trees_root,
                               int[:] trees_left_child,
                               int[:] trees_right_child,
                               unsigned long long[:] node_indices,
                               int idx_last_node,
                               bint include_leaves) nogil:

    cdef int cur_node = trees_root[idx_tree]
    cdef unsigned long long left_value, right_value, max_index = 0
    while cur_node < idx_last_node:
        if _is_leaf_node(cur_node, trees_left_child, trees_right_child):
            if not include_leaves:
                node_indices[cur_node] = 0
        else:
            left_value = 2 * node_indices[cur_node] + 1
            right_value = 2 * node_indices[cur_node] + 2
            node_indices[trees_left_child[cur_node]] = left_value
            node_indices[trees_right_child[cur_node]] = right_value
            max_index = max(max_index, left_value)
            max_index = max(max_index, right_value)
        cur_node += 1

    cdef int height = most_significant_bit(max_index + 1)
    return height

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _is_leaf_node(int idx_node,
                               int[:] trees_left_child,
                               int[:] trees_right_child) nogil:
    return trees_left_child[idx_node] == -1 and trees_right_child[idx_node] == -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int most_significant_bit(long long v) nogil:

    cdef long long *b = [0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000, 0xFFFFFFFF00000000]
    cdef unsigned int *S = [1, 2, 4, 8, 16, 32]

    # result of log2(v) will go here
    cdef unsigned int r = 0
    # unroll for speed...
    cdef int i = 5
    while i >= 0:
        if (v & b[i]):
            v >>= S[i];
            r |= S[i];
        i -= 1
    
    return r