# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors:  Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


"""
This package implements several topological analysis focused on the
topological characteristics of ensemble-based LtR models. These
functionalities can be applied to several models,
so as to have a direct comparison of the shape of the resulting
forests (e.g., trained by different LtR algorithms).
"""

import numpy as np
import scipy.stats

from ..model import RTEnsemble
from ._efficient_topological import efficient_topological_analysis


def topological_analysis(model, include_leaves=True):
    """
    This method implements the topological analysis of a ensemble-based
    LtR model. Given a model, it studies the shape of each tree composing
    the model and return several information useful for having insights
    about the shape of the trees, their completeness (level by level) as
    well as min/max/mean height and the fraction of trees having a specific
    node (where each node is identified by a pair of coordinates row-col,
    with row highlighting the depth and col the column with respect to a
    full binary tree).

    Parameters
    ----------
    model : RTEnsemble
        The model to analyze
    include_leaves : bool
        Whether the leaves has to be included in the analysis or not

    Returns
    -------
    object : TopologicalAnalysisResult
        The topological result, to use for retrieving several information
    """
    return TopologicalAnalysisResult(model, include_leaves)


class TopologicalAnalysisResult(object):
    """
    This class is used to return the topological analysis made on the model.
    Several low-level information are stored in this class, and then
    re-elaborated to provide high-level analysis.
    """

    def __init__(self, model, include_leaves):
        """
        Analyze the model in a topological perspective

        Parameters
        ----------
        model : RTEnsemble
            the model to analyze from the topological perspective
        include_leaves : bool
            Whether the leaves has to be included in the analysis or not

        Attributes
        ----------
        model : RTEnsemble
            The model analyzed
        height_trees : numpy array
            The ordered height of each trees composing the ensemble
        topology : scipy.sparse.csr_matrix
            The matrix used to store low-level information related to the
            aggregated shape of the trees. Each matrix cell identifies a
            tree node with a pair of coordinates row-col, with row
            highlighting the depth and col the column with respect
            to a full binary tree.
        """
        self.model = model
        self.topology, self.height_trees = efficient_topological_analysis(model, include_leaves)

    def describe_tree_height(self):
        """
        Computes several descriptive statistics of the height of the trees.

        Returns
        -------
        nobs : int
           Number of trees
        minmax: tuple of ndarrays or floats
           Minimum and maximum height of trees
        mean : ndarray or float
           Arithmetic mean of tree heights.
        variance : ndarray or float
           Unbiased variance of the tree heights.
           denominator is number of trees minus one.
        skewness : ndarray or float
           Skewness, based on moment calculations with denominator equal to
           the number of trees, i.e. no degrees of freedom correction.
        kurtosis : ndarray or float
           Kurtosis (Fisher).  The kurtosis is normalized so that it is
           zero for the normal distribution.  No degrees of freedom are used.
        """
        return scipy.stats.describe(self.height_trees)

    def avg_tree_shape(self):
        """
        Computes the fraction of trees having each node with respect to a
        full binary tree. The fraction is obtained by normalizing the count
        by the number of trees composing the ensemble model.

        Returns
        -------
        fractions : scipy.sparse.csr_matrix
            Sparse matrix with the same shape of the topology matrix, where
            each matrix cell identifies a tree node by a pair of coordinates
            row-col, with row highlighting the depth and col the column with
            respect to a full binary tree. Each cell value highlights how many
            trees have the specific node, normalized by the number of trees.
        """
        return self.topology / self.model.n_trees

    def fullness_per_level(self):
        """
        Computes the normalized number of trees with full level i, for each
        level of a full binary tree. The normalization is done by the number
        of trees.

        Returns
        -------
        fullness : np.array
            An array long as the maximum height of a tree in the ensemble, and
            where the j-th cell highlight how much the j-th level of the trees
            is full (normalized by the number of trees).
        """
        # Row-sums are directly supported, and the structure of the CSR format means that
        # the difference between successive values in the indptr array correspond exactly
        # to the number of nonzero elements in each row.
        sums = self.topology.sum(axis=1).A1
        counts = np.diff(self.topology.indptr)
        return sums / counts / self.model.n_trees
