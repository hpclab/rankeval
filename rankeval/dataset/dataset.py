# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This module implements the generic class for loading/dumping a dataset from/to file.
"""
import numpy as np
import copy

from .svmlight_format import load_svmlight_file, dump_svmlight_file


class Dataset(object):
    """
    This class describe the dataset object, with its utility and features

    Attributes
    ----------
    X : numpy 2d array of float
        It is a dense numpy matrix of shape (n_samples, n_features),
    y : numpy 1d array of float
        It is a ndarray of shape (n_samples,) with the gold label
    query_ids : numpy 1d array of int
        It is a ndarray of shape(nsamples,)
    name : str
        The name to give to the dataset
    n_instances : int
        The number of instances in the dataset
    n_features : int
        The number of features in the dataset
    n_queries : int
        The number of queries in the dataset
    """

    def __init__(self, X, y, query_ids, name=None):
        """
        This module implements the generic class for loading/dumping a dataset from/to file.

        Parameters
        ----------
        X : numpy.ndarray
            The matrix with feature values
        y : numpy.array
            The vector with label values
        query_ids : numpy.array
            The vector with the query_id for each sample.
        """

        if len(query_ids) == X.shape[0]:
            # convert from query_ids per sample to query offset
            self.query_ids = np.append(np.unique(query_ids, return_index=True)[1],
                                       query_ids.size)
        else:
            self.query_ids = query_ids

        self.X, self.y = X, y
        self.name = "Dataset %s" % (self.X.shape,)
        if name is not None:
            self.name = name

        self.n_instances = len(self.y)
        self.n_features = self.X.shape[1]
        self.n_queries = len(self.query_ids) - 1

    @staticmethod
    def load(f, name=None, format="svmlight"):
        """
        This static method implements the loading of a dataset from file.

        Parameters
        ----------
        f : path
            The file name of the dataset to load
        name : str
            The name to be given to the current dataset
        format : str
            The format of the dataset file to load (actually supported is only "svmlight" format)

        Returns
        -------
        dataset : Dataset
            The dataset read from file
        """
        if format == "svmlight":
            X, y, query_ids = load_svmlight_file(f, query_id=True)
        else:
            raise TypeError("Dataset format %s is not yet supported!" % format)
        return Dataset(X, y, query_ids, name)

    def subset_features(self, features):
        """
        Create a new Dataset with only the features identified by the given
        features parameters (indices). It is useful for performing feature
        selection.

        Parameters
        ----------
        features : numpy array or list
            The indices of the features to select in the resulting dataset

        Returns
        -------
        dataset : rankeval.dataset.Dataset
            The resulting dataset with the given subset of features
        """
        new_dataset = copy.deepcopy(self)
        new_dataset.X = new_dataset.X[:, features]
        return new_dataset

    def dump(self, f, format):
        """
        This method implements the writing of a previously loaded dataset according to the given format on file

        Parameters
        ----------
        f : path
            The file path where to store the dataset
        format : str
            The format to use for dumping the dataset on file (actually supported is only "svmlight" format)
        """
        if len(self.query_ids) != self.X.shape[0]:
            # we need to unroll the query_ids (it is compacted: it reports only
            # the offset where a new query id starts)
            query_ids = np.ndarray(self.X.shape[0], dtype=np.float32)
            last_idx = 0
            for qid, qid_offset in enumerate(self.query_ids, start=1):
                for idx in np.arange(last_idx, qid_offset):
                    query_ids[idx] = qid
                last_idx = qid_offset
        else:
            query_ids = self.query_ids

        if format == "svmlight":
            dump_svmlight_file(self.X, self.y, f, query_ids)
        else:
            raise TypeError("Dataset format %s is not yet supported!" % format)

    def clear_X(self):
        """
        This method clears the space used by the dataset instance for storing X (the dataset features).
        This space is used only for scoring, thus it can be freed after.

        """
        del self.X
        self.X = None

    def query_offset_iterator(self):
        """
        This method implements and iterator over the offsets of the query_ids
        in the dataset.

        Returns
        -------
        offsets : tuple of (int, int)
            The row index of instances belonging to the same query.
            The two indices represent (start, end) offsets.

        """
        for i in np.arange(len(self.query_ids) - 1):
            yield self.query_ids[i], self.query_ids[i+1]

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return (self.X == other.X).all() and \
               (self.y == other.y).all() and \
               (self.query_ids == other.query_ids).all()

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)