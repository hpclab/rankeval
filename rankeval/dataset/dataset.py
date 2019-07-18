# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This module implements the generic class for loading/dumping a dataset from/to
file.
"""
import numpy as np
import numbers

from .svmlight_format import load_svmlight_file, dump_svmlight_file


class Dataset(object):
    """
    This class describe the dataset object, with its utility and features

    Attributes
    ----------
    X : numpy 2d array of float
        It is a dense numpy matrix of shape (n_instances, n_features),
    y : numpy 1d array of float
        It is a ndarray of shape (n_instances,) with the gold label
    query_ids : numpy 1d array of int
        It is a ndarray of shape(n_queries,)
    query_offsets : numpy 1d array of int
        It is a ndarray of shape(n_queries+1, ) with the start and end offsets
        of each query. In particular. the i-th query has indices ranging in
        [ query_offsets[i], query_offsets[i+1] ), with the latter excluded.
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
        This module implements the generic class for loading/dumping a dataset
        from/to file.

        Parameters
        ----------
        X : numpy.ndarray
            The matrix with feature values
        y : numpy.array
            The vector with label values
        query_ids : numpy.array
            The vector with the query_id for each sample.
        """
        if query_ids.size != X.shape[0]:
            raise Exception("query_ids has wrong size. Expected %s but got %s" % (X.shape[0], query_ids.size))

        # convert from query_ids per sample to query offset
        self.query_ids, self.query_offsets = \
            np.unique(query_ids, return_index=True)

        # resort the arrays per offset (if the file does not contains qids in
        # order, the np.unique will return qids with a different ordering...
        idx_sort = np.argsort(self.query_offsets)
        self.query_ids = self.query_ids[idx_sort]
        self.query_offsets = self.query_offsets[idx_sort]

        self.query_offsets = np.append(self.query_offsets, query_ids.size)

        self.X, self.y = X, y
        self.name = "Dataset %s" % (self.X.shape,)
        if name is not None:
            self.name = name

        self.n_instances = self.y.size
        self.n_features = self.X.shape[1]
        self.n_queries = self.query_ids.size

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
            The format of the dataset file to load (actually supported is only
            "svmlight" format)

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
        return Dataset(self.X[:, features].copy(),
                       self.y,
                       self.get_qids_dataset(),
                       name=self.name)

    def dump(self, f, format):
        """
        This method implements the writing of a previously loaded dataset
        according to the given format on file

        Parameters
        ----------
        f : path
            The file path where to store the dataset
        format : str
            The format to use for dumping the dataset on file (actually
            supported is only "svmlight" format)
        """
        # we need to unroll the query_ids and query_offsets.
        # They are represented compact: they report only the query ids and the
        # offsets where each query starts and ends.
        query_ids = np.ndarray(self.n_instances, dtype=np.int32)
        for qid, start_offset, end_offset in self.query_iterator():
            for idx in np.arange(start_offset, end_offset):
                query_ids[idx] = qid

        if format == "svmlight":
            dump_svmlight_file(self.X, self.y, f, query_ids)
        else:
            raise TypeError("Dataset format %s is not yet supported!" % format)

    def split(self, train_size, vali_size=0, random_state=None):
        """
        This method splits the dataset into train/validation/test partition.
        It shuffle the query ids before partitioning. If vali_size=0, it means
        the method will not create a validation set, thus returning only
        train and test sets. Otherwise it will return train/vali/test sets.

        Parameters
        ----------
        train_size : float
            The ratio of query ids in the training set. It should be between
            0 and 1.
        vali_size : float
            The ratio of query ids in the validation set. It should be between
            0 and 1. 0 means no validation to be created.
        random_state : int
            If int, random_state is the seed used by the random number
            generator. If RandomState instance, random_state is the random
            number generator. If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        (train, vali, test) datasets : tuple of rankeval.dataset.Dataset
            The resulting datasets with the given fraction of query ids in each
            partition.
        """

        if train_size < 0 or train_size > 1 or (train_size + vali_size) > 1:
            raise Exception("train and/or validation sizes are not correct!")

        train_qn = int(round(train_size * self.n_queries))
        vali_qn = int(round(vali_size * self.n_queries))
        test_qn = self.n_queries - train_qn - vali_qn

        qid_map = np.ndarray(self.n_instances, dtype=np.uint32)
        for qid, start_offset, end_offset in self.query_iterator():
            for idx in np.arange(start_offset, end_offset):
                qid_map[idx] = qid

        # add queries shuffling
        rng = Dataset._check_random_state(random_state)
        qids_permutation = rng.permutation(self.query_ids)

        train_qid = qids_permutation[:train_qn]
        vali_qid = qids_permutation[train_qn:train_qn+vali_qn]
        test_qid = qids_permutation[-test_qn:]

        train_mask = np.in1d(qid_map, train_qid)
        vali_mask = np.in1d(qid_map, vali_qid)
        test_mask = np.in1d(qid_map, test_qid)

        train_dataset = Dataset(self.X[train_mask], self.y[train_mask],
                                qid_map[train_mask], name=self.name + ' Train')
        if vali_size:
            vali_dataset = Dataset(self.X[vali_mask], self.y[vali_mask],
                                   qid_map[vali_mask], name=self.name + ' Vali')
        test_dataset = Dataset(self.X[test_mask], self.y[test_mask],
                               qid_map[test_mask], name=self.name + ' Test')

        if not vali_size:
            return train_dataset, test_dataset
        else:
            return train_dataset, vali_dataset, test_dataset

    def subset(self, query_ids, name=None):
        """
        This method return a subset of the dataset according to the query_ids
        parameter.

        Parameters
        ----------
        query_ids : numpy 1d array of int
            It is a ndarray with the query_ids to select
        name : str
            The name to give to the dataset

        Returns
        -------
        datasets : rankeval.dataset.Dataset
            The resulting dataset with only the query_ids requested
        """
        qid_map = self.get_qids_dataset()
        mask = np.in1d(qid_map, query_ids)

        return Dataset(self.X[mask], self.y[mask],
                       qid_map[mask], name=name)

    def query_iterator(self):
        """
        This method implements and iterator over the offsets of the query_ids
        in the dataset.

        Returns
        -------
        offsets : tuple of (int, int, int)
            The query_id and the row index of instances belonging to the query.
            The two indices represent (start, end) offsets.

        """
        for i in np.arange(self.n_queries):
            yield self.query_ids[i], \
                  self.query_offsets[i], self.query_offsets[i+1]

    def get_query_sizes(self):
        """
        This method return the size of each query set.

        Returns
        -------
        sizes : numpy 1d array of int
            It is a ndarray of shape (n_queries,)
        """
        return np.ediff1d(self.query_offsets)

    def get_qids_dataset(self, dtype=np.int32):
        """
        This method returns the query ids array in linear representation, i.e.,
        with the qid of each instance. Useful for creating a new dataset
        starting from a different one.

        Returns
        -------
        query_ids : numpy 1d array
            It is a ndarray of shape (n_instances,)
        """
        query_ids = np.empty(shape=self.n_instances, dtype=dtype)
        for qid, start_offset, end_offset in self.query_iterator():
            query_ids[start_offset:end_offset] = qid
        return query_ids

    @staticmethod
    def _check_random_state(seed):
        """
        Turn seed into a np.random.RandomState instance (took for sklearn)

        Parameters
        ----------
        seed : None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with it.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        """
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)

    def __str__(self):
        return self.name

    def __hash__(self):
        return int( self.y[:100].sum() + self.X[:100,0].sum() )

    def __eq__(self, other):
        # use != instead of == because it is more efficient for sparse matrices:
        x_eq = not(self.X != other.X).any()
        return x_eq and (self.y == other.y).all() and \
               (self.query_ids == other.query_ids).all()

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
