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
import hashlib

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
            raise Exception(
                "query_ids has wrong size. Expected %s but got %s" % (
                X.shape[0], query_ids.size))

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
        self._hash_cached = None

    @staticmethod
    def load(f, name=None, format="svmlight"):
        """
        This static method implements the loading of a dataset from file.

        Parameters
        ----------
        f : {str, file-like, int}
            (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
            be uncompressed on the fly. If an integer is passed, it is assumed
            to be a file descriptor. A file-like or file descriptor will not be
            closed by this function. A file-like object must be opened in
            binary mode.
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

    def dump(self, f, format="svmlight"):
        """
        This method implements the writing of a previously loaded dataset
        according to the given format on file

        Parameters
        ----------
        f : {str, file-like, int}
            (Path to) a file to dump. If a path ends in ".gz" or ".bz2", it will
            be compressed on the fly. If an integer is passed, it is assumed
            to be a file descriptor. A file-like or file descriptor will not be
            closed by this function. A file-like object must be opened in
            text mode.
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
            partition. If qids_only=True, the methods yields only the query ids
            of each fold, without creating the dataset.
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
        vali_qid = qids_permutation[train_qn:train_qn + vali_qn]
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
        This method implements an iterator over the offsets of the query_ids
        in the dataset.

        Returns
        -------
        offsets : tuple of (int, int, int)
            The query_id and the row index of instances belonging to the query.
            The two indices represent (start, end) offsets.

        """
        for i in np.arange(self.n_queries):
            yield self.query_ids[i], \
                  self.query_offsets[i], self.query_offsets[i + 1]

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

    def get_query_offsets(self, query_id):
        """
        This method return the offsets (start, end) of a given query_id in the
        dataset. Useful for debugging/analyzing in details the behaviours of a
        given model on specific set of queries.

        Parameters
        ----------
        query_id: int
            The query to search in the dataset

        Returns
        -------
        offsets : tuple of (int, int)
            The query index of instances belonging to the query.
            The two indices represent (start, end) offsets.

        """
        idx = np.where(self.query_ids == query_id)[0]
        if idx.size == 0:
            raise LookupError("query_id {:d} is missing from the dataset") \
                .format(query_id)
        # Take first element
        idx = idx[0]
        return self.query_offsets[idx], self.query_offsets[idx + 1]

    def kfold(self, n_folds=5, qids_only=False, shuffle=True):
        """
        This method generates a k-fold splitting of the dataset, i.e., it splits
        the dataset in n_folds and provide train/vali/test splitting of data
        for each iteration (fold). Folds are rotated avoiding overlapping.
        Shuffle the queries by default before splitting.

        Parameters
        ----------
        n_folds: int
            The number of folds. Must be at least 2
        qids_only : bool
            Whether to yield only the query ids of each split in place of the
            Dataset
        shuffle : bool
            Whether to shuffle the queries before splitting the folds

        Yields:
        -------
        (train, vali, test) datasets : tuple of rankeval.dataset.Dataset
            The datasets for that split. If qids_only=True, the methods yields
            only the query ids of each fold, without creating the dataset.
        """
        fold_size = int(np.floor(self.n_queries / n_folds))
        qids = np.copy(self.query_ids)
        if shuffle:
            np.random.shuffle(qids)

        split_points = [fold_size * i for i in np.arange(n_folds)]

        for cur_fold in np.arange(n_folds):
            idx_train = split_points[cur_fold]
            idx_vali = split_points[(cur_fold - 2) % n_folds]
            idx_test = split_points[(cur_fold - 1) % n_folds]

            qids_train = qids[np.arange(
                idx_train,
                idx_vali + qids.size if idx_vali < idx_train else idx_vali
            ) % qids.size]
            qids_vali = qids[np.arange(
                idx_vali,
                idx_test + qids.size if idx_test < idx_vali else idx_test
            ) % qids.size]
            qids_test = qids[np.arange(
                idx_test,
                idx_train + qids.size if idx_train < idx_test else idx_train
            ) % qids.size]

            if qids_only:
                yield qids_train, qids_vali, qids_test
            else:
                yield self.subset(qids_train), \
                      self.subset(qids_vali), \
                      self.subset(qids_test)

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

        # Cache the hash given the computational cost to compute it
        # ASSUMPTION: the object is unmodifiable!
        if self._hash_cached is None:

            h = hashlib.md5()
            for arr in [self.X, self.y, self.query_ids]:
                h.update(arr)
            self._hash_cached = int(h.hexdigest(), 16)

        return self._hash_cached

    def __eq__(self, other):
        # use != instead of == because it is more efficient for sparse matrices:
        x_eq = not(self.X != other.X).any()
        return x_eq and (self.y == other.y).all() and \
               (self.query_ids == other.query_ids).all()

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
