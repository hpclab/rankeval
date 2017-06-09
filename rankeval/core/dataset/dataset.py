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

from rankeval.core.dataset.svmlight_format import load_svmlight_file, dump_svmlight_file


class Dataset(object):

    def __init__(self, f, name=None, format="svmlight"):
        """
        This module implements the generic class for loading/dumping a dataset from/to file.

        Parameters
        ----------
        f : path
            The file name of the dataset to load
        name : str
            The name to be given to the current dataset
        format : str
            The format of the dataset file to load (actually supported is only "svmlight" format)

        Attributes
        -------
        f : path
            The file name of the dataset to load
        X : numpy 2d array of float
            It is a dense numpy matrix of shape (n_samples, n_features),
        y : numpy 1d array of float
            It is a ndarray of shape (n_samples,) with the gold label
        query_ids : numpy 1d array of int
            It is a ndarray of shape(nsamples,)
        """
        if format == "svmlight":
            self.X, self.y, self.query_ids = load_svmlight_file(f, query_id=True)
        else:
            raise TypeError("Dataset format %s is not yet supported!" % format)

        self.file = self.name = f
        if name is not None:
            self.name = name

        self.n_instances = len(self.y)
        self.n_queries = len(self.query_ids) - 1

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
            # we need to unroll the query_ids (it is compacted: it reports only the offset where a new query id starts)
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
