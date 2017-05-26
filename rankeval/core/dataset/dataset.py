"""
This module implements the generic class for loading/dumping a dataset from/to file.
"""
import numpy as np

from rankeval.core.dataset.svmlight_format import load_svmlight_file, dump_svmlight_file


class Dataset(object):

    def __init__(self, dataset_file, dataset_format="svmlight"):
        """
        This module implements the generic class for loading/dumping a dataset from/to file.

        Parameters
        ----------
        dataset_file : path
            The file to the dataset file to load
        dataset_format : str
            The format of the dataset file to load (actually supported is only "svmlight" format)

        Attributes
        -------
        X : numpy 2d array of float
            It is a dense numpy matrix of shape (n_samples, n_features),
        y : numpy 1d array of float
            It is a ndarray of shape (n_samples,) with the gold label
        query_ids : numpy 1d array of int
            It is a ndarray of shape(nsamples,)
        """
        if dataset_format == "svmlight":
            self.X, self.y, self.query_ids = load_svmlight_file(dataset_file, query_id=True)
        else:
            raise TypeError("Dataset format %s is not yet supported!" % dataset_format)

    def dump(self, dataset_file, dataset_format):
        """
        This method implements the writing of a previously loaded dataset according to the given format on file

        Parameters
        ----------
        dataset_file : path
            The file path where to store the dataset
        dataset_format : str
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

        if dataset_format == "svmlight":
            dump_svmlight_file(self.X, self.y, dataset_file, query_ids)
        else:
            raise TypeError("Dataset format %s is not yet supported!" % dataset_format)