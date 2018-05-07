# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import defaultdict
import numpy as np

from rankeval.metrics.dcg import DCG
from rankeval.metrics.metric import Metric


class NDCG(Metric):
    """
    This class implements NDCG with several parameters.

    """

    def __init__(self, name='NDCG', cutoff=None, no_relevant_results=1.0,
                 implementation="exp"):
        """
        This is the constructor of NDCG, an object of type Metric, with the
        name NDCG.
        The constructor also allows setting custom values
            - cutoff: the top k results to be considered at per query level
            - no_relevant_results: is a float values indicating how to treat
                the cases where then are no relevant results
            - ties: indicates how we should consider the ties
            - implementation: indicates whether to consider the flat or the
                exponential NDCG formula

        Parameters
        ----------
        name: string
            NDCG
        cutoff: int
            The top k results to be considered at per query level (e.g. 10)
        no_relevant_results: float
            Float indicating how to treat the cases where then are no relevant
            results (e.g. 0.5). Default is 1.0.
        implementation: string
            Indicates whether to consider the flat or the exponential DCG
            formula: "flat" or "exp" (default).
        """

        super(self.__class__, self).__init__(name)
        self.cutoff = cutoff
        self.no_relevant_results = no_relevant_results
        self.implementation = implementation
        self.dcg = DCG(cutoff=self.cutoff,
                       implementation=self.implementation)

        self._current_dataset = None
        self._current_rel_qid = None
        self._cache_idcg_score = defaultdict(int)

    def eval(self, dataset, y_pred):
        """
        The method computes NDCG by taking as input the dataset and the
        predicted document scores (obtained with the scoring methods). It
        returns the averaged NDCG score over the entire dataset and the
        detailed NDCG scores per query.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply NDCG.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the
            dataset.

        Returns
        -------
        avg_score: float
            Represents the average NDCG over all NDCG scores per query.
        detailed_scores: numpy array of floats
            Represents the detailed NDCG scores for each query. It has the
            length of n_queries.

        """
        # used to cache ideal DCG scores on a dataset basis
        self._current_dataset = dataset
        self._current_rel_qid = 0

        # Compute the ideal DCG scores only once and cache them
        if self._current_dataset not in self._cache_idcg_score:

            idcg_score = np.ndarray(shape=dataset.n_queries, dtype=np.float32)
            for rel_id, (qid, q_y, _) in enumerate(
                    self.query_iterator(dataset, dataset.y)):
                idcg_score[rel_id] = self.dcg.eval_per_query(q_y, q_y)

            self._cache_idcg_score[self._current_dataset] = idcg_score

        return super(self.__class__, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the NDCG score per query. It is called by the
        eval function which averages and aggregates the scores for each query.

        It calculates NDCG per query as <dcg_score/idcg_score>.
        If there are no relevant results, NDCG returns the values set by default
        or by the user when creating the metric.

        Parameters
        ----------
        y: numpy array
            Represents the labels of instances corresponding to one query in the
            dataset (ground truth).
        y_pred: numpy array.
            Represents the predicted document scores obtained during the model
            scoring phase for that query.

        Returns
        -------
        dcg: float
            Represents the DCG score for one query.
        """
        dcg_score = self.dcg.eval_per_query(y, y_pred)

        if self._current_rel_qid is not None:
            idcg_score = \
                self._cache_idcg_score[self._current_dataset][self._current_rel_qid]
            self._current_rel_qid += 1
        else:
            idcg_score = self.dcg.eval_per_query(y, y)

        if idcg_score != 0:
            ndcg = dcg_score / idcg_score
        else:
            ndcg = self.no_relevant_results
        return ndcg

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s








