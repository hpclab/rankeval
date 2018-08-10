# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import numpy as np
from rankeval.metrics.metric import Metric


class MAP(Metric):
    """
    This class implements MAP with several parameters. We implemented MAP as in
    https://www.kaggle.com/wiki/MeanAveragePrecision, adapted from:
    http://en.wikipedia.org/wiki/Information_retrieval
    https://www.ethz.ch/content/dam/ethz/special-interest/gess/computational-social-science-dam/documents/education/Spring2017/ML/LinkPrediction.pdf
    http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    """

    def __init__(self, name='MAP', cutoff=None, no_relevant_results=1.0):
        """
        This is the constructor of MAP, an object of type Metric, with
        the name MAP. The constructor also allows setting custom values in the
        following parameters.

        Parameters
        ----------
        name: string
            MAP
        cutoff: int
            The top k results to be considered at per query level (e.g. 10),
            otherwise the default value is None and is computed on all the
            instances of a query.
        no_relevant_results: float
            Float indicating how to treat the cases where then are no relevant
            results (e.g. 0.5). Default is 1.0.
        """
        super(MAP, self).__init__(name)
        self.cutoff = cutoff
        self.no_relevant_results = no_relevant_results

    def eval(self, dataset, y_pred):
        """
        This method takes the AP@k for each query and calculates the average,
        thus MAP@k.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply MAP.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in
            the dataset.

        Returns
        -------
        avg_score: float
            The overall MAP@k score (averages over the detailed MAP scores).
        detailed_scores: numpy 1d array of floats
            The detailed AP@k scores for each query, an array of length of
            the number of queries.
        """
        return super(MAP, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This methods computes AP@k at per query level (on the instances
        belonging to a specific query). The AP@k per query is calculated as

        ap@k = sum( P(k) / min(m,n) ), for k=1,n

        where:
            - P(k) means the precision at cut-off k in the item list. P(k)
            equals 0 when the k-th item is not followed upon recommendation
            - m is the overall number of relevant documents
            - n is the number of predicted documents

        If the denominator is zero, P(k)/min(m,n) is set to zero.

        Parameters
        ----------
        y: numpy array
            Represents the labels of instances corresponding to one query in
            the dataset (ground truth).
        y_pred: numpy array.
            Represents the predicted document scores obtained during the model
            scoring phase for that query.

        Returns
        -------
        map : float
            The MAP per query.
        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_retrieved = len(idx_y_pred_sorted)
        precision_at_i = 0.
        n_relevant_retrieved_at_i = 0.
        for i in range(n_retrieved):
            if y[idx_y_pred_sorted[i]] > 0:
                n_relevant_retrieved_at_i += 1
                precision_at_i += n_relevant_retrieved_at_i / (i + 1)

        if n_relevant_retrieved_at_i > 0:
            return precision_at_i / min(n_retrieved, np.count_nonzero(y))
        else:
            return self.no_relevant_results

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s
