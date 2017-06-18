# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision


import numpy as np
from rankeval.core.metrics.metric import Metric


class MAP(Metric):
    """
    https://www.kaggle.com/wiki/MeanAveragePrecision
    """

    def __init__(self, name='MAP', cutoff=None, threshold=0):
        """


        Parameters
        ----------
        name: string
        cutoff: int
        threshold: float
        """
        super(MAP, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold


    def eval(self, dataset, y_pred):
        """
        This method take the AP@k for each query and calculates the average, thus MAP@ksss.

        Parameters
        ----------
        dataset : Dataset
        y_pred : numpy.array

        Returns
        -------
        float
            The overall MAP@k score (averages over the detailed MAP scores).
        numpy.array
            The detailed AP@k scores for each query, an array of length of the number of queries.
        """
        return super(MAP, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        This methods computes AP@k at per query level (on the instances belonging to a specific query).
        The AP@k per query is calculated as

        Parameters
        ----------
        y : numpy.array
        y_pred : numpy.array

        Returns
        -------
        float
            The MAP per query.
        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_retrieved = len(idx_y_pred_sorted)
        precision_at_i = 0.
        n_relevant_retrieved_at_i = 0.
        for i in range(n_retrieved):
            if y[idx_y_pred_sorted[i]] != 0:
                n_relevant_retrieved_at_i += 1
                precision_at_i += float(n_relevant_retrieved_at_i) / (i + 1)

        return precision_at_i / n_retrieved


    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        s += "[>{}]".format(self.threshold)
        return s
