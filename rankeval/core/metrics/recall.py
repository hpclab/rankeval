# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from rankeval.core.metrics.metric import Metric


class Recall(Metric):
    """
    This class implements Recall as: (relevant docs & retrieved docs) / relevant docs.
    
    """

    def __init__(self, name='Recall', cutoff=None, threshold=1):
        """
        This is the constructor of Precision, an object of type Metric, with the name Precision.
        The constructor also allows setting custom values for cutoff and threshold, otherwise it uses the default values.

        Parameters
        ----------
        name: string
        cutoff: int
        threshold: float
        """
        super(Recall, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold


    def eval(self, dataset, y_pred):
        """
        This method computes the Recall score over the entire dataset and the detailed scores per query. It calls the
        eval_per query method for each query in order to get the detailed Recall score.

        Parameters
        ----------
        dataset : Dataset
        y_pred : numpy.array

        Returns
        -------
        float
            The overall Recall score (averages over the detailed precision scores).
        numpy.array
            The detailed Recall scores for each query, an array of length of the number of queries.
        """
        return super(Recall, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        This methods computes Recall at per query level (on the instances belonging to a specific query).
        The Recall per query is calculated as <(relevant docs & retrieved docs) / relevant docs>.

        Parameters
        ----------
        y : numpy.array
        y_pred : numpy.array

        Returns
        -------
        float
            The Recall score per query.

        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_relevant_retrieved = (y[idx_y_pred_sorted] >= self.threshold).sum()
        n_relevant = (y >= self.threshold).sum()  # todo see how to deal with recall@k n_rel
        return float(n_relevant_retrieved) / n_relevant

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        s += "[>={}]".format(self.threshold)
        return s