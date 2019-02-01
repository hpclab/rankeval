# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from rankeval.metrics.metric import Metric


class PrecisionMax(Metric):
    """
    This class implements Precision as:
    (relevant docs & retrieved docs) / retrieved docs.

    The particularity of this implementation is that the threshold is not global
    to all the queries as in `Precision`, but is dynamically computed for every
    query considering the document with top label. Thus this metric enables to
    compute the precision regardless of a fixed label.
    """

    def __init__(self, name='P', cutoff=None):
        """
        This is the constructor of Precision, an object of type Metric, with
        the name P. The constructor also allows setting custom values for cutoff
        and threshold, otherwise it uses the default values.

        Parameters
        ----------
        name: string
            P
        cutoff: int
            The top k results to be considered at per query level (e.g. 10)

        """
        super(PrecisionMax, self).__init__(name)
        self.cutoff = cutoff

    def eval(self, dataset, y_pred):
        """
        This method computes the Precision score over the entire dataset and
        the detailed scores per query. It calls the eval_per query method for
        each query in order to get the detailed Precision score.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply Precision.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the
            dataset.

        Returns
        -------
        avg_score: float
            The overall Precision score (averages over the detailed precision
            scores).
        detailed_scores: numpy 1d array of floats
            The detailed Precision scores for each query, an array of length of
            the number of queries.
        """
        return super(PrecisionMax, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This methods computes Precision at per query level (on the instances
        belonging to a specific query). The Precision per query is calculated as
        <(relevant docs & retrieved docs) / retrieved docs>.

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
        precision: float
            The precision per query.
        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_relevant_retrieved = (y[idx_y_pred_sorted] >= y.max()).sum()
        return float(n_relevant_retrieved) / len(idx_y_pred_sorted)

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s
