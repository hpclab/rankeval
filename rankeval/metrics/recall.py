# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from rankeval.metrics.metric import Metric


class Recall(Metric):
    """
    This class implements Recall as:
    (relevant docs & retrieved docs) / relevant docs.

    It allows setting custom values for cutoff and threshold, otherwise it uses
    the default values.

    """

    _threshold = 1

    def __init__(self, name='R', no_relevant_results=0.0,
                 cutoff=None, threshold=_threshold):
        """
        This is the constructor of Recall, an object of type Metric, with
        the name R. The constructor also allows setting custom values
        for cutoff and threshold, otherwise it uses the default values.

        Parameters
        ----------
        name: string
            R
        no_relevant_results: float
            Float indicating how to treat the cases where then are no relevant
            results (e.g. 0.0).
        cutoff: int
            The top k results to be considered at per query level (e.g. 10)
        threshold: float
            This parameter considers relevant results all instances with labels
            different from 0, thus with a minimum label value of 1. It can be
            set to other values as well (e.g. 3), in the range of possible
            labels.
        """
        super(Recall, self).__init__(name)
        self.no_relevant_results = no_relevant_results
        self.cutoff = cutoff
        self.threshold = threshold

    def eval(self, dataset, y_pred):
        """
        This method computes the Recall score over the entire dataset and the
        detailed scores per query. It calls the eval_per query method for each
        query in order to get the detailed Recall score.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply Recall.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the
            dataset.

        Returns
        -------
        avg_score: float
            The overall Recall score (averages over the detailed precision
            scores).
        detailed_scores: numpy 1d array of floats
            The detailed Recall scores for each query, an array of length of
            the number of queries.
        """
        return super(Recall, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        This methods computes Recall at per query level (on the instances
        belonging to a specific query). The Recall per query is calculated as
        <(relevant docs & retrieved docs) / relevant docs>.

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
        recall: float
            The Recall score per query.

        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_relevant_retrieved = (y[idx_y_pred_sorted] >= self.threshold).sum()
        n_relevant = (y >= self.threshold).sum()

        if n_relevant != 0:
            return float(n_relevant_retrieved) / n_relevant
        else:
            return self.no_relevant_results

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        if self.threshold != self._threshold:
            s += "[>{}]".format(self.threshold)
        return s
