# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

"""
import numpy as np
from rankeval.metrics import Metric


class MRR(Metric):
    """
    This class implements Mean Reciprocal Rank.

    """

    _threshold = 1

    def __init__(self, name='MRR', cutoff=None, threshold=_threshold):
        """
        This is the constructor of MRR, an object of type Metric, with the
        name MRR. The constructor also allows setting custom values in the
        following parameters.

        Parameters
        ----------
        name: string
            MRR
        cutoff: int
            The top k results to be considered at per query level (e.g. 10)
        threshold: float
            This parameter considers relevant results all instances with labels
            different from 0, thus with a minimum label value of 1. It can be
            set to other values as well (e.g. 3), in the range of possible labels.
        """
        super(MRR, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold

    def eval(self, dataset, y_pred):
        """
        The method computes MRR by taking as input the dataset and the predicted
        document scores. It returns the averaged MRR score over the entire
        dataset and the detailed MRR scores per query.

        The mean reciprocal rank is the average of the reciprocal ranks of
        results for a sample of queries.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply MRR.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance
            in the dataset.

        Returns
        -------
        avg_score: float
            Represents the average MRR over all MRR scores per query.
        detailed_scores: numpy 1d array of floats
            Represents the detailed MRR scores for each query. It has
            the length of n_queries.

        """
        return super(MRR, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the MRR score per query. It is called by the
        eval function which averages and aggregates the scores for each query.

        We compute the reciprocal rank. The reciprocal rank of a query response
        is the multiplicative inverse of the rank of the first correct answer.

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
        mrr: float
            Represents the MRR score for one query.
        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        # rank of max predicted score
        rank_max = None
        for i, idx in enumerate(idx_y_pred_sorted):
            if y[idx] >= self.threshold:
                rank_max = i
                break

        if rank_max is not None:
            return 1./(rank_max+1)
        else:
            return 0.

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        if self.threshold != self._threshold:
            s += "[>{}]".format(self.threshold)
        return s
