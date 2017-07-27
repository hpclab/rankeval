# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# http://olivier.chapelle.cc/pub/err.pdf
import numpy as np
from rankeval.metrics import Metric


class ERR(Metric):
    """
    This class implements Expected Reciprocal Rank as proposed
    in http://olivier.chapelle.cc/pub/err.pdf

    """

    def __init__(self, name='ERR', cutoff=None):
        """
        This is the constructor of ERR, an object of type Metric,
        with the name ERR. The constructor also allows setting custom values
        in the following parameters.

        Parameters
        ----------
        name: string
            ERR
        cutoff: int
            The top k results to be considered at per query level (e.g. 10)

        """

        super(ERR, self).__init__(name)
        self.cutoff = cutoff

    def eval(self, dataset, y_pred):
        """
        The method computes ERR by taking as input the dataset and the
        predicted document scores. It returns the averaged ERR score over
        the entire dataset and the detailed ERR scores per query.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply ERR.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance
            in the dataset.

        Returns
        -------
        avg_score: float
            Represents the average ERR over all ERR scores per query.
        detailed_scores: numpy 1d array of floats
            Represents the detailed ERR scores for each query. It has the
            length of n_queries.

        """
        return super(ERR, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the ERR score per query. It is called by
        the eval function which averages and aggregates the scores
        for each query.

        Parameters
        ----------
        y: numpy array
            Represents the labels of instances corresponding to one query in
            the dataset (ground truth).
        y_pred: numpy array.
            Represents the predicted document scores obtained during
            the model scoring phase for that query.

        Returns
        -------
        err: float
            Represents the ERR score for one query.

        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        max_grade = y.max()  # max relevance score
        prob_step_down = 1.0
        err = 0.0

        for i, idx in enumerate(idx_y_pred_sorted):
            utility = (pow(2., y[idx]) - 1.) / pow(2., max_grade)
            err += prob_step_down * (utility / (i + 1.))
            prob_step_down *= (1. - utility)

        return err

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s