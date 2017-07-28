# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import scipy.stats as stats
from rankeval.metrics.metric import Metric


class SpearmanRho(Metric):
    """
    This class implements Spearman's Rho.
    We use the Spearman Rho coefficient implementation from scipy.

    """

    def __init__(self, name='Rho'):
        """
        This is the constructor of Spearman Rho, an object of type Metric, with
        the name Rho. The constructor also allows setting custom values in the
        following parameters.

        Parameters
        ----------
        name: string
            Rho
        """
        super(SpearmanRho, self).__init__(name)

    def eval(self, dataset, y_pred):
        """
        This method computes the Spearman Rho tau score over the entire dataset
        and the detailed scores per query. It calls the eval_per query method
        for each query in order to get the detailed Spearman Rho score.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply Spearman Rho.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the
            dataset.

        Returns
        -------
        avg_score: float
            The overall Spearman Rho score (averages over the detailed scores).
        detailed_scores: numpy 1d array of floats
            The detailed Spearman Rho scores for each query, an array of length
            of the number of queries.
        """
        return super(SpearmanRho, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This methods computes Spearman Rho at per query level (on the instances
        belonging to a specific query).

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
        rho: float
            The Spearman Rho per query.
        """
        spearman_rho = stats.spearmanr(y, y_pred)
        return spearman_rho.correlation

    def __str__(self):
        s = self.name
        return s
