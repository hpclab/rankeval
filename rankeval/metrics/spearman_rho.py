# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

import scipy.stats as stats

from rankeval.metrics.metric import Metric


class SpearmanRho(Metric):
    """


    We use the Spearman Rho coefficient implementation from scipy.
    """

    def __init__(self, name='SpearmanRho'):
        """


        Parameters
        ----------
        name: string
        cutoff: int
        threshold: float
        """
        super(SpearmanRho, self).__init__(name)


    def eval(self, dataset, y_pred):
        """
        This method computes the Spearman Rho tau score over the entire dataset and the detailed scores per query.
        It calls the eval_per query method for each query in order to get the detailed Spearman Rho score.

        Parameters
        ----------
        dataset : Dataset
        y_pred : numpy.array

        Returns
        -------
        float
            The overall Spearman Rho score (averages over the detailed scores).
        numpy.array
            The detailed Spearman Rho scores for each query, an array of length of the number of queries.
        """
        return super(SpearmanRho, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        This methods computes Spearman Rho at per query level (on the instances belonging to a specific query).


        Parameters
        ----------
        y : numpy.array
        y_pred : numpy.array

        Returns
        -------
        float
            The Spearman Rho per query.
        """
        spearman_rho = stats.spearmanr(y, y_pred)
        return spearman_rho.correlation


    def __str__(self):
        s = self.name
        return s