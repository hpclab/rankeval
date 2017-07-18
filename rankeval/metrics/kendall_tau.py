# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import scipy.stats as stats

from rankeval.metrics.metric import Metric


class Kendalltau(Metric):
    """
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.kendalltau.html

    We use the Kendall tau coefficient implementation from scipy.
    """

    def __init__(self, name='Kendalltau'):
        """


        Parameters
        ----------
        name: string
        cutoff: int
        threshold: float
        """
        super(Kendalltau, self).__init__(name)


    def eval(self, dataset, y_pred):
        """
        This method computes the Kendall tau score over the entire dataset and the detailed scores per query.
        It calls the eval_per query method for each query in order to get the detailed Kendall tau score.

        Parameters
        ----------
        dataset : Dataset
        y_pred : numpy.array

        Returns
        -------
        float
            The overall Kendall tau score (averages over the detailed scores).
        numpy.array
            The detailed Kendall tau scores for each query, an array of length of the number of queries.
        """
        return super(Kendalltau, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        This methods computes Kendall tau at per query level (on the instances belonging to a specific query).
        The Kendall tau per query is calculated as:

            tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))

        where P is the number of concordant pairs, Q the number of discordant pairs, T the number of ties only in x,
        and U the number of ties only in y. If a tie occurs for the same pair in both x and y, it is not added to
        either T or U.


        Whether to use lexsort or quicksort as the sorting method for the initial sort of the inputs.
        Default is lexsort (True), for which kendalltau is of complexity O(n log(n)).
        If False, the complexity is O(n^2), but with a smaller pre-factor (so quicksort may be faster for small arrays).


        Parameters
        ----------
        y : numpy.array
        y_pred : numpy.array

        Returns
        -------
        float
            The Kendall tau per query.
        """
        kendall_tau = stats.kendalltau(y, y_pred, initial_lexsort=True)
        return kendall_tau.correlation


    def __str__(self):
        s = self.name
        return s