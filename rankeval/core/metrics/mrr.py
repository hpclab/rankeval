# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
https://en.wikipedia.org/wiki/Mean_reciprocal_rank

"""
import numpy as np
from rankeval.core.metrics import Metric


class MRR(Metric):
    """

    """
    def __init__(self, name='MRR', cutoff=None, threshold=0):
        """

        Parameters
        ----------
        name
        cutoff
        """
        super(MRR, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold


    def eval(self, dataset, y_pred):
        """
        The mean reciprocal rank is the average of the reciprocal ranks of results for a sample of queries

        Parameters
        ----------
        dataset
        y_pred

        Returns
        -------

        """
        return super(MRR, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        We compute the reciprocal rank.
        The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer.

        Parameters
        ----------
        y
        y_pred

        Returns
        -------

        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        # rank of max predicted score
        rank_max = None
        for i, idx in enumerate(idx_y_pred_sorted):
            if y[idx] > self.threshold:
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
        s += "[>{}]".format(self.threshold)
        return s
