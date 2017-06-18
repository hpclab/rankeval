# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import numpy as np
from rankeval.core.metrics import Metric


class RBP(Metric):
    """
    Ranked biased Precision.
    Alistair Moffat and Justin Zobel. 2008. Rank-biased precision for measurement of retrieval effectiveness.
    ACM Trans. Inf. Syst. 27, 1, Article 2 (December 2008), 27 pages. DOI=http://dx.doi.org/10.1145/1416950.1416952

    RBP is an extension of P@k. User has certain chance to view each result.

    RBP = E(# viewed relevant results) / E(# viewed results)

    p allows simulating different types of users, e.g.,
        p = 0.95 for persistent user
        p = 0.8 for patient users
        p = 0.5 for impatient users
        p = 0 for i'm feeling lucky - P@1

    """

    def __init__(self, name='RBP', cutoff=None, threshold=0, p=0.5):
        """

        Parameters
        ----------
        name
        cutoff
        threshold
        """
        super(RBP, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold
        self.p = p

    def eval(self, dataset, y_pred):
        """

        Parameters
        ----------
        dataset
        y_pred

        Returns
        -------

        """
        return super(RBP, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """

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

        discount = np.power(self.p, np.arange(len(idx_y_pred_sorted)))
        gain = y[idx_y_pred_sorted] >= self.threshold

        rbp = (1. - self.p) * (gain * discount).sum()
        return rbp

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        s += "[>={}]".format(self.threshold)
        return s