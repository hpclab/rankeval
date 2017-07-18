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
    Expected Reciprocal Rank
    http://olivier.chapelle.cc/pub/err.pdf
    """
    def __init__(self, name='ERR', cutoff=None, threshold=0):
        """

        Parameters
        ----------
        name
        cutoff
        threshold
        """
        super(ERR, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold


    def eval(self, dataset, y_pred):
        """

        Parameters
        ----------
        dataset
        y_pred

        Returns
        -------

        """
        return super(ERR, self).eval(dataset, y_pred)

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
        s += "[>{}]".format(self.threshold)
        return s