# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import numpy as np

from rankeval.metrics import Metric


class Pfound(Metric):
    """
    pFound

    https://www.quora.com/How-can-search-quality-be-measured
    http://romip.ru/romip2009/15_yandex.pdf
    """
    def __init__(self, name='Pfound', cutoff=None, threshold=0, p_abandonment=0.15):
        """

        Parameters
        ----------
        name
        cutoff
        threshold
        """
        super(Pfound, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold
        self.p_abandonment = p_abandonment


    def eval(self, dataset, y_pred):
        """

        Parameters
        ----------
        dataset
        y_pred

        Returns
        -------

        """
        return super(Pfound, self).eval(dataset, y_pred)


    def eval_per_query(self, y, y_pred):
        """
        pFound was implemented as ERR in section 7.2 http://olivier.chapelle.cc/pub/err.pdf
            - http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf
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
            err += prob_step_down * utility * pow(self.p_abandonment, i)
            prob_step_down *= (1. - utility)

        return err


    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        s += "[>{}]".format(self.threshold)
        return s