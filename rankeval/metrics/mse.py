# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np

from rankeval.metrics import Metric


class MSE(Metric):
    """
    Mean squared error

    """
    def __init__(self, name='MSE', cutoff=None):
        """

        Parameters
        ----------
        name
        cutoff
        """
        super(self.__class__, self).__init__(name)
        self.cutoff = cutoff

    def eval(self, dataset, y_pred):
        """

        Parameters
        ----------
        dataset
        y_pred

        Returns
        -------

        """
        # return super(self.__class__, self).eval(dataset, y_pred)

        self.detailed_scores = np.zeros(dataset.n_queries, dtype=np.float32)

        for qid, q_y, q_y_pred in self.query_iterator(dataset, y_pred):
            self.detailed_scores[qid] = \
                self.eval_per_query(q_y, q_y_pred) / dataset.n_instances
        return self.detailed_scores.sum(), self.detailed_scores

    def eval_per_query(self, y, y_pred):
        """

        Parameters
        ----------
        y
        y_pred

        Returns
        -------

        """
        if self.cutoff is not None:
            idx = np.argsort(y_pred)[::-1][:self.cutoff]
            return ((y[idx] - y_pred[idx]) ** 2).sum()
        else:
            return ((y - y_pred) ** 2.0).sum()

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s



