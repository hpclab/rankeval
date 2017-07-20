# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np

from rankeval.metrics import Metric


class RMSE(Metric):
    """
    Root mean squared error

    """
    def __init__(self, name='RMSE', cutoff=None):
        """

        Parameters
        ----------
        name
        cutoff
        """
        super(RMSE, self).__init__(name)
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
        return super(RMSE, self).eval(dataset, y_pred)

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

        mse = ((y[idx_y_pred_sorted] - y_pred[idx_y_pred_sorted]) ** 2).mean()
        return np.sqrt(mse)

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s



