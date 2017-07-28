# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from rankeval.metrics import Metric, MSE


class RMSE(Metric):
    """
    This class implements Root mean squared error (RMSE) with
    several parameters.

    """
    def __init__(self, name='RMSE', cutoff=None):
        """
        This is the constructor of RMSE, an object of type Metric, with the
        name RMSE. The constructor also allows setting custom values in the
        following parameters.

        Parameters
        ----------
        name: string
            RMSE
        cutoff: int
            The top k results to be considered at per query level (e.g. 10),
            otherwise the default value is None and is computed on all the
            instances of a query.
        """
        super(self.__class__, self).__init__(name)
        self.cutoff = cutoff
        self._mse = MSE(cutoff=cutoff)

    def eval(self, dataset, y_pred):
        """
        This method takes the RMSE for each query and calculates
        the average RMSE.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply RMSE.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance
            in the dataset.

        Returns
        -------
        avg_score: float
            The overall RMSE score (averages over the detailed RMSE scores).
        detailed_scores: numpy 1d array of floats
            The detailed RMSE@k scores for each query, an array of length of
            the number of queries.
        """
        return super(self.__class__, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the RMSE score per query. It is called by
        the eval function which averages and aggregates the scores
        for each query.

        Parameters
        ----------
        y: numpy array
            Represents the labels of instances corresponding to one query in
            the dataset (ground truth).
        y_pred: numpy array.
            Represents the predicted document scores obtained during the model
            scoring phase for that query.

        Returns
        -------
        rmse: float
            Represents the RMSE score for one query.
        """
        mse = self._mse.eval_per_query(y, y_pred)
        return np.sqrt(mse)

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s



