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
    This class implements Mean squared error (MSE) with several parameters.

    """
    def __init__(self, name='MSE', cutoff=None):
        """
        This is the constructor of MSE, an object of type Metric, with
        the name MSE. The constructor also allows setting custom values in
        the following parameters.

        Parameters
        ----------
        name: string
            MSE
        cutoff: int
            The top k results to be considered at per query level (e.g. 10),
            otherwise the default value is None and is computed on all the
            instances of a query.
        """
        super(self.__class__, self).__init__(name)
        self.cutoff = cutoff

    def eval(self, dataset, y_pred):
        """
        This method takes the MSE for each query and calculates
        the average MSE.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply MSE.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance
            in the dataset.

        Returns
        -------
        avg_score: float
            The overall MSE score (summed over the detailed MSE scores).
        detailed_scores: numpy 1d array of floats
            The detailed MSE@k scores for each query, an array of length of
            the number of queries.
        """
        # return super(self.__class__, self).eval(dataset, y_pred)

        self.detailed_scores = np.zeros(dataset.n_queries, dtype=np.float32)

        for qid, q_y, q_y_pred in self.query_iterator(dataset, y_pred):
            self.detailed_scores[qid] = \
                self.eval_per_query(q_y, q_y_pred) / dataset.n_instances
        return self.detailed_scores.sum(), self.detailed_scores

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the MSE score per query. It is called by
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
            Represents the MSE score for one query.

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



