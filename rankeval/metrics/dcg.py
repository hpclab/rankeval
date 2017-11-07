# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from rankeval.metrics.metric import Metric


class DCG(Metric):
    """
    This class implements DCG with several parameters.
    """

    def __init__(self, name='DCG', cutoff=None, implementation="flat"):
        """
        This is the constructor of DCG, an object of type Metric,
        with the name DCG. The constructor also allows setting custom values
        in the following parameters.

        Parameters
        ----------
        name: string
            DCG
        cutoff: int
            The top k results to be considered at per query level (e.g. 10).
        no_relevant_results: float
            Float indicating how to treat the cases where then are no relevant
            results (e.g. 0.5).
        implementation: string
            Indicates whether to consider the flat or the exponential DCG
            formula (e.g.  {"flat", "exp"}).

        """

        super(DCG, self).__init__(name)
        self.cutoff = cutoff
        self.implementation = implementation

    def eval(self, dataset, y_pred):
        """
        The method computes DCG by taking as input the dataset and
        the predicted document scores. It returns the averaged DCG score
        over the entire dataset and the detailed DCG scores per query.
        
        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply DCG.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance
            in the dataset.

        Returns
        -------
        avg_score: float 
            Represents the average DCG over all DCG scores per query.
        detailed_scores: numpy 1d array of floats
            Represents the detailed DCG scores for each query.
            It has the length of n_queries.

        """

        return super(DCG, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the DCG score per query. It is called by
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
        dcg: float
            Represents the DCG score for one query.

        """

        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        discount = np.log2(np.arange(idx_y_pred_sorted.size) + 2)

        if self.implementation == "flat":
            gain = y[idx_y_pred_sorted]
        elif self.implementation == "exp":
            gain = np.exp2(y[idx_y_pred_sorted]) - 1.0

        dcg = (gain / discount).sum()
        return dcg

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s