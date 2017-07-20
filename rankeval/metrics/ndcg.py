# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from rankeval.metrics.dcg import DCG

from rankeval.metrics.metric import Metric


class NDCG(Metric):
    """
    This class implements NDCG with several parameters.

    Attributes
    ----------
    name: string
        NDCG
    cutoff: int
        The top k results to be considered at per query level (e.g. 10)
    no_relevant_results: float
        Float indicating how to treat the cases where then are no relevant results (e.g. 0.5)
    implementation: string
        Indicates whether to consider the flat or the exponential DCG formula (e.g.  {"flat", "exp"})
    """

    def __init__(self, name='NDCG', cutoff=None, no_relevant_results=0.0, implementation="flat"):
        """
        This is the constructor of NDCG, an object of type Metric, with the name NDCG.
        The constructor also allows setting custom parameter values.

        Parameters
        ----------
        name : string
        cutoff: int
        no_relevant_results: float
        implementation: string
            it can range between {"flat", "exp"}
        """
        super(NDCG, self).__init__(name)
        self.cutoff = cutoff
        self.no_relevant_results = no_relevant_results
        self.implementation = implementation
        self.dcg = DCG(cutoff=self.cutoff, implementation=self.implementation)

    def eval(self, dataset, y_pred):
        """
        The method computes NDCG by taking as input the dataset and the predicted document scores.
        It returns the averaged NDCG score over the entire dataset and the detailed NDCG scores per query.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply NDCG.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the dataset.

        Returns
        -------
        avg_score: float
            Represents the average NDCG over all NDCG scores per query.
        detailed_scores: numpy array of floats
            Represents the detailed NDCG scores for each query. It has the length of n_queries.

        """
        return super(NDCG, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the NDCG score per query. It is called by the eval function which averages and
        aggregates the scores for each query.

        It calculates NDCG per query as <dcg_score/idcg_score>.
        If there are no relevant results, NDCG returns the values set by default
        or by the user when creating the metric.

        Parameters
        ----------
        y: numpy array
            Represents the labels of instances corresponding to one query in the dataset (ground truth).
        y_pred: numpy array.
            Represents the predicted document scores obtained during the model scoring phase for that query.

        Returns
        -------
        dcg: float
            Represents the DCG score for one query.

        """
        dcg_score = self.dcg.eval_per_query(y, y_pred)
        idcg_score = self.dcg.eval_per_query(y, y)

        if idcg_score != 0:
            ndcg = dcg_score / idcg_score
        else:
            ndcg = self.no_relevant_results
        return ndcg

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s







