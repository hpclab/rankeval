# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import numpy as np
from rankeval.metrics import Metric


class RBP(Metric):
    """
    This class implements Ranked biased Precision (RBP) with several parameters.
    We implemented RBP as in: Alistair Moffat and Justin Zobel. 2008.
     Rank-biased precision for measurement of retrieval effectiveness.
    ACM Trans. Inf. Syst. 27, 1, Article 2 (December 2008), 27 pages.
    DOI=http://dx.doi.org/10.1145/1416950.1416952

    RBP is an extension of P@k. User has certain chance to view each result.

    RBP = E(# viewed relevant results) / E(# viewed results)

    p is based on the user model perspective and allows simulating different
    types of users, e.g.:
        p = 0.95 for persistent user
        p = 0.8 for patient users
        p = 0.5 for impatient users
        p = 0 for i'm feeling lucky - P@1

    The use of different values of p reflects different ways in which ranked
    lists can be used. Values close to 1.0 are indicative of highly persistent
    users, who scrutinize many answers before ceasing their search. For example,
    at p = 0.95, there is a roughly 60% likelihood that a user will enter a
    second page of 10 results, and a 35% chance that they will go to a third
    page. Such users obtain a relatively low per-document utility from a search
    unless a high number of relevant documents are encountered, scattered
    through a long prefix of the ranking.

    """

    _threshold = 1

    def __init__(self, name='RBP', cutoff=None, threshold=_threshold, p=0.5):
        """
        This is the constructor of RBP, an object of type Metric, with the name
        RBP. The constructor also allows setting custom values in the following
        parameters.

        Parameters
        ----------
        name: string
            RBP
        cutoff: int
            The top k results to be considered at per query level (e.g. 10)
        threshold: float
            This parameter considers relevant results all instances with labels
            different from 0, thus with a minimum label value of 1. It can be
            set to other values as well (e.g. 3), in the range of possible
            labels.
        p: float
            This parameter which simulates user type, and consequently the
            probability that a viewer actually inspects the document at rank k.
        """
        super(RBP, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold
        self.p = p

    def eval(self, dataset, y_pred):
        """
        This method takes the RBP for each query and calculates the average RBP.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply RBP.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the
            dataset.

        Returns
        -------
        avg_score: float
            The overall RBP score (averages over the detailed MAP scores).
        detailed_scores: numpy 1d array of floats
            The detailed RBP@k scores for each query, an array of length of the
            number of queries.

        """
        return super(RBP, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the RBP score per query. It is called by the
        eval function which averages and aggregates the scores for each query.

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
        rbp: float
            Represents the RBP score for one query.
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
        if self.threshold != self._threshold:
            s += "[>{}]".format(self.threshold)
        return s
