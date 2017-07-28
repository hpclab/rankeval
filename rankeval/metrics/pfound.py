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
    This class implements Pfound with several parameters.

    The ERR metric is very similar to the pFound metric used by
    Yandex (Segalovich, 2010).
    [http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf].

    In fact pFound is identical to the ERR variant described in
    (Chapelle et al., 2009, Section 7.2). We implemented pFound similar
    to ERR in section 7.2 of http://olivier.chapelle.cc/pub/err.pdf.

    """
    def __init__(self, name='Pf', cutoff=None, p_abandonment=0.15):
        """
        This is the constructor of Pfound, an object of type Metric, with
        the name Pf. The constructor also allows setting custom values in
        the following parameters.

        Parameters
        ----------
        name: string
            Pf
        cutoff: int
            The top k results to be considered at per query level (e.g. 10),
            otherwise the default value is None and is computed on all the
            instances of a query.
        p_abandonment: float
            This parameter indicates the probability of abandonment, i.e.
            the user stops looking a the ranked list due to an external reason.
            The original cascade model of ERR has later been extended to include
            an abandonment probability: if the user is not satisfied at a given
            position, he will examine the next url with probability y, but has
            a probability 1-y of abandoning.

        """
        super(Pfound, self).__init__(name)
        self.cutoff = cutoff
        self.p_abandonment = p_abandonment

    def eval(self, dataset, y_pred):
        """
        The method computes Pfound by taking as input the dataset and the
        predicted document scores. It returns the averaged Pfound score over
        the entire dataset and the detailed Pfound scores per query.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which to apply Pfound.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in
            the dataset.

        Returns
        -------
        avg_score: float
            Represents the average Pfound over all Pfound scores per query.
        detailed_scores: numpy 1d array of floats
            Represents the detailed Pfound scores for each query. It has the
            length of n_queries.
        """
        return super(Pfound, self).eval(dataset, y_pred)

    def eval_per_query(self, y, y_pred):
        """
        This method helps compute the Pfound score per query. It is called by
        the eval function which averages and aggregates the scores for each
        query.

        Parameters
        ----------
        y: numpy array
            Represents the labels of instances corresponding to one query in
            the dataset (ground truth).
        y_pred: numpy array
            Represents the predicted document scores obtained during the model
            scoring phase for that query.

        Returns
        -------
        pfound: float
            Represents the Pfound score for one query.

        """
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        max_grade = y.max()  # max relevance score
        prob_step_down = 1.0
        pfound = 0.0

        for i, idx in enumerate(idx_y_pred_sorted):
            utility = (pow(2., y[idx]) - 1.) / pow(2., max_grade)
            pfound += prob_step_down * utility * pow(self.p_abandonment, i)
            prob_step_down *= (1. - utility)

        return pfound

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@{}".format(self.cutoff)
        return s