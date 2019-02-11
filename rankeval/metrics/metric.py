# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""

"""
import numpy as np
from abc import ABCMeta, abstractmethod
import six


class Metric(six.with_metaclass(ABCMeta)):
    """
    Metric is an abstract class which provides an interface for specific metrics.
    It also offers 2 methods, one for iterating over the indeces for a certain
    query and another for iterating over the entire dataset based on those
    indices.

    Some intuitions:
    https://stats.stackexchange.com/questions/159657/metrics-for-evaluating-ranking-algorithms
    """

    @abstractmethod
    def __init__(self, name):
        """
        The constructor for any metric; it initializes that metric with the
        proper name.
        
        Parameters
        ----------
        name : string
            Represents the name of that metric instance.
        """
        self.name = name
        self.detailed_scores = None

    @abstractmethod
    def eval(self, dataset, y_pred):
        """
        This abstract method computes a specific metric over the predicted
        scores for a test dataset. It calls the eval_per query method for each
        query in order to get the detailed metric score.

        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which we want to apply the metric.
        y_pred : numpy 1d array of float
            Represents the predicted document scores for each instance in the
            dataset.

        Returns
        -------
        avg_score: float
            Represents the average values of a metric over all metric scores
            per query.
        detailed_scores: numpy 1d array of floats
            Represents the detailed metric scores for each query. It has the
            length of n_queries.
        """
        self.detailed_scores = np.zeros(dataset.n_queries, dtype=np.float32)

        for rel_qid, (qid, q_y, q_y_pred) in enumerate(
                self.query_iterator(dataset, y_pred)):
            self.detailed_scores[rel_qid] = self.eval_per_query(q_y, q_y_pred)
        return np.nanmean(self.detailed_scores), self.detailed_scores

    @abstractmethod
    def eval_per_query(self, y, y_pred):
        """
        This methods helps to evaluate the predicted scores for a specific
        query within the dataset.

        Parameters
        ----------
        y: numpy array
            Represents the instance labels corresponding to the queries in the
            dataset (ground truth).
        y_pred: numpy array.
            Represents the predicted document scores obtained during the model
            scoring phase for that query.

        Returns
        -------
        : float
            Represents the metric score for one query.
        """

    def query_iterator(self, dataset, y_pred):
        """
        This method iterates over dataset document scores and predicted scores
        in blocks of instances which belong to the same query.
        Parameters
        ----------
        dataset :  Datatset
        y_pred  : numpy array

        Returns
        -------
        : int
            The query id.
        : numpy.array
            The document scores of the instances in the labeled dataset
            (instance labels) belonging to the same query id.
        : numpy.array
            The predicted scores for the instances in the dataset belonging to
            the same query id.
        """
        for query_id, start_offset, end_offset in dataset.query_iterator():
            yield (query_id,
                   dataset.y[start_offset:end_offset],
                   y_pred[start_offset:end_offset])
