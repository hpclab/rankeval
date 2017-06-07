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
    It also offers 2 methods, one for iterating over the indeces for a certain query 
    and another for iterating over the entire dataset based on those indeces. 
    """

    @abstractmethod
    def __init__(self, name):
        """
        The constructor for any metric; it initializes that metric with the proper name.
        
        Parameters
        ----------
        name : string
            Represents the name of that metric instance.
        """
        self.name = name

    @abstractmethod
    def eval(self, dataset, y_pred):
        """
        
        Parameters
        ----------
        dataset
        y_pred

        Returns
        -------

        """
        self.detailed_scores = np.zeros(dataset.n_queries, dtype=np.float32)

    @abstractmethod
    def eval_per_query(self, y, y_pred):
        """
        
        Parameters
        ----------
        y
        y_pred

        Returns
        -------

        """

    def query_indeces(self, dataset, y_pred):
        """
        
        Parameters
        ----------
        dataset : Dataset
        y_pred : numpy array

        Returns
        -------

        """
        assert len(y_pred) == len(dataset.y)
        for i in np.arange(len(dataset.query_ids)-1):
            yield np.arange(dataset.query_ids[i], dataset.query_ids[i+1])


    def query_iterator(self, dataset, y_pred):
        """
        
        :param self: 
        :param dataset: 
        :param y_pred: 
        :return: 
        """
        for query_id, query_line_indeces in enumerate(self.query_indeces(dataset, y_pred)):
            yield query_id, dataset.y[query_line_indeces], y_pred[query_line_indeces]