"""

"""
import numpy as np
from abc import ABCMeta, abstractmethod
import six


class Metric(six.with_metaclass(ABCMeta)):
    """
    
    """
    @abstractmethod
    def __init__(self, name):
        pass

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

    def query_indeces(self, dataset, y_pred):
        """
        
        Parameters
        ----------
        dataset : Dataset
            fdfdsfsd
        y_pred : numpy array
            fdfdfd

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