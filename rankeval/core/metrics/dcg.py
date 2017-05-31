import numpy as np
from rankeval.core.metrics.metric import Metric


class DCG(Metric):
    """
    This class implements DCG with several parameters
    
    
    Implementation:
    [1] Burges - "exp"
    [0] Jarvelin - "flat" 
    """

    def __init__(self, name='DCG', cutoff=None, no_relevant_results=0.0, ties=True, implementation="flat"):
        super(DCG, self).__init__(name)
        self.cutoff = cutoff
        self.no_relevant_results = no_relevant_results
        self.ties = ties
        self.implementation = implementation


    def eval(self, dataset, y_pred):
        """
        
        The method computer DCG by taking as input the dataset and the predicted document scores
        (obtained with the scoring methods). It returns the averaged DCG score over the entire dataset and the 
        detailed DCG scores per query.
        
        Parameters
        ----------
        dataset : Dataset
            Represents the Dataset object on which we want to apply the evaluation metric.
        y_pred : numpy 1d array of float
            Represent the predicted document scores for each sample/instance in the dataset.

        Returns
        -------
        avg_score: float 
            Represents the average DCG over all DCG scores per query.
        detailed_scores: numpy array of floats
            Represents the detailed DCG scores for each query. It has the length of n_queries.

        """
        super(DCG, self).eval(dataset)
        for query_id, query_y, query_y_pred in self.query_iterator(dataset, y_pred):
            self.detailed_scores[query_id] = self.eval_per_query(query_y, query_y_pred)
        return self.detailed_scores.mean(), self.detailed_scores


    def eval_per_query(self, y, y_pred):
        """
        
        This method helps compute the DCG score per query. It is called by the eval function which averages and 
        aggregates the scores for each query.
        
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
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        if self.implementation == "flat":
            dcg = sum([y[rank] / (np.math.log(float(k) + 2.0, 2))
                       for k, rank in enumerate(idx_y_pred_sorted)])
        elif self.implementation == "exp":
            dcg = sum([(2.0 ** y[rank] - 1.0) / (np.math.log(float(k) + 2.0, 2))
                       for k, rank in enumerate(idx_y_pred_sorted)])
        return dcg


    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@%d" % self.cutoff
        # TODO
        return s