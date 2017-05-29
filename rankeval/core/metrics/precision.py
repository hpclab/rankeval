import numpy as np
from rankeval.core.metrics.metric import Metric


class Precision(Metric):

    def __init__(self, name='Precision', cutoff=None, threshold=0):
        self.name = name
        self.cutoff = cutoff
        self.threshold = threshold

    def eval(self, dataset, y_pred):
        super(Precision, self).eval(dataset)

        for query_id, query_y, query_y_pred in self.query_iterator(dataset, y_pred):

            query_y_pred_sorted = np.argsort(query_y_pred)[::-1]
            if self.cutoff is not None:
                query_y_pred_sorted = query_y_pred_sorted[:self.cutoff]

            n_relevant_retrieved = (query_y[query_y_pred_sorted] > self.threshold).sum()

            self.detailed_scores[query_id] = n_relevant_retrieved / len(query_y_pred_sorted)

        return self.detailed_scores.mean(), self.detailed_scores

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@%d" % self.cutoff
        s += "[>%d]" % self.threshold
        return s