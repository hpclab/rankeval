import numpy as np
from rankeval.core.metrics.metric import Metric


class Precision(Metric):
    """
    This class implements Precision as: (relevant docs & retrieved docs) / retrieved docs
    """

    def __init__(self, name='Precision', cutoff=None, threshold=0):
        super(Precision, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold

    def eval(self, dataset, y_pred):
        super(Precision, self).eval(dataset)
        for query_id, query_y, query_y_pred in self.query_iterator(dataset, y_pred):
            self.detailed_scores[query_id] = self.eval_per_query(query_y, query_y_pred)
        return self.detailed_scores.mean(), self.detailed_scores


    def eval_per_query(self, y, y_pred):
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_relevant_retrieved = (y[idx_y_pred_sorted] > self.threshold).sum()
        precision =  float(n_relevant_retrieved) / len(idx_y_pred_sorted)

        return precision

    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@%d" % self.cutoff
        s += "[>%d]" % self.threshold
        return s