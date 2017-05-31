from rankeval.core.metrics.dcg import DCG
from rankeval.core.metrics.metric import Metric


class NDCG(Metric):
    """
    This class implements DCG with several parameters


    IMPL:
    [1] Burges - "exp"
    [0] Jarvelin - "flat" 
    """

    def __init__(self, name='NDCG', cutoff=None, no_relevant_results=0.0, ties=True, implementation="flat"):
        super(NDCG, self).__init__(name)
        self.cutoff = cutoff
        self.no_relevant_results = no_relevant_results
        self.ties = ties
        self.implementation = implementation
        self.dcg = DCG(cutoff=self.cutoff, no_relevant_results=self.no_relevant_results,
                  ties=self.ties, implementation=self.implementation)

    def eval(self, dataset, y_pred):
        super(NDCG, self).eval(dataset)
        for query_id, query_y, query_y_pred in self.query_iterator(dataset, y_pred):
            self.detailed_scores[query_id] = self.eval_per_query(query_y, query_y_pred)
        return self.detailed_scores.mean(), self.detailed_scores


    def eval_per_query(self, y, y_pred):
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
            s += "@%d" % self.cutoff
        # TODO
        return s








