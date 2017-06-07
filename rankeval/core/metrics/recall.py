# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from rankeval.core.metrics.metric import Metric


class Recall(Metric):
    """
    This class implements Recall as: (relevant docs & retrieved docs) / relevant docs
    
    """

    def __init__(self, name='Recall', cutoff=None, threshold=0):
        super(Recall, self).__init__(name)
        self.cutoff = cutoff
        self.threshold = threshold

    def eval(self, dataset, y_pred):
        super(Recall, self).eval(dataset, y_pred)

        for query_id, query_y, query_y_pred in self.query_iterator(dataset, y_pred):
            self.detailed_scores[query_id] = self.eval_per_query(query_y, query_y_pred)
        return self.detailed_scores.mean(), self.detailed_scores


    def eval_per_query(self, y, y_pred):
        idx_y_pred_sorted = np.argsort(y_pred)[::-1]
        if self.cutoff is not None:
            idx_y_pred_sorted = idx_y_pred_sorted[:self.cutoff]

        n_relevant_retrieved = (y[idx_y_pred_sorted] > self.threshold).sum()
        n_relevant = (y > self.threshold).sum()
        recall = float(n_relevant_retrieved) / n_relevant

        return recall


    def __str__(self):
        s = self.name
        if self.cutoff is not None:
            s += "@%d" % self.cutoff
        s += "[>%d]" % self.threshold
        return s