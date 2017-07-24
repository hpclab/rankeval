# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
The :mod:`rankeval.metrics` module includes the definition and implementation of
the most common metrics adopted in the Learning to Rank community.
"""

from .metric import Metric
from .precision import Precision
from .recall import Recall
from .ndcg import NDCG
from .dcg import DCG
from .err import ERR
from .kendall_tau import Kendalltau
from .map import MAP
from .mrr import MRR
from .pfound import Pfound
from .rbp import RBP
from .mse import MSE
from .rmse import RMSE
from .spearman_rho import SpearmanRho

__all__ = ['Metric',
           'Precision',
           'Recall',
           'NDCG',
           'DCG',
           'ERR',
           'Kendalltau',
           'MAP',
           'MRR',
           'Pfound',
           'RBP',
           'MSE',
           'RMSE',
           'SpearmanRho']
