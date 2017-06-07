# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Cristina Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
The :mod:`rankeval.core.metrics` module includes the definition and implementation of
the most common metrics adopted in the Learning to Rank community.
"""

from .metric import Metric
from .precision import Precision
from .recall import Recall
from .ndcg import NDCG

__all__ = ['Metric',
           'Precision',
           'Recall',
           'NDCG']
