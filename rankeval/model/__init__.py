# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
The :mod:`rankeval.model` module includes utilities to load a model
and dump it according to several supported model's format.
"""

from .proxy_LightGBM import ProxyLightGBM
from .proxy_QuickRank import ProxyQuickRank
from .proxy_ScikitLearn import ProxyScikitLearn
from .proxy_XGBoost import ProxyXGBoost
from .proxy_Jforests import ProxyJforests
from .proxy_CatBoost import ProxyCatBoost
from .rt_ensemble import RTEnsemble

__all__ = ['RTEnsemble',
           'ProxyQuickRank',
           'ProxyLightGBM',
           'ProxyXGBoost',
           'ProxyScikitLearn',
           'ProxyJforests',
           'ProxyCatBoost'
]
