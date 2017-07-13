# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
The :mod:`rankeval.core.model` module includes utilities to load a model
and dump it according to several supported model's format.
"""

from rankeval.core.model.rt_ensemble import RTEnsemble
from rankeval.core.model.proxy_QuickRank import ProxyQuickRank
from rankeval.core.model.proxy_LightGBM import ProxyLightGBM
from rankeval.core.model.proxy_XGBoost import ProxyXGBoost
from rankeval.core.model.proxy_ScikitLearn import ProxyScikitLearn

__all__ = ['RTEnsemble',
           'ProxyQuickRank',
           'ProxyLightGBM',
           'ProxyXGBoost',
           'ProxyScikitLearn']
