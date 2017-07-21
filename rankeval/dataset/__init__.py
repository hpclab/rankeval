# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
The :mod:`rankeval.dataset` module includes utilities to load datasets
and dump datasets according to several supported formats.
"""

from .dataset import Dataset
from .dataset_container import DatasetContainer

__all__ = ['Dataset',
           'DatasetContainer']