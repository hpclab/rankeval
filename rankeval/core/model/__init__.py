
"""
The :mod:`rankeval.core.model` module includes utilities to load a model
and dump it according to several supported model's format.
"""

from .proxy_model import ProxyModel
from .regression_tree_ensemble import RegressionTreeEnsemble

__all__ = ['ProxyModel',
           'RegressionTreeEnsemble']
