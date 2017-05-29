"""
The :mod:`rankeval.core.model` module includes utilities to load a model
and dump it according to several supported model's format.
"""

from .rt_ensemble import RTEnsemble
from .proxy_quickrank import ProxyQuickRank

__all__ = ['RTEnsemble',
           'ProxyQuickRank']
