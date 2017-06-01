"""
The :mod:`rankeval.core.metrics` module includes the definition and implementation of
the most common metrics adopted in the Learning to Rank community.
"""

from .metric import Metric
from .precision import Precision

__all__ = ['Metric',
           'Precision']
