"""
This package serve the root of the RankEval package.
"""

import logging

__version__ = '0.1'

class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass

logger = logging.getLogger('rankeval')
if len(logger.handlers) == 0:	# To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())