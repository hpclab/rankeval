"""
This package serve the root of the RankEval package.
"""

import os
import io

cur_dir = os.path.dirname(__file__)

__version__ = io.open(os.path.join(cur_dir, 'VERSION'),
                      encoding='utf-8').read().strip()
