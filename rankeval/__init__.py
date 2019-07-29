"""
This package serve the root of the RankEval package.
"""

import os
import io

cur_dir = os.path.dirname(__file__)

__version__ = io.open(os.path.join(cur_dir, 'VERSION'),
                      encoding='utf-8').read().strip()


def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False      # Probably standard Python interpreter
