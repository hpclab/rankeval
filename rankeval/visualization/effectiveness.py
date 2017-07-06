from __future__ import print_function

"""
This package provides visualizations for several effectiveness analysis focused on assessing
the performance of the models in terms of accuracy. 
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from rankeval.core.dataset import Dataset
from rankeval.core.model import RTEnsemble
from rankeval.core.metrics import Metric


plt.style.use("seaborn-notebook")


def pretty_print_model_performance(performance):
    """
    Prints model performance for each dataset in performance in a tabular form.
    These additional information will be displayed only if working inside a ipython notebook. (?)

    Parameters
    ----------
    performance: xarray.DataArray

    Returns
    -------

    """
    try:
        from IPython.display import display, HTML
        for dataset in performance.coords['dataset'].values:
            display(HTML("<h3>Dateset: %s</h3>" % dataset))
            display(performance.sel(dataset=dataset).to_pandas())
    except ImportError:
        pass


def plot_model_performance(performance, params = plt.rcParams):
    """
    Plots model performance
    I suggest a bar plot for each dataset.
    Parameters
    ----------
    performance

    Returns
    -------
    plt params
    """

    pass


# http://xarray.pydata.org/en/stable/examples/monthly-means.html