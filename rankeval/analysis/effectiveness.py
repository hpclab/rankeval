"""
This package implements several effectiveness analysis focused on assessing
the performance of the models in terms of accuracy. These functionalities can
be applied to several models at the same time, so to have a direct comparison
of the analysis performed.
"""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>

import numpy as np
import pandas as pd
import xarray as xr

from rankeval.core.dataset import Dataset
from rankeval.core.model import RTEnsemble
from rankeval.core.metrics import Metric


def model_performance(datasets=[], models=[], metrics=[], display=False):
    """
    This method implements the model performance analysis (part of the effectiveness analysis category).

    Parameters
    ----------
    datasets : list of Dataset
        The datasets to use for analyzing the behaviour of the model using the given metrics and models
    models : list of RTEnsemble
        The models to analyze
    metrics : list of Metric
        The metrics to use for the analysis
    display : bool
        True if the method has to display interestingly insights using inline plots/tables
        These additional information will be displayed only if working inside a ipython notebook.

    Returns
    -------
    metric_scores : xarray DataArray
        A DataArray containing the metric scores of the models using the given metrics on the given datasets.
    """
    data = np.zeros(shape=(len(datasets), len(models), len(metrics)), dtype=np.float32)
    for idx_dataset, dataset in enumerate(datasets):
        for idx_model, model in enumerate(models):
            scorer = model.score(dataset, detailed=False)
            for idx_metric, metric in enumerate(metrics):
                data[idx_dataset][idx_model][idx_metric] = metric.eval(dataset, scorer.y_pred)[0]

    performance = xr.DataArray(data,
                               name='Model performance',
                               coords=[datasets, models, metrics],
                               dims=['dataset', 'model', 'metric'])

    if display:
        try:
            from IPython.display import display, HTML
            for dataset in performance.coords['dataset'].values:
                display(HTML("<h3>Dateset: %s</h3>" % dataset))
                display(performance.sel(dataset=dataset).to_pandas())
        except ImportError:
            pass

    return performance


def tree_wise_performance(datasets=[], models=[], metrics=[], step=10, display=False):
    """

    Parameters
    ----------
    datasets : list of Dataset
        The datasets to use for analyzing the behaviour of the model using the given metrics and models
    models : list of RTEnsemble
        The models to analyze
    metrics : list of Metric
        The metrics to use for the analysis
    step : int
        Step-size identifying evenly spaced number of trees for evaluating the top=k model performance.
        (e.g., step=100 means the method will evaluate the model performance at 100, 200, 300, etc trees).
    display : bool
        True if the method has to display interestingly insights using inline plots/tables
        These additional information will be displayed only if working inside a ipython notebook.

    Returns
    -------

    """
    def get_tree_steps(model_trees):
        trees = range(step-1, model_trees, step)
        if trees[-1] != model_trees-1:
            trees.append(model_trees-1)
        return np.array(trees)

    max_num_trees = 0
    for model in models:
        if model.n_trees > max_num_trees:
            max_num_trees = model.n_trees

    tree_steps = get_tree_steps(max_num_trees)

    data = np.empty(shape=(len(datasets), len(models), len(tree_steps), len(metrics)), dtype=np.float32)
    data.fill(np.nan)

    for idx_dataset, dataset in enumerate(datasets):
        for idx_model, model in enumerate(models):
            scorer = model.score(dataset, detailed=True)

            # the document scores are accumulated along for the various top-k (in order to avoid useless re-scoring)
            y_pred = np.zeros(dataset.n_instances)

            for idx_top_k, top_k in enumerate(np.arange(start=step-1, stop=model.n_trees, step=step)):

                # compute the document scores using only top-k trees of the model on the given dataset
                for idx_tree in np.arange(start=idx_top_k*step, stop=top_k+1):
                    for idx_instance in np.arange(dataset.n_instances):
                        y_pred[idx_instance] += scorer.partial_y_pred[idx_instance][idx_tree]

                # compute the metric score using the predicted document scores
                for idx_metric, metric in enumerate(metrics):
                    metric_score, _ = metric.eval(dataset, y_pred)
                    data[idx_dataset][idx_model][idx_top_k][idx_metric] = metric_score

    performance = xr.DataArray(data,
                               name='Tree-Wise performance',
                               coords=[datasets, models, tree_steps+1, metrics],
                               dims=['dataset', 'model', 'k', 'metric'])
    return performance
