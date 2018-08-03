# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Claudio Lucchese <claudio.lucchese@isti.cnr.it>, Cristina I. Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This package implements several statistical significance tests.
"""

from __future__ import print_function

import math

import numpy as np
import xarray as xr

from ..dataset import Dataset
from ..metrics.metric import Metric
from rankeval.metrics import MSE

from ipywidgets import IntProgress
from IPython.display import display

def statistical_significance(datasets, model_a, model_b, metrics,
                             n_perm=100000, cache=False):
    """
    This method computes the statistical significance of the performance difference between model_a and.

    Parameters
    ----------
    datasets : list of Dataset
        The datasets to use for analyzing the behaviour of the model using the given metrics and models
    model_a : RTEnsemble
        The first model considered.
    model_b : RTEnsemble
        The second model considered.
    metrics : list of Metric
        The metrics to use for the analysis
    n_perm : int
        Number of permutations for the randomization test.
    cache : bool
        Whether to cache or not the intermediate scoring results of each model
        on each dataset. Caching enable instant access to the scores (after the
        first scoring) but coudl make use also of a huge quantity of memory.

    Returns
    -------
    stat_sig : xarray.DataArray
        A DataArray containing the statistical significance of the performance difference
        between any pair of models on the given dataset.
    """

    progress_bar = IntProgress(min=0, max=len(datasets)*len(metrics), 
                               description="Iterating datasets and metrics")
    display(progress_bar)    

    data = np.zeros(shape=(len(datasets), len(metrics), 2), dtype=np.float32)
    for idx_dataset, dataset in enumerate(datasets):
        y_pred_a = model_a.score(dataset, detailed=False, cache=cache)
        y_pred_b = model_b.score(dataset, detailed=False, cache=cache)
        for idx_metric, metric in enumerate(metrics):
            progress_bar.value += 1

            metrics_a = metric.eval(dataset, y_pred_a)[1]
            metrics_b = metric.eval(dataset, y_pred_b)[1]

            p1, p2 = _randomization(metrics_a, metrics_b, n_perm=n_perm)

            data[idx_dataset][idx_metric][0] = p1
            data[idx_dataset][idx_metric][1] = p2

    progress_bar.bar_style = "success"
    progress_bar.close()

    performance = xr.DataArray(data,
                               name='Statistical Significance',
                               coords=[datasets, metrics, ['one-sided', 'two-sided']],
                               dims=['dataset', 'metric', 'p-value'])

    return performance


def _randomization(metric_scores_a, metric_scores_b, n_perm=100000):
    """
    This method computes the randomization test as described in [1].

    Parameters
    ----------
    metric_scores_a : numpy array
        Vector of per-query metric scores for the IR system A.
    metric_scores_b : numpy array
        Vector of per-query metric scores for the IR system B.
    n_perm : int
        Number of permutations evaluated in the randomization test.

    Returns
    -------
    metric_scores : (float, float)
        A tuple (p-value_1, p-value_2) being respectively the one-sided and two-sided p-values.

    References
    ----------
    .. [1] Smucker, Mark D., James Allan, and Ben Carterette.
        "A comparison of statistical significance tests for information retrieval evaluation."
        In Proceedings of the sixteenth ACM conference on Conference on information and knowledge management, pp. 623-632. ACM, 2007.
    """
    progress_bar = IntProgress(min=0, max=10, description="Randomization Test")
    display(progress_bar)    

    # find the best system
    metric_scores_a_mean = np.mean(metric_scores_a)
    metric_scores_b_mean = np.mean(metric_scores_b)

    best_metrics = metric_scores_a
    worst_metrics = metric_scores_b
    if metric_scores_a_mean < metric_scores_b_mean:
        best_metrics = metric_scores_b
        worst_metrics = metric_scores_a

    difference = np.mean(best_metrics) - np.mean(worst_metrics)
    abs_difference = np.abs(difference)

    p1 = 0.0  # one-sided
    p2 = 0.0  # two-sided
    N = float(len(metric_scores_a))

    a_sum = np.sum(best_metrics)
    b_sum = np.sum(worst_metrics)

    # repeat n_prem times
    for i in range(n_perm):
        if i % (n_perm/10)==0: progress_bar.value+=1
        
        # select a random subset
        sel = np.random.choice([False, True], len(metric_scores_a))

        a_sel_sum = np.sum(best_metrics[sel])
        b_sel_sum = np.sum(worst_metrics[sel])

        # compute avg performance of randomized models
        a_mean = (a_sum - a_sel_sum + b_sel_sum) / N
        b_mean = (b_sum - b_sel_sum + a_sel_sum) / N

        # performance difference
        delta = a_mean - b_mean

        if delta >= difference:
            p1 += 1.
        if np.abs(delta) >= abs_difference:
            p2 += 1.

    progress_bar.bar_style = "success"
    progress_bar.close()

    p1 /= n_perm
    p2 /= n_perm

    return p1, p2


def _kfold_scoring(dataset, k, algo):
    """
    Scored the given datset with the given algo unsing k-fold train/test.

    Parameters
    ----------
    dataset : rankeval.dataset.Dataset
        The dataset instance.
    k : int
        Number of folds.
    algo : function
        See :func:`bias_variance`.

    Returns
    -------
    score : numpy.ndarray
        A vecotr of num_instances scores.
    """
    progress_bar = IntProgress(min=0, max=k, description="Processing k folds")
    display(progress_bar)    

    scores = np.zeros(dataset.n_instances, dtype=np.float32)
    query_sizes = dataset.get_query_sizes()
    # shuffle queries
    shuffled_qid = np.random.permutation(dataset.n_queries)
    chunk_query_size = int(math.ceil(dataset.n_queries/float(k)))
    for p in range(0, dataset.n_queries, chunk_query_size):
        progress_bar.value += 1

        # p-th fold is used for testing
        test_rows = np.full(dataset.n_instances,
                            fill_value=False,
                            dtype=np.bool)
        for q in shuffled_qid[p: p + chunk_query_size]:
            test_rows[dataset.query_offsets[q]:dataset.query_offsets[q+1]] = True
        # other folds are used for training
        train_rows = np.logical_not(test_rows)

        train_q = np.full(dataset.n_queries,
                          fill_value=True,
                          dtype=np.bool)
        train_q[shuffled_qid[p: p+chunk_query_size]] = False

        # get algorithm predictions
        fold_scores = algo(
            dataset.X[train_rows],
            dataset.y[train_rows],
            query_sizes[train_q],
            dataset.X[test_rows]
        )
        # update scores for the current fold
        scores[test_rows] = fold_scores
        
    progress_bar.bar_style = "success"
    progress_bar.close()
        
    return scores


def _multi_kfold_scoring(dataset, algo, L=10, k=2):
    """
    Performs multiple scorings of the given dataset.

    Parameters
    ----------
    dataset : rankeval.dataset.Dataset
        The dataset instance.
    algo : function
        See :func:`bias_variance`.
    L : int
        Number of iterations
    k : int
        Number of folds.

    Returns
    -------
    score : numpy.ndarray
        A matrix num_instances x L.
    """
    progress_bar = IntProgress(min=0, max=L, description="Computing L scores")
    display(progress_bar)    

    scores = np.zeros( (dataset.n_instances, L), dtype=np.float32)

    for l in range(L):
        progress_bar.value += 1

        scores[:,l] = _kfold_scoring(dataset, k, algo)

    progress_bar.bar_style = "success"
    progress_bar.close()
    
    return scores


def bias_variance(datasets=[], algos=[], metrics=[], L=10, k=2):
    """
    This method computes the bias vs. variance decomposition of the error.
    The approach used here is based on the works of [Webb05]_ and [Dom05]_.

    Each instance of the dataset is scored `L` times.
    A single scoring is achieved by splitting the dataset at random into
    `k` folds. Each fold is scored by the model `M` trained on the remainder folds.
    [Webb05]_ recommends the use of 2 folds.

    If metric is MSE then the standard decomposition is used.
    The Bias for and instance `x` is defined as mean squared error of the `L` trained models
    w.r.t. the true label `y`, denoted with :math:`{\\sf E}_{L} [M(x) - y]^2`. 
    The Variance for an instance `x` is measured across the `L` trained models: 
    :math:`{\\sf E}_{L} [M(x) - {\\sf E}_{L} M(x)]^2`. 
    Both are averaged over all instances in the dataset.

    If metric is any of the IR quality measures, we resort to the bias variance
    decomposition of the mean squared error of the given metric w.r.t. its ideal value,
    e.g., for the case of NDCG, :math:`{\\sf E}_{L} [1 - NDCG]^2`. 
    Recall that, a formal Bias/Variance decomposition was not proposed yet.


    Parameters
    ----------
    dataset : rankeval.dataset.Dataset
        The dataset instance.
    algo : function
        This should be a wrapper of learning algorithm.
        The function should accept four parameters: `train_X`, `train_Y`, `train_q`, `test_X`.
            - `train_X`: numpy.ndarray storing a 2-D matrix of size num_docs x num_features
            - `train_Y`: numpy.ndarray storing a vector of document's relevance labels
            - `train_q`: numpy.ndarray storing a vector of query lengths
            - `test_X`: numpy.ndarray as for `train_X`
        
        A model is trained on `train_X`, `train_Y`, `train_q`, and used to score `test_X`.
        An numpy.ndarray with such score must be returned.
    metric : "mse" or rankeval.metrics.metric.Metric
        The metric used to compute the error.
    L : int
        Number of iterations
    k : int
        Number of folds.

    Returns
    -------
    bias_variance : xarray.DataArray
        A DataArray containing the bias/variance decomposition of the error
        for any given dataset, algorithm and metric.

    References
    ----------
    .. [Webb05] Webb, Geoffrey I., and Paul Conilione. "Estimating bias and variance from data." 
            Pre-publication manuscript (`pdf <http://www.csse.monash.edu/webb/-Files/WebbConilione06.pdf>`_) (2005).
    .. [Dom05] Domingos P. A unified bias-variance decomposition. 
            In Proceedings of 17th International Conference on Machine Learning 2000 (pp. 231-238).
    """
    assert(k>=2)
    assert(L>=2)
    assert(len(datasets)>0)
    assert(len(metrics)>0)
    for metric in metrics:
        assert isinstance(metric, Metric)

    progress_bar = IntProgress(min=0, max=len(datasets)*len(metrics)*len(algos),
                               description="Iterating datasets and metrics")
    display(progress_bar)    

    data = np.zeros(shape=(len(datasets), len(metrics), len(algos), 3), dtype=np.float32)
    for idx_dataset, dataset in enumerate(datasets):
        for idx_algo, algo in enumerate(algos):
            for idx_metric, metric in enumerate(metrics):
                progress_bar.value += 1
                
                scores = _multi_kfold_scoring(dataset, algo=algo, L=L, k=k)
                
                avg_error = 0.
                avg_bias = 0.
                avg_var = 0.
                if not isinstance(metric, MSE):
                    # mse over metric, assume error is 1-metric
                    # not exactly domingos paper
                    q_scores = np.empty((dataset.n_queries, L), dtype=np.float32) 
                    for i in range(L):
                        q_scores[:,i] = metric.eval(dataset=dataset, y_pred=scores[:,i])[1]            
                    avg_error = np.mean( (q_scores-1.)**2. )
                    avg_pred  = np.mean(q_scores, axis=1)
                    avg_bias  = np.mean((avg_pred - 1.)**2.)
                    avg_var   = np.mean( (q_scores-avg_pred.reshape((-1,1)))**2. )
                else:
                    # mse
                    avg_error = np.mean( (scores-dataset.y.reshape((-1,1)))**2. )
                    avg_pred  = np.mean(scores, axis=1)
                    avg_bias  = np.mean((avg_pred - dataset.y)**2.)
                    avg_var   = np.mean( (scores-avg_pred.reshape((-1,1)))**2. )

                data[idx_dataset][idx_metric][idx_algo][0] = avg_error
                data[idx_dataset][idx_metric][idx_algo][1] = avg_bias
                data[idx_dataset][idx_metric][idx_algo][2] = avg_var
                

    progress_bar.bar_style = "success"
    progress_bar.close()

    performance = xr.DataArray(data,
                               name='Bias/Variance Decomposition',
                               coords=[datasets, metrics, [a.__name__ for a in algos], 
                               ['Error', 'Bias', 'Variance']],
                               dims=['dataset', 'metric', 'algo', 'error'])

    return performance