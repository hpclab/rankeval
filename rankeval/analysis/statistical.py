# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Claudio Lucchese <claudio.lucchese@isti.cnr.it>, Cristina I. Muntean <cristina.muntean@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This package implements several statistical significance tests.
"""

import numpy as np
import xarray as xr


def statistical_significance(datasets, model_a, model_b, metrics, n_perm=100000):
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

    Returns
    -------
    stat_sig : xarray.DataArray
        A DataArray containing the statistical significance of the performance difference
        between any pair of models on the given dataset,
    """

    data = np.zeros(shape=(len(datasets), len(metrics), 2), dtype=np.float32)
    for idx_dataset, dataset in enumerate(datasets):
        scorer_a = model_a.score(dataset, detailed=False)
        scorer_b = model_b.score(dataset, detailed=False)
        for idx_metric, metric in enumerate(metrics):
            metrics_a = metric.eval(dataset, scorer_a.y_pred)[1]
            metrics_b = metric.eval(dataset, scorer_b.y_pred)[1]

            p1, p2 = _randomization(metrics_a, metrics_b, n_perm=n_perm)

            data[idx_dataset][idx_metric][0] = p1
            data[idx_dataset][idx_metric][1] = p2

    performance = xr.DataArray(data,
                               name='Statistical Significance',
                               coords=[datasets, metrics, ['one-sided', 'two-sided']],
                               dims=['dataset', 'metric', '1-2 sided'])

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

    p1 /= n_perm
    p2 /= n_perm

    return p1, p2
