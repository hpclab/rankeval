# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This package implements several statistical significance tests.
"""

# Authors: Claudio Lucchese <claudio.lucchese@isti.cnr.it>, Cristina I. Muntean <cristina.muntean@isti.cnr.it>
# License: <TO BE DEFINED>

import numpy as np
import xarray as xr


def randomization(metric_scores_A, metric_scores_B, n_perm=100000):
	"""
	This method computes the randomization test as described in [1].

	Parameters
	----------
	metric_scores_A : numpy array 
		Vector of per-query metric scores for the IR system A.
	metric_scores_B : numpy array 
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
	metric_scores_A_mean = np.mean(metric_scores_A)
	metric_scores_B_mean = np.mean(metric_scores_B)

	best_metrics = metric_scores_A
	worst_metrics = metric_scores_B
	if metric_scores_A_mean < metric_scores_B_mean:
		best_metrics = metric_scores_B
		worst_metrics = metric_scores_A

	difference = np.mean(best_metrics) - np.mean(worst_metrics)
	abs_difference = np.abs(difference)

	p1 = 0.0 # one-sided
	p2 = 0.0 # two-sided
	N = float(len(metric_scores_A))

	# repeat n_prem times
	for i in range(n_perm):
		# select a random subset
		sel = np.random.choice([False, True], len(metric_scores_A))

		A_mean = (np.sum(best_metrics [np.logical_not(sel)]) + np.sum(worst_metrics[sel]) )/N
		B_mean = (np.sum(worst_metrics[np.logical_not(sel)]) + np.sum(best_metrics [sel]) )/N

		delta = A_mean - B_mean

		if delta>=difference:
			p1 += 1.
		if np.abs(delta)>=abs_difference:
			p2 += 1.

	p1 /= n_perm
	p2 /= n_perm


	return (p1,p2)
