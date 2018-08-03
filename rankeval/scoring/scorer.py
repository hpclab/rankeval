# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Salvatore Trani <salvatore.trani@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Class for efficient scoring of an ensemble-based model composed of binary regression trees on a given dataset.
"""

from ..dataset import Dataset
from ._efficient_scoring import basic_scoring, detailed_scoring


class Scorer(object):
    """
    Class for efficient scoring of an ensemble-based model composed of binary regression trees on a given dataset.

    This class can be used for simple or detailed scoring, depending on the mode selected at scoring time.
    The document scores are cached as to avoid useless re-scoring. Thus, calling multiple times the `score` method
    does not involve the scoring activity to be executed again, except for a detailed scoring following a basic scoring.
    Indeed in this situation the scoring has to be repeated as to analyze in depth the scoring behaviour.

    Parameters
    ----------
    model: RTEnsemble
        The model to use for scoring
    dataset: Dataset
        The dataset to use for scoring

    Attributes
    ----------
    model : RTEnsemble
        The model to use for scoring
    dataset : Dataset
        The dataset to use for scoring
    y_pred : numpy array of float
        The predicted scores produced by the given model for each sample of the given dataset X
    partial_y_pred : numpy 2d-array of float
        The predicted score of each tree of the model for each dataset instance

    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        # Save the predicted scores for each dataset instance
        self.y_pred = None

        # Save the partial scores of each tree for each dataset instance
        # (if detailed scoring is True)
        self.partial_y_pred = None

        # Save the leaf id of each tree for each dataset instance
        # (if detailed scoring is True)
        self.out_leaves = None

    def score(self, detailed):
        """

        Parameters
        ----------
        detailed : bool
            True if the class has to performs a detailed scoring, false otherwise

        Returns
        -------
        y : numpy array of float
            the predicted scores produced by the given model for each sample of the given dataset X

        Attributes
        ----------
        self.y : array of float
            The predicted scores of each dataset instance
        """

        # Skip the scoring if it has already been done (return cached results)
        if not detailed and self.y_pred is not None or \
                        detailed and self.out_leaves is not None:
            return self.y_pred

        if detailed:
            self.out_leaves, self.partial_y_pred = \
                detailed_scoring(self.model, self.dataset.X)
            self.y_pred = self.partial_y_pred.sum(axis=1)
        else:
            self.y_pred = basic_scoring(self.model, self.dataset.X)

        return self.y_pred

    def get_predicted_scores(self):
        """
        Provide an accessor to the predicted scores produced by the given model for each sample of the given dataset X

        Returns
        -------
        scores : numpy array of float
            The predicted scores produced by the given model for each sample of the given dataset X

        """
        if self.y_pred is None:
            self.score(detailed=False)
        return self.y_pred

    def get_partial_predicted_scores(self):
        """
        Provide an accessor to the partial scores produced by the given model
        for each sample of the given dataset X. Each partial score reflects the
        score produced by a single tree of the ensemble model to a single
        dataset instance. Thus, the returned numpy matrix has a shape of
        (n_instances, n_trees). The partial scores does not take into account
        the tree weights, thus for producing the final score is needed to
        multiply each row for the tree weight vector.

        Returns
        -------
        scores : numpy 2d-array of float
            The predicted score of each tree of the model for each dataset instance
        """
        if self.partial_y_pred is None:
            self.score(detailed=True)
        return self.partial_y_pred

    def get_predicted_leaves(self):
        """
        Provide an accessor to the leaves that identify the exit nodes of each
        sample of the given dataset X using the given model.

        Each leaf value reflects the output node of a single tree of the
        ensemble model to a single dataset instance. Thus, the returned numpy
        matrix has a shape of (n_instances, n_trees).

        Returns
        -------
        scores : numpy 2d-array of int
            The leaves predicted by each tree of the model on scoring
            each dataset instance.

        """
        if self.self.y_leaves is None:
            self.score(detailed=True)
        return self.self.y_leaves

