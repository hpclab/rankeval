"""Class for efficient scoring of an ensemble-based model composed of binary regression trees on a given dataset."""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>

import numpy as np

from rankeval.core.model.regression_trees_ensemble import RegressionTreesEnsemble
from rankeval.core.dataset import Dataset
from rankeval.core.scoring._efficient_scoring import basic_scoring, detailed_scoring


class Scorer(object):
    """
    Class for efficient scoring of an ensemble-based model composed of binary regression trees on a given dataset.

    Notes
    ----------
    This class can be used for a simple or detailed scoring, depending on the mode selected at initialization time

    Parameters
    ----------
    model: RegressionTreesEnsemble
        The model to use for scoring
    dataset: Dataset
        The dataset to use for scoring

    Attributes
    ----------
    model : RegressionTreesEnsemble
        The model to use for scoring
    dataset : Dataset
        The dataset to use for scoring
    y : numpy array of float
        The predicted scores produced by the given model for each sample of the given dataset X
    partial_y : numpy 2d-array of float
        The predicted score of each tree of the model for each dataset instance

    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        # Save the predicted scores for each dataset instance
        self.y = None

        # Save the partial scores of each tree for each dataset instance (if detailed scoring is True)
        self.partial_y = None

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

        # Skip the scoring if it has already been computed (return cached results)
        if not detailed and self.y is not None or detailed and self.partial_y is not None:
            return self.y

        if detailed:
            self.partial_y = detailed_scoring(self.model, self.dataset.X)
            self.y = self.partial_y.sum(axis=1)
        else:
            self.y = basic_scoring(self.model, self.dataset.X)

        return self.y

    def get_predicted_scores(self):
        """
        Provide an accessor to the predicted scores produced by the given model for each sample of the given dataset X

        Returns
        -------
        scores : numpy array of float
            The predicted scores produced by the given model for each sample of the given dataset X

        """
        if self.y is None:
            self.score(detailed=False)
        return self.y

    def get_partial_scores(self):
        """
        Provide an accessor to the partial scores produced by the given model for each sample of the given dataset X.
        Each partial score reflects the score produced by a single tree of the ensemble model to a single dataset
        instance. Thus, the returned numpy matrix has a shape of (n_instances, n_trees). The partial scores does not
        take into account the tree weights, thus for producing the final score is needed to multiply each row for the
        tree weight vector.

        Returns
        -------
        scores : numpy 2d-array of float
            The predicted score of each tree of the model for each dataset instance

        """
        if self.partial_y is None:
            self.score(detailed=True)
        return self.partial_y
