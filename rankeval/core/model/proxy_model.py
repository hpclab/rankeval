"""Base class providing the skeleton implementation for loading/storing a model from/to file."""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>


class ProxyModel(object):
    """
    Base class providing the skeleton implementation for loading/storing a model from/to file.

    Warning
    ----------
    This class should not be used directly. Use the different subclass implementations supporting the load from the
    most common formats.
    """

    @staticmethod
    def load(file_path):
        """
        Load the model from the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved

        Returns
        -------
        model : RegressionTreesEnsemble
            The loaded model as a RegressionTreesEnsemble object
        """
        raise NotImplementedError("Method not implemented in the base class!")

    @staticmethod
    def save(file_path):
        """
        Save the model onto the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has to be saved

        Returns
        -------
        status : bool
            Returns true if the save is successful, false otherwise
        """
        raise NotImplementedError("Method not implemented in the base class!")
