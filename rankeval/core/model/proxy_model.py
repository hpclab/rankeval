"""Base class providing the proxy implementation for loading/saving a model from/to file."""

# Author: Salvatore Trani <salvatore.trani@isti.cnr.it>
# License: <TO DEFINE>
from rankeval.core.model.proxy_quickrank import ProxyQuickRank


class ProxyModel(object):
    """
    Base class providing the skeleton implementation for loading/storing a model from/to file.

    Depending from the format parameter given, it call the right sublcass method
    """

    @staticmethod
    def load(file_path, format="quickrank"):
        """
        Load the model from the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has been saved
        format : str
            The format of the model to load

        Returns
        -------
        model : RegressionTreeEnsemble
            The loaded model as a RegressionTreeEnsemble object
        """
        if format == "quickrank":
            return ProxyQuickRank.load(file_path)
        else:
            raise TypeError("Model format %s not yet supported!" % format)

    @staticmethod
    def save(file_path, format="quickrank"):
        """
        Save the model onto the file identified by file_path.

        Parameters
        ----------
        file_path : str
            The path to the filename where the model has to be saved
        format : str
            The format of the model to load

        Returns
        -------
        status : bool
            Returns true if the save is successful, false otherwise
        """
        if format == "quickrank":
            return ProxyQuickRank.save(file_path)
        else:
            raise TypeError("Model format %s not yet supported!" % format)
