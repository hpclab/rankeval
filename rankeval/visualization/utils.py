
"""
This package provides some basic visualization utilities. 
"""

from pandas import DataFrame, MultiIndex

def xarray3D_to_dataframe(data):
    """
    Build 3D xarray as a multi-index pandas dataframe.

    Parameters
    ----------
    data: 3D xarray.DataArray

    Returns
    -------
    frame: pandas.DataFrame

    """
    import pandas as pd

    frame = DataFrame( data.values.reshape(-1,), 
        columns=[data.dims[-1]],
        index=MultiIndex.from_product( [ [str(z) for z in x.values] for x in data.coords.values() ] ))

    return frame
