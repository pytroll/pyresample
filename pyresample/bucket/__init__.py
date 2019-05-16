# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2019  Pytroll developers
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Code for resampling using bucket resampling.

Bucket resampling is useful for calculating averages and hit-counts
when aggregating data to coarser scale grids.
"""

import dask.array as da
import xarray as xr
import numpy as np
from pyresample._spatial_mp import Proj


def round_to_resolution(arr, resolution):
    """Round the values in *arr* to closest resolution element.

    Parameters
    ----------
    arr : list, tuple, Numpy or Dask array
        Array to be rounded
    resolution : float
        Resolution unit to which data are rounded

    Returns
    -------
    data : Numpy or Dask array
        Source data rounded to the closest resolution unit
    """
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    return resolution * np.round(arr / resolution)


def _get_proj_coordinates(lons, lats, x_res, y_res, prj):
    """Calculate projection coordinates and round them to the closest
    resolution unit.

    Parameters
    ----------
    lons : Numpy or Dask array
        Longitude coordinates
    lats : Numpy or Dask array
        Latitude coordinates
    x_res : float
        Resolution of the output in X direction
    y_res : float
        Resolution of the output in Y direction
    prj : pyproj Proj object
        Object defining the projection transformation

    Returns
    -------
    data : Numpy or Dask array
        Stack of rounded projection coordinates in X- and Y direction
    """
    proj_x, proj_y = prj(lons, lats)
    proj_x = round_to_resolution(proj_x, x_res)
    proj_y = round_to_resolution(proj_y, y_res)

    return np.stack((proj_x, proj_y))


def get_bucket_indices(adef, lons, lats):
    """Calculate projection indices.

    Parameters
    ----------
    adef : AreaDefinition
        Definition of the output area.
    lons : Numpy or Dask array
        Longitude coordinates of the input data
    lats : Numpy or Dask array
        Latitude coordinates of the input data

    Returns
    -------
    x_idxs : Dask array
        X indices of the target grid where the data are put
    y_idxs : Dask array
        Y indices of the target grid where the data are put

    """

    prj = Proj(adef.proj_dict)

    lons = lons.ravel()
    lats = lats.ravel()

    # Create output grid coordinates in projection units
    x_res = (adef.area_extent[2] - adef.area_extent[0]) / adef.width
    y_res = (adef.area_extent[3] - adef.area_extent[1]) / adef.height
    x_vect = da.arange(adef.area_extent[0] + x_res / 2.,
                       adef.area_extent[2] - x_res / 2., x_res)
    # Orient so that 0-meridian is pointing down
    y_vect = da.arange(adef.area_extent[3] - y_res / 2.,
                       adef.area_extent[1] + y_res / 2.,
                       -y_res)

    result = da.map_blocks(_get_proj_coordinates, lons, lats, x_res, y_res,
                           prj, new_axis=0,
                           chunks=(2,) + lons.chunks)
    proj_x = result[0, :]
    proj_y = result[1, :]

    # Calculate array indices
    x_idxs = ((proj_x - np.min(x_vect)) / x_res).astype(np.int)
    y_idxs = ((np.max(y_vect) - proj_y) / y_res).astype(np.int)

    # Get valid index locations
    idxs = ((x_idxs >= 0) & (x_idxs < adef.width) &
            (y_idxs >= 0) & (y_idxs < adef.height))
    y_idxs = da.where(idxs, y_idxs, -1)
    x_idxs = da.where(idxs, x_idxs, -1)

    return x_idxs, y_idxs


def get_sum_from_bucket_indices(data, x_idxs, y_idxs, target_shape):
    """Calculate sums for each bin with drop-in-a-bucket resampling.

    Parameters
    ----------
    data : Numpy or Dask array
        Data to be resampled and summed
    x_idxs : Numpy or Dask array
        X indices of the target array for each data point
    y_idxs : Numpy or Dask array
        Y indices of the target array for each data point
    target_shape : tuple
        Shape of the target grid

    Returns
    -------
    data : Numpy or Dask array
        Bin-wise sums in the target grid
    """

    if isinstance(data, xr.DataArray):
        data = data.data
    data = data.ravel()
    # Remove NaN values from the data when used as weights
    weights = da.where(np.isnan(data), 0, data)

    # Convert X- and Y-indices to raveled indexing
    idxs = y_idxs * target_shape[1] + x_idxs
    # idxs = idxs.ravel()

    out_size = target_shape[0] * target_shape[1]

    # Calculate the sum of the data falling to each bin
    sums, _ = da.histogram(idxs, bins=out_size, range=(0, out_size),
                           weights=weights, density=False)
    return sums.reshape(target_shape)


def get_count_from_bucket_indices(x_idxs, y_idxs, target_shape):
    """Count the number of occurances for each bin using drop-in-a-bucket
    resampling.

    Parameters
    ----------
    x_idxs : Numpy or Dask array
        X indices of the target array for each data point
    y_idxs : Numpy or Dask array
        Y indices of the target array for each data point
    target_shape : tuple
        Shape of the target grid

    Returns
    -------
    data : Dask array
        Bin-wise count of hits for each target grid location
    """

    # Convert X- and Y-indices to raveled index
    idxs = y_idxs * target_shape[1] + x_idxs

    out_size = target_shape[0] * target_shape[1]

    # Calculate the sum of the data falling to each bin
    counts, _ = da.histogram(idxs, bins=out_size, range=(0, out_size))

    return counts.reshape(target_shape)


def resample_bucket_average(adef, data, lons, lats,
                            fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Calculate bin-averages using bucket resampling.

    Parameters
    ----------
    adef : AreaDefinition
        Definition of the target area
    data : Numpy or Dask array
        Data to be binned and averaged
    lons : Numpy or Dask array
        Longitude coordinates of the input data
    lats : Numpy or Dask array
        Latitude coordinates of the input data
    fill_value : float
        Fill value to replace missing values.  Default: np.nan
    x_idxs : Numpy or Dask array
        Pre-calculated resampling indices for X dimension. Optional.
    y_idxs : Numpy or Dask array
        Pre-calculated resampling indices for Y dimension. Optional.

    Returns
    -------
    average : Dask array
        Binned and averaged data.
    """
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    sums = get_sum_from_bucket_indices(data, x_idxs, y_idxs, adef.shape)
    counts = get_count_from_bucket_indices(x_idxs, y_idxs, adef.shape)

    average = sums / counts
    average = da.where(counts == 0, fill_value, average)

    return average


def resample_bucket_fractions(adef, data, lons, lats, categories=None,
                              fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Get fraction of occurences for each given categorical value.

    Parameters
    ----------
    adef : AreaDefinition
        Definition of the target area
    data : Numpy or Dask array
        Categorical data to be processed
    lons : Numpy or Dask array
        Longitude coordinates of the input data
    lats : Numpy or Dask array
        Latitude coordinates of the input data
    categories : iterable or None
        One dimensional list of categories in the data, or None.  If None, 
        categories are determined from the data.
    fill_value : float
        Fill value to replace missing values.  Default: np.nan
    x_idxs : Numpy or Dask array
        Pre-calculated resampling indices for X dimension. Optional.
    y_idxs : Numpy or Dask array
        Pre-calculated resampling indices for Y dimension. Optional.
    
    """
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    if categories is None:
        categories = da.unique(data)
    results = {}
    counts = get_count_from_bucket_indices(x_idxs, y_idxs, adef.shape)
    counts = counts.astype(float)
    for cat in categories:
        cat_data = da.where(data == cat, 1.0, 0.0)

        sums = get_sum_from_bucket_indices(cat_data, x_idxs, y_idxs, adef.shape)
        results[cat] = sums.astype(float) / counts

    return results
