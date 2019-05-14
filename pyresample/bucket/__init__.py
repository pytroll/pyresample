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
    """Round the values in *arr* to closest resolution element."""
    return resolution * np.round(arr / resolution)


def _get_proj_coordinates(lons, lats, x_res, y_res, prj):
    """Calculate projection coordinates and round them to the closest
    resolution unit."""
    proj_x, proj_y = prj(lons, lats)
    proj_x = round_to_resolution(proj_x, x_res)
    proj_y = round_to_resolution(proj_y, y_res)

    return np.stack((proj_x, proj_y))


def get_bucket_indices(adef, lons, lats):
    """Get projection indices"""

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
    """Resample the given data with drop-in-a-bucket resampling.  Return
    the sum and counts for each bin as two Numpy arrays."""

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
    """Resample the given data with drop-in-a-bucket resampling.  Return
    the sum and counts for each bin as two Numpy arrays."""

    # Convert X- and Y-indices to raveled index
    idxs = y_idxs * target_shape[1] + x_idxs
    # idxs = idxs.ravel()

    out_size = target_shape[0] * target_shape[1]

    # Calculate the sum of the data falling to each bin
    counts, _ = da.histogram(idxs, bins=out_size, range=(0, out_size))

    return counts.reshape(target_shape)


def resample_bucket_average(adef, data, lats, lons,
                            fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Calculate bin-averages using bucket resampling."""
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    sums = get_sum_from_bucket_indices(data, x_idxs, y_idxs, adef.shape)
    counts = get_count_from_bucket_indices(x_idxs, y_idxs, adef.shape)

    average = sums / counts
    average = da.where(counts == 0, fill_value, average)

    return average


def resample_bucket_fractions(adef, data, lats, lons, categories,
                              fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Get fraction of occurences for each given categorical value."""
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    results = []
    for cat in categories:
        cat_data = da.where(data == cat, 1.0, 0.0)

        sums = get_sum_from_bucket_indices(cat_data, x_idxs, y_idxs, adef.shape)
        results.append(sums)

    counts = get_count_from_bucket_indices(x_idxs, y_idxs, adef.shape)
    fractions = [res / counts for res in results]

    return fractions
