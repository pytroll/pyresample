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

import numpy as np
from pyresample._spatial_mp import Proj

def round_to_resolution(arr, resolution):
    """Round the values in *arr* to closest resolution element."""
    return resolution * np.round(arr / resolution)


def get_bucket_indices(adef, lons, lats):
    """Get projection indices"""

    prj = Proj(adef.proj_dict)

    # Create output grid coordinates in projection units
    x_res = (adef.area_extent[2] - adef.area_extent[0]) / adef.width
    y_res = (adef.area_extent[3] - adef.area_extent[1]) / adef.height
    x_vect = np.arange(adef.area_extent[0] + x_res / 2.,
                       adef.area_extent[2] - x_res / 2., x_res)
    # Orient so that 0-meridian is pointing down
    y_vect = np.arange(adef.area_extent[3] - y_res / 2.,
                       adef.area_extent[1] + y_res / 2.,
                       -y_res)

    lons = np.ravel(lons)
    lats = np.ravel(lats)

    # Round coordinates to the closest resolution element
    proj_x, proj_y = prj(lons, lats)
    proj_x = round_to_resolution(proj_x, x_res)
    proj_y = round_to_resolution(proj_y, y_res)

    # Calculate array indices
    x_idxs = ((proj_x - np.min(x_vect)) / x_res).astype(np.int)
    y_idxs = ((np.max(y_vect) - proj_y) / y_res).astype(np.int)

    # Mark invalid array indices with -1
    idxs = ((x_idxs < 0) | (x_idxs >= adef.width) |
            (y_idxs < 0) | (y_idxs >= adef.height))
    y_idxs[idxs] = -1
    x_idxs[idxs] = -1

    return x_idxs, y_idxs


def get_sample_from_bucket_indices(data, x_idxs, y_idxs, target_shape):
    """Resample the given data with drop-in-a-bucket resampling.  Return
    the sum and counts for each bin as two Numpy arrays."""

    sums = np.zeros(target_shape)
    count = np.zeros(target_shape)
    data = np.ravel(data)

    idxs = (x_idxs > 0) & (y_idxs > 0) & np.isfinite(data)
    x_idxs = x_idxs[idxs]
    y_idxs = y_idxs[idxs]
    data = data[idxs]

    # Place the data to proper locations
    for i in range(x_idxs.size):
        x_idx = x_idxs[i]
        y_idx = y_idxs[i]
        #if x_idx < 0 | y_idx < 0 | np.isnan(data[i]):
        #    continue
        sums[y_idx, x_idx] += data[i]
        count[y_idx, x_idx] += 1

    return sums, count


def resample_bucket_average(adef, data, lats, lons,
                            fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Calculate bin-averages using bucket resampling."""
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    sums, counts = get_sample_from_bucket_indices(data, x_idxs, y_idxs,
                                                  adef.shape)
    average = sums / counts
    average = np.where(count == 0, fill_value, average)

    return average


def resample_bucket_counts(adef, data, lats, lons,
                           fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Get number of hits for each bin."""
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    _, counts = get_sample_from_bucket_indices(data, x_idxs, y_idxs,
                                               adef.shape)

    return counts


def resample_bucket_fractions(adef, data, lats, lons, categories,
                              fill_value=np.nan, x_idxs=None, y_idxs=None):
    """Get fraction of occurences for each given categorical value."""
    if x_idxs is None or y_idxs is None:
        x_idxs, y_idxs = get_bucket_indices(adef, lons, lats)
    results = []
    for cat in categories:
        cat_data = np.where(data == cat, 1.0, 0.0)
        sums, counts = get_sample_from_bucket_indices(cat_data, x_idxs, y_idxs,
                                                      adef.shape)
        results.append(sums)

    fractions = [res / counts for res in results]

    return fractions
