#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2020 Pyresample developers.
#
# This file is part of Pyresample
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Code for resampling using bilinear algorithm for irregular grids.

The algorithm is taken from

http://www.ahinson.com/algorithms_general/Sections/InterpolationRegression/InterpolationIrregularBilinear.pdf

"""

import warnings

import numpy as np
from pykdtree.kdtree import KDTree

from ._base import BilinearBase, _resample, find_indices_outside_min_and_max, get_slicer


def resample_bilinear(data, source_geo_def, target_area_def, radius=50e3,
                      neighbours=32, nprocs=1, fill_value=0,
                      reduce_data=True, segments=None, epsilon=0):
    """Resample using bilinear interpolation.

    data : numpy array
        Array of single channel data points or
        (source_geo_def.shape, k) array of k channels of datapoints
    source_geo_def : object
        Geometry definition of source data
    target_area_def : object
        Geometry definition of target area
    radius : float, optional
        Cut-off distance in meters
    neighbours : int, optional
        Number of neighbours to consider for each grid point when
        searching the closest corner points
    nprocs : int, optional
        Number of processor cores to be used for getting neighbour info
    fill_value : {int, None}, optional
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned with undetermined
        pixels masked
    reduce_data : bool, optional
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    segments : int or None
        Number of segments to use when resampling.
        If set to None an estimate will be calculated
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time

    Returns
    -------
    data : numpy array
        Source data resampled to target geometry

    """
    if nprocs > 1:
        from pyresample._spatial_mp import cKDTree_MP as kdtree_class
    else:
        kdtree_class = KDTree

    # Deprecation warning is suppressed outside __main__ by default, so use FutureWarning
    warnings.warn(
        "Usage of resample_bilinear() is deprecated, please use NumpyResamplerBilinear class instead",
        FutureWarning, stacklevel=2)

    resampler = NumpyBilinearResampler(
        source_geo_def,
        target_area_def,
        radius,
        neighbours=neighbours,
        epsilon=epsilon,
        reduce_data=reduce_data
    )
    resampler.get_bil_info(kdtree_class=kdtree_class, nprocs=nprocs)
    result = resampler.get_sample_from_bil_info(data, fill_value=fill_value, output_shape=None)

    return result


def get_sample_from_bil_info(data, t__, s__, input_idxs, idx_arr,
                             output_shape=None):
    """Resample data using bilinear interpolation.

    Parameters
    ----------
    data : numpy array
        1d array to be resampled
    t__ : numpy array
        Vertical fractional distances from corner to the new points
    s__ : numpy array
        Horizontal fractional distances from corner to the new points
    input_idxs : numpy array
        Valid indices in the input data
    idx_arr : numpy array
        Mapping array from valid source points to target points
    output_shape : tuple, optional
        Tuple of (y, x) dimension for the target projection.
        If None (default), do not reshape data.

    Returns
    -------
    result : numpy array
        Source data resampled to target geometry

    """
    # Deprecation warning is suppressed outside __main__ by default, so use FutureWarning
    warnings.warn(
        "Usage of get_sample_from_bil_info() is deprecated, please use NumpyResamplerBilinear class instead",
        FutureWarning, stacklevel=2)

    # Reduce data
    new_data = data[input_idxs]
    # Add a small "machine epsilon" so that tiny variations are not discarded
    epsilon = 1e-6
    data_min = np.nanmin(new_data) - epsilon
    data_max = np.nanmax(new_data) + epsilon

    new_data = new_data[idx_arr]

    # Get neighbour data to separate variables
    corner_points = (new_data[:, 0], new_data[:, 1], new_data[:, 2], new_data[:, 3])

    result = _resample(corner_points, (s__, t__))

    if hasattr(result, 'mask'):
        mask = result.mask
        result = result.data
        result[mask] = np.nan

    try:
        with np.errstate(invalid='ignore'):
            idxs = find_indices_outside_min_and_max(result, data_min, data_max)
        result[idxs] = np.nan
    except TypeError:
        pass

    if output_shape is not None:
        result = result.reshape(output_shape)

    return result


def get_bil_info(source_geo_def, target_area_def, radius=50e3, neighbours=32,
                 nprocs=1, masked=False, reduce_data=True, segments=None,
                 epsilon=0):
    """Calculate information needed for bilinear resampling.

    source_geo_def : object
        Geometry definition of source data
    target_area_def : object
        Geometry definition of target area
    radius : float, optional
        Cut-off distance in meters
    neighbours : int, optional
        Number of neighbours to consider for each grid point when
        searching the closest corner points
    nprocs : int, optional
        Number of processor cores to be used for getting neighbour info
    masked : bool, optional
        If true, return masked arrays, else return np.nan values for
        invalid points (default)
    reduce_data : bool, optional
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time

    Returns
    -------
    t__ : numpy array
        Vertical fractional distances from corner to the new points
    s__ : numpy array
        Horizontal fractional distances from corner to the new points
    input_idxs : numpy array
        Valid indices in the input data
    idx_arr : numpy array
        Mapping array from valid source points to target points

    """
    if nprocs > 1:
        from pyresample._spatial_mp import cKDTree_MP as kdtree_class
    else:
        kdtree_class = KDTree

    # Deprecation warning is suppressed outside __main__ by default, so use FutureWarning
    warnings.warn(
        "Usage of get_bil_info() is deprecated, please use NumpyResamplerBilinear class instead",
        FutureWarning, stacklevel=2)

    numpy_resampler = NumpyBilinearResampler(
        source_geo_def,
        target_area_def,
        radius,
        neighbours=neighbours,
        epsilon=epsilon,
        reduce_data=reduce_data
    )
    numpy_resampler.get_bil_info(kdtree_class=kdtree_class, nprocs=nprocs)

    return (
        numpy_resampler.bilinear_t,
        numpy_resampler.bilinear_s,
        #  deepcode ignore W0212: Legacy access
        numpy_resampler._valid_input_index,
        #  deepcode ignore W0212: Legacy access
        numpy_resampler._index_array
    )


class NumpyBilinearResampler(BilinearBase):
    """Bilinear interpolation using Numpy."""

    def resample(self, data, fill_value=0, nprocs=1):
        """Resample the given data."""
        if nprocs > 1:
            from pyresample._spatial_mp import cKDTree_MP as kdtree_class
        else:
            kdtree_class = KDTree
        self.get_bil_info(kdtree_class=kdtree_class, nprocs=nprocs)
        return self.get_sample_from_bil_info(data, fill_value=fill_value, output_shape=None)

    def get_sample_from_bil_info(self, data, fill_value=None, output_shape=None):
        """Resample using pre-computed resampling LUTs."""
        del output_shape

        res = _resample(
            self._slice_data(data, fill_value),
            (self.bilinear_s, self.bilinear_t)
        )

        return self._finalize_output_data(data, res, fill_value)

    def _slice_data(self, data, fill_value):
        data = _check_data_shape(data, self._valid_input_index)
        res = get_slicer(data)(data, self.slices_x, self.slices_y, self.mask_slices, fill_value)

        return res

    def _finalize_output_data(self, data, res, fill_value):
        reshaped_res = self._reshape_to_target_area(res, data.ndim)
        return _apply_fill_value_or_mask_data(reshaped_res, fill_value)

    def _reshape_to_target_area(self, res, ndim):
        shp = self._target_geo_def.shape
        if ndim == 3:
            res = np.reshape(res, (res.shape[0],) + shp)
            # Place the "channel" dimension last for backwards compatibility
            res = np.moveaxis(res, 0, -1)
        else:
            res = np.reshape(res, (shp[0], shp[1]))
        res = np.squeeze(res)

        return res


def _apply_fill_value_or_mask_data(result, fill_value):
    if fill_value is None:
        result = np.ma.masked_invalid(result)
    else:
        result[np.isnan(result)] = fill_value

    return np.squeeze(result)


def _check_data_shape(data, input_idxs):
    """Check data shape and adjust if necessary."""
    # Handle multiple datasets
    if data.ndim > 2 and data.shape[0] * data.shape[1] == input_idxs.shape[0]:
        # Move the "channel" dimension first
        data = np.moveaxis(data, -1, 0)

    # Ensure two dimensions
    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    return data


class NumpyResamplerBilinear(NumpyBilinearResampler):
    """Wrapper for the old resampler class."""

    def __init__(self, source_geo_def,
                 target_geo_def,
                 radius_of_influence,
                 **kwargs):
        """Initialize resampler."""
        warnings.warn("Use of NumpyResamplerBilinear is deprecated, use NumpyBilinearResampler instead", stacklevel=2)

        super(NumpyResamplerBilinear, self).__init__(
            source_geo_def,
            target_geo_def,
            radius_of_influence,
            **kwargs)
