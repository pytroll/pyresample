#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017

# Author(s):

#   Panu Lahtinen <panu.lahtinen@fmi.fi>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Code for resampling using bilinear algorithm for irregular grids.

The algorithm is taken from

http://www.ahinson.com/algorithms_general/Sections/InterpolationRegression/InterpolationIrregularBilinear.pdf

"""

import numpy as np
from pyresample._spatial_mp import Proj
import warnings

from pyresample import kd_tree


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

    # Calculate the resampling information
    t__, s__, input_idxs, idx_ref = get_bil_info(source_geo_def,
                                                 target_area_def,
                                                 radius=radius,
                                                 neighbours=neighbours,
                                                 nprocs=nprocs,
                                                 masked=False,
                                                 reduce_data=reduce_data,
                                                 segments=segments,
                                                 epsilon=epsilon)

    data = _check_data_shape(data, input_idxs)

    result = np.nan * np.zeros((target_area_def.size, data.shape[1]))
    for i in range(data.shape[1]):
        result[:, i] = get_sample_from_bil_info(data[:, i], t__, s__,
                                                input_idxs, idx_ref,
                                                output_shape=None)

    if fill_value is None:
        result = np.ma.masked_invalid(result)
    else:
        result[np.isnan(result)] = fill_value

    # Reshape to target area shape
    shp = target_area_def.shape
    result = result.reshape((shp[0], shp[1], data.shape[1]))
    # Remove extra dimensions
    result = np.squeeze(result)

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

    # Reduce data
    new_data = data[input_idxs]
    # Add a small "machine epsilon" so that tiny variations are not discarded
    epsilon = 1e-6
    data_min = np.nanmin(new_data) - epsilon
    data_max = np.nanmax(new_data) + epsilon

    new_data = new_data[idx_arr]

    # Get neighbour data to separate variables
    p_1 = new_data[:, 0]
    p_2 = new_data[:, 1]
    p_3 = new_data[:, 2]
    p_4 = new_data[:, 3]

    result = (p_1 * (1 - s__) * (1 - t__) +
              p_2 * s__ * (1 - t__) +
              p_3 * (1 - s__) * t__ +
              p_4 * s__ * t__)

    if hasattr(result, 'mask'):
        mask = result.mask
        result = result.data
        result[mask] = np.nan

    try:
        with np.errstate(invalid='ignore'):
            idxs = (result > data_max) | (result < data_min)
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
    segments : int or None
        Number of segments to use when resampling.
        If set to None an estimate will be calculated
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

    # Check source_geo_def
    # if isinstance(source_geo_def, tuple):
    #     from pyresample.geometry import SwathDefinition
    #     lons, lats = _mask_coordinates(source_geo_def[0], source_geo_def[1])
    #     source_geo_def = SwathDefinition(lons, lats)

    # Calculate neighbour information
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (input_idxs, output_idxs, idx_ref, dists) = \
            kd_tree.get_neighbour_info(source_geo_def, target_area_def,
                                       radius, neighbours=neighbours,
                                       nprocs=nprocs, reduce_data=reduce_data,
                                       segments=segments, epsilon=epsilon)

    del output_idxs, dists

    # Reduce index reference
    input_size = input_idxs.sum()
    index_mask = (idx_ref == input_size)
    idx_ref = np.where(index_mask, 0, idx_ref)

    # Get output projection as pyproj object
    proj = Proj(target_area_def.proj_str)

    # Get output x/y coordinates
    out_x, out_y = _get_output_xy(target_area_def, proj)

    # Get input x/y coordinates
    in_x, in_y = _get_input_xy(source_geo_def, proj, input_idxs, idx_ref)

    # Get the four closest corner points around each output location
    pt_1, pt_2, pt_3, pt_4, idx_ref = \
        _get_bounding_corners(in_x, in_y, out_x, out_y, neighbours, idx_ref)

    # Calculate vertical and horizontal fractional distances t and s
    t__, s__ = _get_ts(pt_1, pt_2, pt_3, pt_4, out_x, out_y)

    # Mask NaN values
    if masked:
        mask = np.isnan(t__) | np.isnan(s__)
        t__ = np.ma.masked_where(mask, t__)
        s__ = np.ma.masked_where(mask, s__)

    return t__, s__, input_idxs, idx_ref


def _get_ts(pt_1, pt_2, pt_3, pt_4, out_x, out_y):
    """Calculate vertical and horizontal fractional distances t and s"""

    # General case, ie. where the the corners form an irregular rectangle
    t__, s__ = _get_ts_irregular(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Cases where verticals are parallel
    idxs = np.isnan(t__) | np.isnan(s__)
    # Remove extra dimensions
    idxs = idxs.ravel()

    if np.any(idxs):
        t__[idxs], s__[idxs] = \
            _get_ts_uprights_parallel(pt_1[idxs, :], pt_2[idxs, :],
                                      pt_3[idxs, :], pt_4[idxs, :],
                                      out_y[idxs], out_x[idxs])

    # Cases where both verticals and horizontals are parallel
    idxs = np.isnan(t__) | np.isnan(s__)
    # Remove extra dimensions
    idxs = idxs.ravel()
    if np.any(idxs):
        t__[idxs], s__[idxs] = \
            _get_ts_parallellogram(pt_1[idxs, :], pt_2[idxs, :], pt_3[idxs, :],
                                   out_y[idxs], out_x[idxs])

    with np.errstate(invalid='ignore'):
        idxs = (t__ < 0) | (t__ > 1) | (s__ < 0) | (s__ > 1)
    t__[idxs] = np.nan
    s__[idxs] = np.nan

    return t__, s__


def _get_ts_irregular(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where none of the sides are parallel."""

    # Get parameters for the quadratic equation
    a__, b__, c__ = _calc_abc(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    t__ = _solve_quadratic(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter s
    s__ = _solve_another_fractional_distance(t__, pt_1[:, 1], pt_3[:, 1],
                                             pt_2[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _get_ts_uprights_parallel(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where uprights are parallel"""

    # Get parameters for the quadratic equation
    a__, b__, c__ = _calc_abc(pt_1, pt_3, pt_2, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    s__ = _solve_quadratic(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter t
    t__ = _solve_another_fractional_distance(s__, pt_1[:, 1], pt_2[:, 1],
                                             pt_3[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _get_ts_parallellogram(pt_1, pt_2, pt_3, out_y, out_x):
    """Get parameters for the case where uprights are parallel"""

    # Pairwise longitudal separations between reference points
    x_21 = pt_2[:, 0] - pt_1[:, 0]
    x_31 = pt_3[:, 0] - pt_1[:, 0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[:, 1] - pt_1[:, 1]
    y_31 = pt_3[:, 1] - pt_1[:, 1]

    t__ = (x_21 * (out_y - pt_1[:, 1]) - y_21 * (out_x - pt_1[:, 0])) / \
          (x_21 * y_31 - y_21 * x_31)
    with np.errstate(invalid='ignore'):
        idxs = (t__ < 0.) | (t__ > 1.)
    t__[idxs] = np.nan

    s__ = (out_x - pt_1[:, 0] + x_31 * t__) / x_21

    with np.errstate(invalid='ignore'):
        idxs = (s__ < 0.) | (s__ > 1.)
    s__[idxs] = np.nan

    return t__, s__


def _solve_another_fractional_distance(f__, y_1, y_2, y_3, y_4, out_y):
    """Solve parameter t__ from s__, or vice versa.  For solving s__,
    switch order of y_2 and y_3."""
    y_21 = y_2 - y_1
    y_43 = y_4 - y_3

    with np.errstate(divide='ignore'):
        g__ = ((out_y - y_1 - y_21 * f__) /
               (y_3 + y_43 * f__ - y_1 - y_21 * f__))

    # Limit values to interval [0, 1]
    with np.errstate(invalid='ignore'):
        idxs = (g__ < 0) | (g__ > 1)
    g__[idxs] = np.nan

    return g__


def _calc_abc(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Calculate coefficients for quadratic equation for
    _get_ts_irregular() and _get_ts_uprights().  For _get_ts_uprights
    switch order of pt_2 and pt_3.
    """
    # Pairwise longitudal separations between reference points
    x_21 = pt_2[:, 0] - pt_1[:, 0]
    x_31 = pt_3[:, 0] - pt_1[:, 0]
    x_42 = pt_4[:, 0] - pt_2[:, 0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[:, 1] - pt_1[:, 1]
    y_31 = pt_3[:, 1] - pt_1[:, 1]
    y_42 = pt_4[:, 1] - pt_2[:, 1]

    a__ = x_31 * y_42 - y_31 * x_42
    b__ = out_y * (x_42 - x_31) - out_x * (y_42 - y_31) + \
        x_31 * pt_2[:, 1] - y_31 * pt_2[:, 0] + \
        y_42 * pt_1[:, 0] - x_42 * pt_1[:, 1]
    c__ = out_y * x_21 - out_x * y_21 + pt_1[:, 0] * pt_2[:, 1] - \
        pt_2[:, 0] * pt_1[:, 1]

    return a__, b__, c__


def _mask_coordinates(lons, lats):
    """Mask invalid coordinate values"""
    lons = lons.ravel()
    lats = lats.ravel()
    idxs = ((lons < -180.) | (lons > 180.) |
            (lats < -90.) | (lats > 90.))
    if hasattr(lons, 'mask'):
        lons = np.ma.masked_where(idxs | lons.mask, lons)
    else:
        lons[idxs] = np.nan
    if hasattr(lats, 'mask'):
        lats = np.ma.masked_where(idxs | lats.mask, lats)
    else:
        lats[idxs] = np.nan

    return lons, lats


def _get_corner(stride, valid, in_x, in_y, idx_ref):
    """Get closest set of coordinates from the *valid* locations"""
    # Find the closest valid pixels, if any
    idxs = np.argmax(valid, axis=1)
    # Check which of these were actually valid
    invalid = np.invert(np.max(valid, axis=1))

    # Replace invalid points with np.nan
    x__ = in_x[stride, idxs]
    x__[invalid] = np.nan
    y__ = in_y[stride, idxs]
    y__[invalid] = np.nan
    idx = idx_ref[stride, idxs]

    return x__, y__, idx


def _get_bounding_corners(in_x, in_y, out_x, out_y, neighbours, idx_ref):
    """Get four closest locations from (in_x, in_y) so that they form a
    bounding rectangle around the requested location given by (out_x,
    out_y).
    """

    # Find four closest pixels around the target location

    # Tile output coordinates to same shape as neighbour info
    out_x_tile = np.tile(out_x, (neighbours, 1)).T
    out_y_tile = np.tile(out_y, (neighbours, 1)).T

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    stride = np.arange(x_diff.shape[0])

    # Upper left source pixel
    valid = (x_diff > 0) & (y_diff < 0)
    x_1, y_1, idx_1 = _get_corner(stride, valid, in_x, in_y, idx_ref)

    # Upper right source pixel
    valid = (x_diff < 0) & (y_diff < 0)
    x_2, y_2, idx_2 = _get_corner(stride, valid, in_x, in_y, idx_ref)

    # Lower left source pixel
    valid = (x_diff > 0) & (y_diff > 0)
    x_3, y_3, idx_3 = _get_corner(stride, valid, in_x, in_y, idx_ref)

    # Lower right source pixel
    valid = (x_diff < 0) & (y_diff > 0)
    x_4, y_4, idx_4 = _get_corner(stride, valid, in_x, in_y, idx_ref)

    # Combine sorted indices to idx_ref
    idx_ref = np.vstack((idx_1, idx_2, idx_3, idx_4)).T

    return (np.vstack((x_1, y_1)).T, np.vstack((x_2, y_2)).T,
            np.vstack((x_3, y_3)).T, np.vstack((x_4, y_4)).T, idx_ref)


def _solve_quadratic(a__, b__, c__, min_val=0.0, max_val=1.0):
    """Solve quadratic equation and return the valid roots from interval
    [*min_val*, *max_val*]

    """

    def int_and_float_to_numpy(val):
        if not isinstance(val, np.ndarray):
            if isinstance(val, (int, float)):
                val = [val]
            val = np.array(val)
        return val

    a__ = int_and_float_to_numpy(a__)
    b__ = int_and_float_to_numpy(b__)
    c__ = int_and_float_to_numpy(c__)

    discriminant = b__ * b__ - 4 * a__ * c__

    # Solve the quadratic polynomial
    with np.errstate(invalid='ignore', divide='ignore'):
        x_1 = (-b__ + np.sqrt(discriminant)) / (2 * a__)
        x_2 = (-b__ - np.sqrt(discriminant)) / (2 * a__)

    # Find valid solutions, ie. 0 <= t <= 1
    x__ = x_1.copy()
    with np.errstate(invalid='ignore'):
        idxs = (x_1 < min_val) | (x_1 > max_val)
    x__[idxs] = x_2[idxs]

    with np.errstate(invalid='ignore'):
        idxs = (x__ < min_val) | (x__ > max_val)
    x__[idxs] = np.nan

    return x__


def _get_output_xy(target_area_def, proj):
    """Get x/y coordinates of the target grid."""
    # Read output coordinates
    out_lons, out_lats = target_area_def.get_lonlats()

    # Replace masked arrays with np.nan'd ndarrays
    out_lons = _convert_masks_to_nans(out_lons)
    out_lats = _convert_masks_to_nans(out_lats)

    # Mask invalid coordinates
    out_lons, out_lats = _mask_coordinates(out_lons, out_lats)

    # Convert coordinates to output projection x/y space
    out_x, out_y = proj(out_lons, out_lats)

    return out_x, out_y


def _get_input_xy(source_geo_def, proj, input_idxs, idx_ref):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = source_geo_def.get_lonlats()

    # Select valid locations
    in_lons = in_lons.ravel()[input_idxs]
    in_lats = in_lats.ravel()[input_idxs]

    # Mask invalid values
    in_lons, in_lats = _mask_coordinates(in_lons, in_lats)

    # Expand input coordinates for each output location
    in_lons = in_lons[idx_ref]
    in_lats = in_lats[idx_ref]

    # Replace masked arrays with np.nan'd ndarrays
    in_lons = _convert_masks_to_nans(in_lons)
    in_lats = _convert_masks_to_nans(in_lats)

    # Convert coordinates to output projection x/y space
    in_x, in_y = proj(in_lons, in_lats)

    return in_x, in_y


def _convert_masks_to_nans(arr):
    """Remove masked array masks and replace corresponding values with nans"""
    if hasattr(arr, 'mask'):
        mask = arr.mask
        arr = arr.data
        arr[mask] = np.nan
    return arr


def _check_data_shape(data, input_idxs):
    """Check data shape and adjust if necessary."""
    # Handle multiple datasets
    if data.ndim > 2 and data.shape[0] * data.shape[1] == input_idxs.shape[0]:
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    # Also ravel single dataset
    elif data.shape[0] != input_idxs.size:
        data = data.ravel()

    # Ensure two dimensions
    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    return data
