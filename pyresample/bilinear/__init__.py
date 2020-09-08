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

import numpy as np
from pyresample._spatial_mp import Proj
from pyresample import data_reduce, geometry
import warnings

from pyresample import kd_tree
from pykdtree.kdtree import KDTree


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
    out_x, out_y = _get_output_xy_masked(target_area_def, proj)

    # Get input x/y coordinates
    in_x, in_y = _get_input_xy_masked(source_geo_def, proj, input_idxs, idx_ref)

    # Get the four closest corner points around each output location
    pt_1, pt_2, pt_3, pt_4, idx_ref = \
        _get_bounding_corners(in_x, in_y, out_x, out_y, neighbours, idx_ref)

    # Calculate vertical and horizontal fractional distances t and s
    t__, s__ = _get_fractional_distances(pt_1, pt_2, pt_3, pt_4, out_x, out_y)

    # Mask NaN values
    if masked:
        mask = np.isnan(t__) | np.isnan(s__)
        t__ = np.ma.masked_where(mask, t__)
        s__ = np.ma.masked_where(mask, s__)

    return t__, s__, input_idxs, idx_ref


def _mask_coordinates(lons, lats):
    """Mask invalid coordinate values."""
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


def _get_output_xy_masked(target_area_def, proj):
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


def _get_input_xy_masked(source_geo_def, proj, input_idxs, idx_ref):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = source_geo_def.get_lonlats()

    # Select valid locations
    in_lons = in_lons.ravel()[input_idxs]
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
    """Remove masked array masks and replace corresponding values with nans."""
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


class BilinearBase(object):
    """Base class for bilinear interpolation."""

    def __init__(self,
                 source_geo_def,
                 target_geo_def,
                 radius_of_influence,
                 neighbours=32,
                 epsilon=0,
                 reduce_data=True):
        """
        Initialize resampler.

        Parameters
        ----------
        source_geo_def : object
            Geometry definition of source
        target_geo_def : object
            Geometry definition of target
        radius_of_influence : float
            Cut off distance in meters
        neighbours : int, optional
            The number of neigbours to consider for each grid point
        epsilon : float, optional
            Allowed uncertainty in meters. Increasing uncertainty
            reduces execution time
        reduce_data : bool, optional
            Perform initial coarse reduction of source dataset in order
            to reduce execution time

        """
        self.bilinear_t = None
        self.bilinear_s = None
        self.slices_x = None
        self.slices_y = None
        self.mask_slices = None
        self.out_coords_x = None
        self.out_coords_y = None

        self._out_coords = {'x': self.out_coords_x, 'y': self.out_coords_y}
        self._valid_input_index = None
        self._valid_output_index = None
        self._index_array = None
        self._distance_array = None
        self._neighbours = neighbours
        self._epsilon = epsilon
        self._reduce_data = reduce_data
        self._source_geo_def = source_geo_def
        self._target_geo_def = target_geo_def
        self._radius_of_influence = radius_of_influence
        self._resample_kdtree = None
        self._target_lons = None
        self._target_lats = None

    def get_bil_info(self):
        """Calculate bilinear neighbour info."""
        if self._source_geo_def.size < self._neighbours:
            warnings.warn('Searching for %s neighbours in %s data points' %
                          (self._neighbours, self._source_geo_def.size))

        self._get_valid_input_index_and_kdtree()
        if self._resample_kdtree is None:
            return

        self._get_target_lonlats()
        self._get_valid_output_indices()
        self._get_index_array()

        # Calculate vertical and horizontal fractional distances t and s
        self._get_fractional_distances()
        self._get_target_proj_vectors()
        self._get_slices()

    def _get_valid_input_index_and_kdtree(self):
        valid_input_index, resample_kdtree = self._create_resample_kdtree()

        if resample_kdtree:
            self._valid_input_index = valid_input_index
            self._resample_kdtree = resample_kdtree
        else:
            # Handle if all input data is reduced away
            self._create_empty_bil_info()

    def _create_empty_bil_info(self):
        self._valid_input_index = np.ones(self._source_geo_def.size, dtype=np.bool)
        self._index_array = np.ones((self._target_geo_def.size, 4), dtype=np.int32)
        self.bilinear_s = np.nan * np.zeros(self._target_geo_def.size)
        self.bilinear_t = np.nan * np.zeros(self._target_geo_def.size)
        self.slices_x = np.zeros((self._target_geo_def.size, 4), dtype=np.int32)
        self.slices_y = np.zeros((self._target_geo_def.size, 4), dtype=np.int32)
        self.out_coords_x, self.out_coords_y = self._target_geo_def.get_proj_vectors()
        self.mask_slices = self._index_array >= self._source_geo_def.size

    def _get_target_lonlats(self):
        self._target_lons, self._target_lats = self._target_geo_def.get_lonlats()

    def _get_valid_output_indices(self):
        self._valid_output_index = ((self._target_lons >= -180) & (self._target_lons <= 180) &
                                    (self._target_lats <= 90) & (self._target_lats >= -90))

    def _get_index_array(self):
        index_array, _ = self._query_resample_kdtree()
        self._index_array = self._reduce_index_array(index_array)

    def _query_resample_kdtree(self,
                               reduce_data=True):
        """Query kd-tree on slice of target coordinates."""
        res = query_no_distance(self._target_lons, self._target_lats,
                                self._valid_output_index, self._resample_kdtree,
                                self._neighbours, self._epsilon,
                                self._radius_of_influence)
        return res, None

    def _reduce_index_array(self, index_array):
        input_size = np.sum(self._valid_input_index)
        index_mask = index_array == input_size
        return np.where(index_mask, 0, index_array)

    def _get_fractional_distances(self):
        out_x, out_y = self._get_output_xy()
        # Get the four closest corner points around each output location
        pt_1, pt_2, pt_3, pt_4, self._index_array = \
            _get_bounding_corners(*self._get_input_xy(),
                                  out_x, out_y,
                                  self._neighbours, self._index_array)
        self.bilinear_t, self.bilinear_s = _get_fractional_distances(
            pt_1, pt_2, pt_3, pt_4, out_x, out_y)

    def _get_output_xy(self):
        return _get_output_xy(self._target_geo_def)

    def _get_input_xy(self):
        return _get_input_xy(self._source_geo_def,
                             Proj(self._target_geo_def.proj_str),
                             self._valid_input_index, self._index_array)

    def _get_target_proj_vectors(self):
        try:
            self.out_coords_x, self.out_coords_y = self._target_geo_def.get_proj_vectors()
        except AttributeError:
            pass

    def _get_slices(self):
        shp = self._source_geo_def.shape
        cols, lines = np.meshgrid(np.arange(shp[1]),
                                  np.arange(shp[0]))

        self.slices_y, self.slices_x = array_slice_for_multiple_arrays(
            self._index_array,
            array_slice_for_multiple_arrays(
                self._valid_input_index,
                (np.ravel(lines), np.ravel(cols))
            )
        )
        self.mask_slices = self._index_array >= self._source_geo_def.size

    def _create_resample_kdtree(self):
        """Set up kd tree on input."""
        valid_input_index, input_coords = self._get_valid_input_index_and_input_coords()
        kdtree = None
        if input_coords.size:
            kdtree = KDTree(input_coords)
        return valid_input_index, kdtree

    def _get_valid_input_index_and_input_coords(self):
        valid_input_index, source_lons, source_lats = \
            _get_valid_input_index(self._source_geo_def,
                                   self._target_geo_def,
                                   self._reduce_data,
                                   self._radius_of_influence)
        input_coords = lonlat2xyz(source_lons, source_lats)
        valid_input_index = np.ravel(valid_input_index)
        input_coords = input_coords[valid_input_index, :].astype(np.float)

        return valid_input_index, input_coords

    def get_sample_from_bil_info(self, data, fill_value=None, output_shape=None):
        """Resample using pre-computed resampling LUTs."""
        del output_shape
        fill_value = _check_fill_value(fill_value, data.dtype)

        res = self._resample(data, fill_value)
        return self._finalize_output_data(data, res, fill_value)

    def _resample(self, data, fill_value):
        p_1, p_2, p_3, p_4 = self._slice_data(data, fill_value)
        s__, t__ = self.bilinear_s, self.bilinear_t

        return (p_1 * (1 - s__) * (1 - t__) +
                p_2 * s__ * (1 - t__) +
                p_3 * (1 - s__) * t__ +
                p_4 * s__ * t__)

    def _slice_data(self, data, fill_value):
        return get_slicer(data)(data, self.slices_x, self.slices_y, self.mask_slices, fill_value)

    def _finalize_output_data(self, data, res, fill_value):
        raise NotImplementedError


def _check_fill_value(fill_value, dtype):
    """Check that fill value is usable for the data."""
    if fill_value is None:
        if np.issubdtype(dtype, np.integer):
            fill_value = 0
        else:
            fill_value = np.nan
    elif np.issubdtype(dtype, np.integer):
        if np.isnan(fill_value):
            fill_value = 0
        elif np.issubdtype(type(fill_value), np.floating):
            fill_value = int(fill_value)

    return fill_value


def _get_output_xy(target_geo_def):
    out_x, out_y = target_geo_def.get_proj_coords()
    return np.ravel(out_x),  np.ravel(out_y)


def _get_input_xy(source_geo_def, proj, valid_input_index, index_array):
    """Get x/y coordinates for the input area and reduce the data."""
    return proj(
        *array_slice_for_multiple_arrays(
            index_array,
            array_slice_for_multiple_arrays(
                valid_input_index,
                mask_coordinates(*_get_raveled_lonlats(source_geo_def))
            )
        )
    )


def array_slice_for_multiple_arrays(idxs, data):
    """Slices multiple arrays using the same indices."""
    return [d[idxs] for d in data]


def mask_coordinates(lons, lats):
    """Mask invalid coordinate values."""
    idxs = (find_indices_outside_min_and_max(lons, -180., 180.) |
            find_indices_outside_min_and_max(lats, -90., 90.))
    return _np_where_for_multiple_arrays(idxs, (np.nan, np.nan), (lons, lats))


def find_indices_outside_min_and_max(data, min_val, max_val):
    """Return array indices outside the given minimum and maximum values."""
    return (data < min_val) | (data > max_val)


def _get_raveled_lonlats(geo_def):
    lons, lats = geo_def.get_lonlats()
    if lons.size == 0 or lats.size == 0:
        raise ValueError('Cannot resample empty data set')
    elif lons.size != lats.size or lons.shape != lats.shape:
        raise ValueError('Mismatch between lons and lats')

    return np.ravel(lons), np.ravel(lats)


def _get_bounding_corners(in_x, in_y, out_x, out_y, neighbours, index_array):
    """Get bounding corners.

    Get four closest locations from (in_x, in_y) so that they form a
    bounding rectangle around the requested location given by (out_x,
    out_y).

    """
    # Find four closest pixels around the target location

    stride, valid_corners = _get_stride_and_valid_corner_indices(
        out_x, out_y, in_x, in_y, neighbours)
    res = []
    indices = []
    for valid in valid_corners:
        x__, y__, idx = _get_corner(stride, valid, in_x, in_y, index_array)
        res.append(np.transpose(np.vstack((x__, y__))))
        indices.append(idx)
    res.append(np.transpose(np.vstack(indices)))

    return res


def _get_fractional_distances(pt_1, pt_2, pt_3, pt_4, out_x, out_y):
    """Calculate vertical and horizontal fractional distances t and s."""
    # General case, ie. where the the corners form an irregular rectangle
    t__, s__ = _invalid_s_and_t_to_nan(
        *_get_fractional_distances_irregular(pt_1, pt_2, pt_3, pt_4, out_y, out_x)
    )
    # Update where verticals are parallel
    t__, s__ = _update_fractional_distances(
        _get_fractional_distances_uprights_parallel,
        t__, s__, (pt_1, pt_2, pt_3, pt_4), out_x, out_y
    )
    # Update where both verticals and horizontals are parallel
    t__, s__ = _update_fractional_distances(
        _get_fractional_distances_parallellogram,
        t__, s__, (pt_1, pt_2, pt_3), out_x, out_y
    )

    return t__, s__


def _invalid_s_and_t_to_nan(t__, s__):
    return _np_where_for_multiple_arrays(
        (find_indices_outside_min_and_max(t__, 0, 1) |
         find_indices_outside_min_and_max(s__, 0, 1)),
        (np.nan, np.nan),
        (t__, s__))


def _get_fractional_distances_irregular(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where none of the sides are parallel."""
    # Get the valid roots from interval [0, 1]
    t__ = _solve_quadratic(
        *_calc_abc(pt_1, pt_2, pt_3, pt_4, out_y, out_x),
        min_val=0., max_val=1.)

    # Calculate parameter s
    s__ = _solve_another_fractional_distance(t__, pt_1[:, 1], pt_3[:, 1],
                                             pt_2[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _solve_quadratic(a__, b__, c__, min_val=0.0, max_val=1.0):
    """Solve quadratic equation.

    Solve quadratic equation and return the valid roots from interval
    [*min_val*, *max_val*].

    """
    discriminant = b__ * b__ - 4 * a__ * c__

    # Solve the quadratic polynomial
    x_1 = (-b__ + np.sqrt(discriminant)) / (2 * a__)
    x_2 = (-b__ - np.sqrt(discriminant)) / (2 * a__)

    # Find valid solutions, ie. 0 <= t <= 1
    x__ = np.where(
        find_indices_outside_min_and_max(x_1, min_val, max_val),
        x_2, x_1)

    x__ = np.where(
        find_indices_outside_min_and_max(x__, min_val, max_val),
        np.nan, x__)

    return x__


def _calc_abc(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Calculate coefficients for quadratic equation.

    In this order of arguments used for _get_fractional_distances_irregular() and
    _get_fractional_distances_uprights().  For _get_fractional_distances_uprights switch order of pt_2 and
    pt_3.

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


def _solve_another_fractional_distance(f__, y_1, y_2, y_3, y_4, out_y):
    """Solve parameter t__ from s__, or vice versa.

    For solving s__, switch order of y_2 and y_3.
    """
    y_21 = y_2 - y_1
    y_43 = y_4 - y_3

    g__ = ((out_y - y_1 - y_21 * f__) /
           (y_3 + y_43 * f__ - y_1 - y_21 * f__))

    # Limit values to interval [0, 1]
    g__ = np.where(
        find_indices_outside_min_and_max(g__, 0, 1),
        np.nan, g__)

    return g__


def _update_fractional_distances(func, t__, s__, points, out_x, out_y):
    idxs = np.ravel(np.isnan(t__) | np.isnan(s__))
    if np.any(idxs):
        t__, s__ = _invalid_s_and_t_to_nan(
            *_np_where_for_multiple_arrays(
                idxs,
                func(
                    *points, out_y, out_x),
                (t__, s__)
            )
        )
    return t__, s__


def _get_fractional_distances_uprights_parallel(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where uprights are parallel."""
    # Get the valid roots from interval [0, 1]
    s__ = _solve_quadratic(
        *_calc_abc(pt_1, pt_3, pt_2, pt_4, out_y, out_x),
        min_val=0., max_val=1.)

    # Calculate parameter t
    t__ = _solve_another_fractional_distance(s__, pt_1[:, 1], pt_2[:, 1],
                                             pt_3[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _get_fractional_distances_parallellogram(pt_1, pt_2, pt_3, out_y, out_x):
    """Get parameters for the case where uprights are parallel."""
    # Pairwise longitudal separations between reference points
    x_21 = pt_2[:, 0] - pt_1[:, 0]
    x_31 = pt_3[:, 0] - pt_1[:, 0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[:, 1] - pt_1[:, 1]
    y_31 = pt_3[:, 1] - pt_1[:, 1]

    t__ = (x_21 * (out_y - pt_1[:, 1]) - y_21 * (out_x - pt_1[:, 0])) / \
          (x_21 * y_31 - y_21 * x_31)
    t__ = np.where(
        find_indices_outside_min_and_max(t__, 0., 1.),
        np.nan, t__)

    s__ = (out_x - pt_1[:, 0] + x_31 * t__) / x_21
    s__ = np.where(
        find_indices_outside_min_and_max(s__, 0., 1.),
        np.nan, s__)

    return t__, s__


def _get_stride_and_valid_corner_indices(out_x, out_y, in_x, in_y, neighbours):
    out_x_tile = _tile_output_coordinate_vector(out_x, neighbours)
    out_y_tile = _tile_output_coordinate_vector(out_y, neighbours)

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    return (np.arange(x_diff.shape[0]), (
        (x_diff > 0) & (y_diff < 0),  # Upper left corners
        (x_diff < 0) & (y_diff < 0),  # Upper right corners
        (x_diff > 0) & (y_diff > 0),  # Lower left corners
        (x_diff < 0) & (y_diff > 0))  # Lower right corners
    )


def _tile_output_coordinate_vector(vector, neighbours):
    return np.reshape(np.tile(vector, neighbours), (neighbours, vector.size)).T


def _get_corner(stride, valid, in_x, in_y, index_array):
    """Get closest set of coordinates from the *valid* locations."""
    x__, y__, idx = _slice_2d_with_stride_and_indices_for_multiple_arrays(
        (in_x, in_y, index_array),
        stride,
        np.argmax(valid, axis=1)  # The closest valid locations
    )
    # Replace invalid points with np.nan
    x__, y__ = _np_where_for_multiple_arrays(
        np.invert(np.max(valid, axis=1)),
        (np.nan, np.nan),
        (x__, y__)
    )

    return x__, y__, idx


def _slice_2d_with_stride_and_indices_for_multiple_arrays(arrays, stride, idxs):
    return [arr[stride, idxs] for arr in arrays]


def _np_where_for_multiple_arrays(idxs, values_for_idxs, otherwise_arrays):
    return [np.where(idxs, values_for_idxs[i], arr) for i, arr in enumerate(otherwise_arrays)]


def _get_valid_input_index(source_geo_def,
                           target_geo_def,
                           reduce_data,
                           radius_of_influence):
    """Find indices of reduced input data."""
    source_lons, source_lats = _get_raveled_lonlats(source_geo_def)

    valid_input_index = np.invert(
        find_indices_outside_min_and_max(source_lons, -180., 180.)
        | find_indices_outside_min_and_max(source_lats, -90., 90.))

    if reduce_data and is_swath_to_grid_or_grid_to_grid(source_geo_def, target_geo_def):
        valid_input_index &= get_valid_indices_from_lonlat_boundaries(
            target_geo_def, source_lons, source_lats, radius_of_influence)

    return valid_input_index, source_lons, source_lats


def is_swath_to_grid_or_grid_to_grid(source_geo_def, target_geo_def):
    """Check whether the resampling is from swath or grid to grid."""
    return (isinstance(source_geo_def, geometry.CoordinateDefinition) and
            isinstance(target_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition))) or \
           (isinstance(source_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition)) and
            isinstance(target_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition)))


def get_valid_indices_from_lonlat_boundaries(
        target_geo_def, source_lons, source_lats, radius_of_influence):
    """Get valid indices from lonlat boundaries."""
    # Resampling from swath to grid or from grid to grid
    lonlat_boundary = target_geo_def.get_boundary_lonlats()

    # Combine reduced and legal values
    return data_reduce.get_valid_index_from_lonlat_boundaries(
        lonlat_boundary[0],
        lonlat_boundary[1],
        source_lons, source_lats,
        radius_of_influence)


def lonlat2xyz(lons, lats):
    """Convert geographic coordinates to cartesian 3D coordinates."""
    R = 6370997.0
    lats = np.deg2rad(lats)
    r_cos_lats = R * np.cos(lats)
    lons = np.deg2rad(lons)
    x_coords = r_cos_lats * np.cos(lons)
    y_coords = r_cos_lats * np.sin(lons)
    z_coords = R * np.sin(lats)

    return np.stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), axis=-1)


def get_slicer(data):
    """Get slicing function for 2D or 3D arrays, depending on the data."""
    if data.ndim == 2:
        return _slice2d
    elif data.ndim == 3:
        return _slice3d
    else:
        raise ValueError


def _slice2d(values, sl_x, sl_y, mask, fill_value):
    # Slice 2D data
    arr = values[(sl_y, sl_x)]
    arr[(mask, )] = fill_value
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _slice3d(values, sl_x, sl_y, mask, fill_value):
    # Slice 3D data
    arr = values[(slice(None), sl_y, sl_x)]
    arr[(slice(None), mask)] = fill_value
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]


def query_no_distance(target_lons, target_lats,
                      valid_output_index, kdtree, neighbours, epsilon, radius):
    """Query the kdtree. No distances are returned."""
    voir = np.ravel(valid_output_index)
    target_lons_valid = np.ravel(target_lons)[voir]
    target_lats_valid = np.ravel(target_lats)[voir]

    _, index_array = kdtree.query(
        lonlat2xyz(target_lons_valid, target_lats_valid),
        k=neighbours,
        eps=epsilon,
        distance_upper_bound=radius)

    return index_array
