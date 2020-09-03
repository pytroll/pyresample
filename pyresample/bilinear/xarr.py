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

"""XArray version of bilinear interpolation."""

try:
    from xarray import DataArray
    import dask.array as da
except ImportError:
    DataArray = None
    da = None

import numpy as np

from pyresample._spatial_mp import Proj

from pykdtree.kdtree import KDTree
from pyresample import data_reduce, geometry, CHUNK_SIZE

from pyresample.bilinear import BilinearBase


CACHE_INDICES = ['bilinear_s',
                 'bilinear_t',
                 'slices_x',
                 'slices_y',
                 'mask_slices',
                 'out_coords_x',
                 'out_coords_y']


class XArrayResamplerBilinear(BilinearBase):
    """Bilinear interpolation using XArray."""

    def _create_empty_bil_info(self):
        """Create dummy info for empty result set."""
        self._valid_input_index = da.ones(self._source_geo_def.size, dtype=np.bool)
        self._index_array = da.ones((self._target_geo_def.size, 4), dtype=np.int32)
        self.bilinear_s = np.nan * da.zeros(self._target_geo_def.size)
        self.bilinear_t = np.nan * da.zeros(self._target_geo_def.size)
        self.slices_x = da.zeros((self._target_geo_def.size, 4), dtype=np.int32)
        self.slices_y = da.zeros((self._target_geo_def.size, 4), dtype=np.int32)
        self.out_coords_x, self.out_coords_y = self._target_geo_def.get_proj_vectors(chunks=CHUNK_SIZE)
        self.mask_slices = self._index_array >= self._source_geo_def.size

    def _reduce_index_array(self, index_array):
        input_size = da.sum(self._valid_input_index)
        index_mask = index_array == input_size
        return da.where(index_mask, 0, index_array)

    def _get_input_xy(self):
        return _get_input_xy(self._source_geo_def,
                             Proj(self._target_geo_def.proj_str),
                             self._valid_input_index, self._index_array)

    def _get_ts(self):
        out_x, out_y = _get_output_xy(self._target_geo_def)
        # Get the four closest corner points around each output location
        pt_1, pt_2, pt_3, pt_4, self._index_array = \
            _get_bounding_corners(*self._get_input_xy(),
                                  out_x, out_y,
                                  self._neighbours, self._index_array)
        self.bilinear_t, self.bilinear_s = _get_ts(pt_1, pt_2, pt_3, pt_4, out_x, out_y)

    def _limit_output_values_to_input(self, data, res, fill_value):
        epsilon = 1e-6
        data_min = da.nanmin(data) - epsilon
        data_max = da.nanmax(data) + epsilon

        idxs = (res > data_max) | (res < data_min)
        res = da.where(idxs, fill_value, res)
        return da.where(np.isnan(res), fill_value, res)

    def _reshape_to_target_area(self, res, ndim):
        shp = self._target_geo_def.shape
        if ndim == 3:
            res = da.reshape(res, (res.shape[0], shp[0], shp[1]))
        else:
            res = da.reshape(res, (shp[0], shp[1]))

        return res

    def _finalize_output_data(self, data, res, fill_value):
        res = self._limit_output_values_to_input(data, res, fill_value)
        res = self._reshape_to_target_area(res, data.ndim)

        self._add_missing_coordinates(data)

        return DataArray(res, dims=data.dims, coords=self._out_coords)

    def _compute_indices(self):
        for idx in CACHE_INDICES:
            var = getattr(self, idx)
            try:
                var = var.compute()
                setattr(self, idx, var)
            except AttributeError:
                continue

    def _add_missing_coordinates(self, data):
        self._add_x_and_y_coordinates()
        for _, dim in enumerate(data.dims):
            if dim not in self._out_coords:
                try:
                    self._out_coords[dim] = data.coords[dim]
                except KeyError:
                    pass
        self._adjust_bands_coordinates_to_match_data(data.coords)

    def _add_x_and_y_coordinates(self):
        if self._out_coords['x'] is None and self.out_coords_x is not None:
            self._out_coords['x'] = self.out_coords_x
            self._out_coords['y'] = self.out_coords_y

    def _adjust_bands_coordinates_to_match_data(self, data_coords):
        if 'bands' in data_coords:
            self._out_coords['bands'] = data_coords['bands']

    def _slice_data(self, data, fill_value):
        if data.ndim == 2:
            slicer = _slice2d
        elif data.ndim == 3:
            slicer = _slice3d
        else:
            raise ValueError

        return slicer(data.values, self.slices_x, self.slices_y, self.mask_slices, fill_value)

    def _get_target_proj_vectors(self):
        try:
            self.out_coords_x, self.out_coords_y = self._target_geo_def.get_proj_vectors(chunks=CHUNK_SIZE)
        except AttributeError:
            pass

    def _get_slices(self):
        shp = self._source_geo_def.shape
        cols, lines = np.meshgrid(np.arange(shp[1]),
                                  np.arange(shp[0]))
        cols = np.ravel(cols)
        lines = np.ravel(lines)

        vii = self._valid_input_index
        ia_ = self._index_array

        # ia_ contains reduced (valid) indices of the source array, and has the
        # shape of the destination array
        rlines = lines[vii][ia_]
        rcols = cols[vii][ia_]

        self.mask_slices = ia_ >= self._source_geo_def.size
        self.slices_y = rlines
        self.slices_x = rcols

    def _get_valid_input_index_and_input_coords(self):
        valid_input_index, source_lons, source_lats = \
            _get_valid_input_index(self._source_geo_def,
                                   self._target_geo_def,
                                   self._reduce_data,
                                   self._radius_of_influence)
        input_coords = lonlat2xyz(source_lons, source_lats)
        valid_input_index = da.ravel(valid_input_index)
        input_coords = input_coords[valid_input_index, :].astype(np.float)

        return da.compute(valid_input_index, input_coords)

    def _create_resample_kdtree(self):
        """Set up kd tree on input."""
        valid_input_index, input_coords = self._get_valid_input_index_and_input_coords()
        kdtree = None
        if input_coords.size:
            kdtree = KDTree(input_coords)
        return valid_input_index, kdtree

    def _query_resample_kdtree(self,
                               reduce_data=True):
        """Query kd-tree on slice of target coordinates."""
        res = query_no_distance(self._target_lons, self._target_lats,
                                self._valid_output_index, self._resample_kdtree,
                                self._neighbours, self._epsilon,
                                self._radius_of_influence)
        return res, None


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


def _get_output_xy(target_geo_def):
    out_x, out_y = target_geo_def.get_proj_coords(chunks=CHUNK_SIZE)
    return da.ravel(out_x),  da.ravel(out_y)


def _get_input_xy(source_geo_def, proj, valid_input_index, index_array):
    """Get x/y coordinates for the input area and reduce the data."""
    in_lons, in_lats = source_geo_def.get_lonlats(chunks=CHUNK_SIZE)

    # Mask invalid values
    in_lons, in_lats = _mask_coordinates(in_lons, in_lats)

    # Select valid locations
    # TODO: direct indexing w/o .compute() results in
    # "ValueError: object too deep for desired array

    in_lons = da.ravel(in_lons)
    in_lons = in_lons.compute()
    in_lons = in_lons[valid_input_index]
    in_lats = da.ravel(in_lats)
    in_lats = in_lats.compute()
    in_lats = in_lats[valid_input_index]
    index_array = index_array.compute()

    # Expand input coordinates for each output location
    in_lons = in_lons[index_array]
    in_lats = in_lats[index_array]

    # Convert coordinates to output projection x/y space
    in_x, in_y = proj(in_lons, in_lats)

    return in_x, in_y


def _mask_coordinates(lons, lats):
    """Mask invalid coordinate values."""
    idxs = ((lons < -180.) | (lons > 180.) |
            (lats < -90.) | (lats > 90.))
    lons = da.where(idxs, np.nan, lons)
    lats = da.where(idxs, np.nan, lats)

    return lons, lats


def _get_bounding_corners(in_x, in_y, out_x, out_y, neighbours, index_array):
    """Get bounding corners.

    Get four closest locations from (in_x, in_y) so that they form a
    bounding rectangle around the requested location given by (out_x,
    out_y).

    """
    # Find four closest pixels around the target location

    # FIXME: how to daskify?
    # Tile output coordinates to same shape as neighbour info
    # Replacing with da.transpose and da.tile doesn't work
    out_x_tile = np.reshape(np.tile(out_x, neighbours),
                            (neighbours, out_x.size)).T
    out_y_tile = np.reshape(np.tile(out_y, neighbours),
                            (neighbours, out_y.size)).T

    # Get differences in both directions
    x_diff = out_x_tile - in_x
    y_diff = out_y_tile - in_y

    stride = np.arange(x_diff.shape[0])

    # Upper left source pixel
    valid = (x_diff > 0) & (y_diff < 0)
    x_1, y_1, idx_1 = _get_corner(stride, valid, in_x, in_y, index_array)

    # Upper right source pixel
    valid = (x_diff < 0) & (y_diff < 0)
    x_2, y_2, idx_2 = _get_corner(stride, valid, in_x, in_y, index_array)

    # Lower left source pixel
    valid = (x_diff > 0) & (y_diff > 0)
    x_3, y_3, idx_3 = _get_corner(stride, valid, in_x, in_y, index_array)

    # Lower right source pixel
    valid = (x_diff < 0) & (y_diff > 0)
    x_4, y_4, idx_4 = _get_corner(stride, valid, in_x, in_y, index_array)

    # Combine sorted indices to index_array
    index_array = np.transpose(np.vstack((idx_1, idx_2, idx_3, idx_4)))

    return (np.transpose(np.vstack((x_1, y_1))),
            np.transpose(np.vstack((x_2, y_2))),
            np.transpose(np.vstack((x_3, y_3))),
            np.transpose(np.vstack((x_4, y_4))),
            index_array)


def _get_corner(stride, valid, in_x, in_y, index_array):
    """Get closest set of coordinates from the *valid* locations."""
    # Find the closest valid pixels, if any
    idxs = np.argmax(valid, axis=1)
    # Check which of these were actually valid
    invalid = np.invert(np.max(valid, axis=1))

    # idxs = idxs.compute()
    index_array = index_array.compute()

    # Replace invalid points with np.nan
    x__ = in_x[stride, idxs]  # TODO: daskify
    x__ = da.where(invalid, np.nan, x__)
    y__ = in_y[stride, idxs]  # TODO: daskify
    y__ = da.where(invalid, np.nan, y__)

    idx = index_array[stride, idxs]  # TODO: daskify

    return x__, y__, idx


def _get_ts(pt_1, pt_2, pt_3, pt_4, out_x, out_y):
    """Calculate vertical and horizontal fractional distances t and s."""
    def invalid_to_nan(t__, s__):
        idxs = (t__ < 0) | (t__ > 1) | (s__ < 0) | (s__ > 1)
        t__ = da.where(idxs, np.nan, t__)
        s__ = da.where(idxs, np.nan, s__)
        return t__, s__

    # General case, ie. where the the corners form an irregular rectangle
    t__, s__ = _get_ts_irregular(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Replace invalid values with NaNs
    t__, s__ = invalid_to_nan(t__, s__)

    # Cases where verticals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    idxs = da.ravel(idxs)

    if da.any(idxs):
        t_new, s_new = _get_ts_uprights_parallel(pt_1, pt_2,
                                                 pt_3, pt_4,
                                                 out_y, out_x)
        t__ = da.where(idxs, t_new, t__)
        s__ = da.where(idxs, s_new, s__)

    # Replace invalid values with NaNs
    t__, s__ = invalid_to_nan(t__, s__)

    # Cases where both verticals and horizontals are parallel
    idxs = da.isnan(t__) | da.isnan(s__)
    # Remove extra dimensions
    idxs = da.ravel(idxs)
    if da.any(idxs):
        t_new, s_new = _get_ts_parallellogram(pt_1, pt_2, pt_3,
                                              out_y, out_x)
        t__ = da.where(idxs, t_new, t__)
        s__ = da.where(idxs, s_new, s__)

    # Replace invalid values with NaNs
    t__, s__ = invalid_to_nan(t__, s__)

    return t__, s__


def _get_ts_irregular(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where none of the sides are parallel."""
    # Get parameters for the quadratic equation
    # TODO: check if needs daskifying
    a__, b__, c__ = _calc_abc(pt_1, pt_2, pt_3, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    t__ = _solve_quadratic(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter s
    s__ = _solve_another_fractional_distance(t__, pt_1[:, 1], pt_3[:, 1],
                                             pt_2[:, 1], pt_4[:, 1], out_y)

    return t__, s__


# Might not need daskifying
def _calc_abc(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Calculate coefficients for quadratic equation.

    In this order of arguments used for _get_ts_irregular() and
    _get_ts_uprights().  For _get_ts_uprights switch order of pt_2 and
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


def _solve_quadratic(a__, b__, c__, min_val=0.0, max_val=1.0):
    """Solve quadratic equation.

    Solve quadratic equation and return the valid roots from interval
    [*min_val*, *max_val*].

    """
    discriminant = b__ * b__ - 4 * a__ * c__

    # Solve the quadratic polynomial
    x_1 = (-b__ + da.sqrt(discriminant)) / (2 * a__)
    x_2 = (-b__ - da.sqrt(discriminant)) / (2 * a__)

    # Find valid solutions, ie. 0 <= t <= 1
    idxs = (x_1 < min_val) | (x_1 > max_val)
    x__ = da.where(idxs, x_2, x_1)

    idxs = (x__ < min_val) | (x__ > max_val)
    x__ = da.where(idxs, np.nan, x__)

    return x__


def _solve_another_fractional_distance(f__, y_1, y_2, y_3, y_4, out_y):
    """Solve parameter t__ from s__, or vice versa.

    For solving s__, switch order of y_2 and y_3.
    """
    y_21 = y_2 - y_1
    y_43 = y_4 - y_3

    g__ = ((out_y - y_1 - y_21 * f__) /
           (y_3 + y_43 * f__ - y_1 - y_21 * f__))

    # Limit values to interval [0, 1]
    idxs = (g__ < 0) | (g__ > 1)
    g__ = da.where(idxs, np.nan, g__)

    return g__


def _get_ts_uprights_parallel(pt_1, pt_2, pt_3, pt_4, out_y, out_x):
    """Get parameters for the case where uprights are parallel."""
    # Get parameters for the quadratic equation
    a__, b__, c__ = _calc_abc(pt_1, pt_3, pt_2, pt_4, out_y, out_x)

    # Get the valid roots from interval [0, 1]
    s__ = _solve_quadratic(a__, b__, c__, min_val=0., max_val=1.)

    # Calculate parameter t
    t__ = _solve_another_fractional_distance(s__, pt_1[:, 1], pt_2[:, 1],
                                             pt_3[:, 1], pt_4[:, 1], out_y)

    return t__, s__


def _get_ts_parallellogram(pt_1, pt_2, pt_3, out_y, out_x):
    """Get parameters for the case where uprights are parallel."""
    # Pairwise longitudal separations between reference points
    x_21 = pt_2[:, 0] - pt_1[:, 0]
    x_31 = pt_3[:, 0] - pt_1[:, 0]

    # Pairwise latitudal separations between reference points
    y_21 = pt_2[:, 1] - pt_1[:, 1]
    y_31 = pt_3[:, 1] - pt_1[:, 1]

    t__ = (x_21 * (out_y - pt_1[:, 1]) - y_21 * (out_x - pt_1[:, 0])) / \
          (x_21 * y_31 - y_21 * x_31)
    idxs = (t__ < 0.) | (t__ > 1.)
    t__ = da.where(idxs, np.nan, t__)

    s__ = (out_x - pt_1[:, 0] + x_31 * t__) / x_21
    idxs = (s__ < 0.) | (s__ > 1.)
    s__ = da.where(idxs, np.nan, s__)

    return t__, s__


def query_no_distance(target_lons, target_lats,
                      valid_output_index, kdtree, neighbours, epsilon, radius):
    """Query the kdtree. No distances are returned."""
    voi = valid_output_index
    voir = da.ravel(voi)
    target_lons_valid = da.ravel(target_lons)[voir]
    target_lats_valid = da.ravel(target_lats)[voir]

    coords = lonlat2xyz(target_lons_valid, target_lats_valid)
    distance_array, index_array = kdtree.query(
        coords.compute(),
        k=neighbours,
        eps=epsilon,
        distance_upper_bound=radius)

    return index_array


def _get_valid_input_index(source_geo_def,
                           target_geo_def,
                           reduce_data,
                           radius_of_influence):
    """Find indices of reduced input data."""
    source_lons, source_lats = source_geo_def.get_lonlats(chunks=CHUNK_SIZE)
    source_lons = da.ravel(source_lons)
    source_lats = da.ravel(source_lats)

    if source_lons.size == 0 or source_lats.size == 0:
        raise ValueError('Cannot resample empty data set')
    elif source_lons.size != source_lats.size or \
            source_lons.shape != source_lats.shape:
        raise ValueError('Mismatch between lons and lats')

    # Remove illegal values
    valid_input_index = ((source_lons >= -180) & (source_lons <= 180) &
                         (source_lats <= 90) & (source_lats >= -90))

    if reduce_data:
        # Reduce dataset
        if (isinstance(source_geo_def, geometry.CoordinateDefinition) and
            isinstance(target_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition))) or \
           (isinstance(source_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition)) and
            isinstance(target_geo_def, (geometry.GridDefinition,
                                        geometry.AreaDefinition))):
            # Resampling from swath to grid or from grid to grid
            lonlat_boundary = target_geo_def.get_boundary_lonlats()

            # Combine reduced and legal values
            valid_input_index &= \
                data_reduce.get_valid_index_from_lonlat_boundaries(
                    lonlat_boundary[0],
                    lonlat_boundary[1],
                    source_lons, source_lats,
                    radius_of_influence)

    if (isinstance(valid_input_index, np.ma.core.MaskedArray)):
        # Make sure valid_input_index is not a masked array
        valid_input_index = valid_input_index.filled(False)

    return valid_input_index, source_lons, source_lats


def lonlat2xyz(lons, lats):
    """Convert geographic coordinates to cartesian 3D coordinates."""
    R = 6370997.0
    x_coords = R * da.cos(da.deg2rad(lats)) * da.cos(da.deg2rad(lons))
    y_coords = R * da.cos(da.deg2rad(lats)) * da.sin(da.deg2rad(lons))
    z_coords = R * da.sin(da.deg2rad(lats))

    return da.stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), axis=-1)
