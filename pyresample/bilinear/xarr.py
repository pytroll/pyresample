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

import warnings

import dask.array as da
import numpy as np
import zarr
from dask import delayed
from xarray import DataArray, Dataset

from pyresample import CHUNK_SIZE
from pyresample._spatial_mp import Proj
from pyresample.bilinear._base import (
    BilinearBase,
    array_slice_for_multiple_arrays,
    find_indices_outside_min_and_max,
    get_slicer,
    get_valid_indices_from_lonlat_boundaries,
    is_swath_to_grid_or_grid_to_grid,
    mask_coordinates,
)
from pyresample.future.resamplers._transform_utils import lonlat2xyz

CACHE_INDICES = ['bilinear_s',
                 'bilinear_t',
                 'slices_x',
                 'slices_y',
                 'mask_slices',
                 'out_coords_x',
                 'out_coords_y']

BIL_COORDINATES = {'bilinear_s': ('x1', ),
                   'bilinear_t': ('x1', ),
                   'slices_x': ('x1', 'n'),
                   'slices_y': ('x1', 'n'),
                   'mask_slices': ('x1', 'n'),
                   'out_coords_x': ('x2', ),
                   'out_coords_y': ('y2', )}


class XArrayBilinearResampler(BilinearBase):
    """Bilinear interpolation using XArray."""

    def resample(self, data, fill_value=None, nprocs=1):
        """Resample the given data."""
        del nprocs
        self.get_bil_info()
        return self.get_sample_from_bil_info(data, fill_value=fill_value, output_shape=None)

    def _create_empty_bil_info(self):
        """Create dummy info for empty result set."""
        self._valid_input_index = da.ones(self._source_geo_def.size, dtype=bool)
        self._index_array = da.ones((self._target_geo_def.size, 4), dtype=np.int32)
        self.bilinear_s = np.nan * da.zeros(self._target_geo_def.size)
        self.bilinear_t = np.nan * da.zeros(self._target_geo_def.size)
        self.slices_x = da.zeros((self._target_geo_def.size, 4), dtype=np.int32)
        self.slices_y = da.zeros((self._target_geo_def.size, 4), dtype=np.int32)
        self.out_coords_x, self.out_coords_y = self._target_geo_def.get_proj_vectors(chunks=CHUNK_SIZE)
        self.mask_slices = self._index_array >= self._source_geo_def.size

    def _get_input_xy(self):
        return _get_input_xy(self._source_geo_def,
                             Proj(self._target_geo_def.proj_str),
                             self._valid_input_index, self._index_array)

    def _get_output_xy(self):
        return _get_output_xy(self._target_geo_def)

    def _limit_output_values_to_input(self, data, res, fill_value):
        epsilon = 1e-6
        data_min = da.nanmin(data) - epsilon
        data_max = da.nanmax(data) + epsilon

        res = da.where(
            find_indices_outside_min_and_max(res, data_min, data_max),
            fill_value, res)

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
            self._out_coords['x'], self._out_coords['y'] = da.compute(self.out_coords_x, self.out_coords_y)

    def _adjust_bands_coordinates_to_match_data(self, data_coords):
        if 'bands' in data_coords:
            self._out_coords['bands'] = data_coords['bands']
        elif 'bands' in self._out_coords:
            del self._out_coords['bands']

    def _slice_data(self, data, fill_value):
        def from_delayed(delayeds, shp):
            return [da.from_delayed(d, shp, np.float32) for d in delayeds]

        if data.ndim == 2:
            shp = self.bilinear_s.shape
        else:
            shp = (data.shape[0],) + self.bilinear_s.shape

        slicer = get_slicer(data)

        return from_delayed(
            self._delayed_slice_data(
                slicer, data, fill_value), shp)

    @delayed(nout=4)
    def _delayed_slice_data(self, slicer, data, fill_value):
        return slicer(data.values, self.slices_x, self.slices_y, self.mask_slices, fill_value)

    def _get_target_proj_vectors(self):
        try:
            self.out_coords_x, self.out_coords_y = self._target_geo_def.get_proj_vectors(chunks=CHUNK_SIZE)
        except AttributeError:
            pass

    def _get_valid_input_index_and_input_coords(self):
        valid_input_index, source_lons, source_lats = \
            _get_valid_input_index(self._source_geo_def,
                                   self._target_geo_def,
                                   self._reduce_data,
                                   self._radius_of_influence)
        input_coords = lonlat2xyz(source_lons, source_lats)
        valid_input_index = np.ravel(valid_input_index)
        input_coords = input_coords[valid_input_index, :].astype(np.float64)

        return da.compute(valid_input_index, input_coords)

    def save_resampling_info(self, filename):
        """Save bilinear resampling look-up tables."""
        zarr_out = Dataset()
        for idx_name, coord in BIL_COORDINATES.items():
            var = getattr(self, idx_name)
            if isinstance(var, np.ndarray):
                var = da.from_array(var, chunks=CHUNK_SIZE)
            else:
                var = var.rechunk(CHUNK_SIZE)
            zarr_out[idx_name] = (coord, var)
        zarr_out.to_zarr(filename)

    def load_resampling_info(self, filename):
        """Load bilinear resampling look-up tables and initialize the resampler."""
        try:
            fid = zarr.open(filename, 'r')
            for val in BIL_COORDINATES:
                cache = da.array(fid[val])
                setattr(self, val, cache)
        except ValueError:
            raise IOError


def _get_output_xy(target_geo_def):
    out_x, out_y = target_geo_def.get_proj_coords(chunks=CHUNK_SIZE)
    return da.compute(np.ravel(out_x), np.ravel(out_y))


def _get_input_xy(source_geo_def, proj, valid_input_index, index_array):
    """Get x/y coordinates for the input area and reduce the data."""
    input_xy_coordinates = da.compute(*mask_coordinates(*_get_raveled_lonlats(source_geo_def)))
    valid_xy_coordinates = array_slice_for_multiple_arrays(valid_input_index, input_xy_coordinates)
    # Expand input coordinates for each output location
    expanded_coordinates = array_slice_for_multiple_arrays(index_array, valid_xy_coordinates)

    return proj(*expanded_coordinates)


def _get_raveled_lonlats(geo_def):
    lons, lats = geo_def.get_lonlats(chunks=CHUNK_SIZE)
    if lons.size == 0 or lats.size == 0:
        raise ValueError('Cannot resample empty data set')
    elif lons.size != lats.size or lons.shape != lats.shape:
        raise ValueError('Mismatch between lons and lats')

    return da.ravel(lons), da.ravel(lats)


def _get_valid_input_index(source_geo_def,
                           target_geo_def,
                           reduce_data,
                           radius_of_influence):
    """Find indices of reduced input data."""
    source_lons, source_lats = _get_raveled_lonlats(source_geo_def)

    valid_input_index = da.invert(
        find_indices_outside_min_and_max(source_lons, -180., 180.) |
        find_indices_outside_min_and_max(source_lats, -90., 90.))

    if reduce_data and is_swath_to_grid_or_grid_to_grid(source_geo_def, target_geo_def):
        valid_input_index &= get_valid_indices_from_lonlat_boundaries(
            target_geo_def, source_lons, source_lats, radius_of_influence)

    return valid_input_index, source_lons, source_lats


class XArrayResamplerBilinear(XArrayBilinearResampler):
    """Wrapper for the old resampler class."""

    def __init__(self, source_geo_def,
                 target_geo_def,
                 radius_of_influence,
                 **kwargs):
        """Initialize resampler."""
        warnings.warn("Use of XArrayResamplerBilinear is deprecated, use XArrayBilinearResampler instead")

        super(XArrayResamplerBilinear, self).__init__(
            source_geo_def,
            target_geo_def,
            radius_of_influence,
            **kwargs)
