#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021
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
"""EWA algorithms operating on numpy arrays."""

import logging

import dask
import dask.array as da
import numpy as np

from pyresample import CHUNK_SIZE
from pyresample.ewa import fornav, ll2cr
from pyresample.future.resamplers.resampler import update_resampled_coords
from pyresample.geometry import SwathDefinition
from pyresample.resampler import BaseResampler

try:
    import xarray as xr
except ImportError:
    xr = None


LOG = logging.getLogger(__name__)


class LegacyDaskEWAResampler(BaseResampler):
    """Resample using an elliptical weighted averaging algorithm.

    This algorithm does **not** use caching or any externally provided data
    mask (unlike the 'nearest' resampler). This algorithm also does not do
    any special handling of dask arrays and will simply pass entire dask
    arrays to delayed versions of the EWA algorithm. Use
    :class:`~pyresample.ewa.DaskEWAResampler` for improved parallel processing
    which should result in reduced overall memory usage when dask is
    configured properly.

    This algorithm works under the assumption that the data is observed
    one scan line at a time. However, good results can still be achieved
    for non-scan based data provided `rows_per_scan` is set to the
    number of rows in the entire swath or by setting it to `None`.

    Args:
        rows_per_scan (int, None):
            Number of data rows for every observed scanline. If None then the
            entire swath is treated as one large scanline.
        weight_count (int):
            number of elements to create in the gaussian weight table.
            Default is 10000. Must be at least 2
        weight_min (float):
            the minimum value to store in the last position of the
            weight table. Default is 0.01, which, with a
            `weight_distance_max` of 1.0 produces a weight of 0.01
            at a grid cell distance of 1.0. Must be greater than 0.
        weight_distance_max (float):
            distance in grid cell units at which to
            apply a weight of `weight_min`. Default is
            1.0. Must be greater than 0.
        weight_delta_max (float):
            maximum distance in grid cells in each grid
            dimension over which to distribute a single swath cell.
            Default is 10.0.
        weight_sum_min (float):
            minimum weight sum value. Cells whose weight sums
            are less than `weight_sum_min` are set to the grid fill value.
            Default is EPSILON.
        maximum_weight_mode (bool):
            If False (default), a weighted average of
            all swath cells that map to a particular grid cell is used.
            If True, the swath cell having the maximum weight of all
            swath cells that map to a particular grid cell is used. This
            option should be used for coded/category data, i.e. snow cover.

    """

    def __init__(self, source_geo_def, target_geo_def):
        """Init EWAResampler."""
        super().__init__(source_geo_def, target_geo_def)
        self.cache = {}

    def resample(self, *args, **kwargs):
        """Run precompute and compute methods.

        .. note::

            This sets the default of 'mask_area' to False since it is
            not needed in EWA resampling currently.

        """
        kwargs.setdefault('mask_area', False)
        return super().resample(*args, **kwargs)

    def _call_ll2cr(self, lons, lats, target_geo_def, swath_usage=0):
        """Wrap ll2cr() for handling dask delayed calls better."""
        new_src = SwathDefinition(lons, lats)

        swath_points_in_grid, cols, rows = ll2cr(new_src, target_geo_def)
        # FIXME: How do we check swath usage/coverage if we only do this
        #        per-block
        # # Determine if enough of the input swath was used
        # grid_name = getattr(self.target_geo_def, "name", "N/A")
        # fraction_in = swath_points_in_grid / float(lons.size)
        # swath_used = fraction_in > swath_usage
        # if not swath_used:
        #     LOG.info("Data does not fit in grid %s because it only %f%% of "
        #              "the swath is used" %
        #              (grid_name, fraction_in * 100))
        #     raise RuntimeError("Data does not fit in grid %s" % (grid_name,))
        # else:
        #     LOG.debug("Data fits in grid %s and uses %f%% of the swath",
        #               grid_name, fraction_in * 100)

        return np.stack([cols, rows], axis=0)

    def precompute(self, cache_dir=None, swath_usage=0, **kwargs):
        """Generate row and column arrays and store it for later use."""
        if self.cache:
            # this resampler should be used for one SwathDefinition
            # no need to recompute ll2cr output again
            return None

        if kwargs.get('mask') is not None:
            LOG.warning("'mask' parameter has no affect during EWA "
                        "resampling")

        del kwargs
        source_geo_def = self.source_geo_def
        target_geo_def = self.target_geo_def

        if cache_dir:
            LOG.warning("'cache_dir' is not used by EWA resampling")

        # Satpy/PyResample don't support dynamic grids out of the box yet
        lons, lats = source_geo_def.get_lonlats()
        if xr is not None and isinstance(lons, xr.DataArray):
            # get dask arrays
            lons = lons.data
            lats = lats.data
        # we are remapping to a static unchanging grid/area with all of
        # its parameters specified
        chunks = (2,) + lons.chunks
        res = da.map_blocks(self._call_ll2cr, lons, lats,
                            target_geo_def, swath_usage,
                            meta=np.array((), dtype=lons.dtype),
                            dtype=lons.dtype, chunks=chunks, new_axis=[0])
        cols = res[0]
        rows = res[1]

        # save the dask arrays in the class instance cache
        # the on-disk cache will store the numpy arrays
        self.cache = {
            "rows": rows,
            "cols": cols,
        }

        return None

    def _call_fornav(self, cols, rows, target_geo_def, data,
                     grid_coverage=0, **kwargs):
        """Wrap fornav() to run as a dask delayed."""
        num_valid_points, res = fornav(cols, rows, target_geo_def,
                                       data, **kwargs)

        if isinstance(data, tuple):
            # convert 'res' from tuple of arrays to one array
            res = np.stack(res)
            num_valid_points = sum(num_valid_points)

        grid_covered_ratio = num_valid_points / float(res.size)
        grid_covered = grid_covered_ratio > grid_coverage
        if not grid_covered:
            msg = "EWA resampling only found %f%% of the grid covered " \
                  "(need %f%%)" % (grid_covered_ratio * 100,
                                   grid_coverage * 100)
            raise RuntimeError(msg)
        LOG.debug("EWA resampling found %f%% of the grid covered",
                  grid_covered_ratio * 100)

        return res

    def _is_data_arr(self, data):
        return xr is not None and isinstance(data, xr.DataArray)

    def _get_rows_per_scan(self, kwargs, data):
        # if the data is scan based then check its metadata or the passed
        # kwargs otherwise assume the entire input swath is one large
        # "scanline"
        if self._is_data_arr(data):
            da_rps = data.attrs.get('rows_per_scan', data.shape[0])
        else:
            da_rps = data.shape[0]
        return kwargs.get('rows_per_scan', da_rps)

    def _get_data_arr_from_xarray(self, data_arr):
        if data_arr.ndim == 3 and 'bands' in data_arr.dims:
            data_in = tuple(data_arr.sel(bands=band).data
                            for band in data_arr['bands'])
        elif data_arr.ndim == 2:
            data_in = data_arr.data
        else:
            raise ValueError("Unsupported data shape for EWA resampling.")
        return data_in

    def _get_data_arr(self, data):
        if self._is_data_arr(data):
            return self._get_data_arr_from_xarray(data)
        if data.ndim != 2:
            raise ValueError("Unsupported data shape for EWA resampling.")
        return data

    def compute(self, data, cache_id=None, fill_value=0, weight_count=10000,
                weight_min=0.01, weight_distance_max=1.0,
                weight_delta_max=10.0, weight_sum_min=-1.0,
                maximum_weight_mode=False, grid_coverage=0, chunks=None,
                **kwargs):
        """Resample the data according to the precomputed X/Y coordinates."""
        rows = self.cache["rows"]
        cols = self.cache["cols"]
        rows_per_scan = self._get_rows_per_scan(kwargs, data)
        data_in = self._get_data_arr(data)

        res = dask.delayed(self._call_fornav, pure=True)(
            cols, rows, self.target_geo_def, data_in,
            grid_coverage=grid_coverage,
            rows_per_scan=rows_per_scan, weight_count=weight_count,
            weight_min=weight_min, weight_distance_max=weight_distance_max,
            weight_delta_max=weight_delta_max, weight_sum_min=weight_sum_min,
            maximum_weight_mode=maximum_weight_mode)
        if isinstance(data_in, tuple):
            new_shape = (len(data_in),) + self.target_geo_def.shape
        else:
            new_shape = self.target_geo_def.shape
        data_arr = da.from_delayed(res, new_shape, dtype=data.dtype,
                                   meta=np.array((), dtype=data.dtype))
        # from delayed creates one large chunk, break it up a bit if we can
        data_arr = data_arr.rechunk([chunks or CHUNK_SIZE] * data_arr.ndim)
        if data.ndim == 3 and data.dims[0] == 'bands':
            dims = ('bands', 'y', 'x')
        elif data.ndim == 2:
            dims = ('y', 'x')
        else:
            dims = data.dims

        if self._is_data_arr(data):
            res = xr.DataArray(data_arr, dims=dims, attrs=data.attrs.copy())
            return update_resampled_coords(data, res, self.target_geo_def)
        return data_arr
