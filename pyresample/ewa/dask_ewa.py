#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020
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
"""Dask-friendly implementation of the EWA resampling algorithm.

The `DaskEWAResampler` class implements the Elliptical Weighted Averaging
(EWA) resampling algorithm in a per-chunk processing scheme. This allows
common dask configurations (number of workers, chunk size, etc) to control
how much data is being worked on at any one time. This limits how much
data is being held in memory at any one time. In cases where not all input
array chunks will be used this implementation should avoid loading/computing
the data for that chunk. In cases where not all output chunks have data in
them, this implementation should avoid unnecessary array creation and memory
usage until necessary.

"""
import math
import logging
from functools import partial
import dask.array as da
from dask.array.core import normalize_chunks
from .ewa import ll2cr
from ._fornav import fornav_weights_and_sums_wrapper, write_grid_image_single
from pyresample.geometry import SwathDefinition
from pyresample.resampler import BaseResampler, update_resampled_coords
import dask
from dask.highlevelgraph import HighLevelGraph
import numpy as np

try:
    import xarray as xr
except ImportError:
    # only used for some use cases
    xr = None

logger = logging.getLogger(__name__)


def _call_ll2cr(lons, lats, target_geo_def, swath_usage=0):
    """Wrap ll2cr() for handling dask delayed calls better."""
    new_src = SwathDefinition(lons, lats)
    swath_points_in_grid, cols, rows = ll2cr(new_src, target_geo_def)
    if swath_points_in_grid == 0:
        return (lons.shape, np.nan, lons.dtype), (lats.shape, np.nan, lats.dtype)
    return np.stack([cols, rows], axis=0)


def call_ll2cr(lons, lats, target_geo_def):
    res = da.map_blocks(_call_ll2cr, lons, lats,
                        target_geo_def, 0,
                        dtype=lons.dtype)
    return res


def _delayed_fornav(ll2cr_result, target_geo_def, y_slice, x_slice, data, kwargs):
    # Adjust cols and rows for this sub-area
    subdef = target_geo_def[y_slice, x_slice]
    weights_dtype = np.float32
    accums_dtype = np.float32
    empty_weights = (subdef.shape, 0, weights_dtype)
    empty_accums = (subdef.shape, 0, accums_dtype)

    # Empty ll2cr results: ((shape, fill, dtype), (shape, fill, dtype))
    if isinstance(ll2cr_result[0], tuple):
        # this source data doesn't fit in the target area at all
        return empty_weights, empty_accums
    cols = ll2cr_result[0]
    rows = ll2cr_result[1]
    if x_slice.start != 0:
        cols = cols - x_slice.start
    if y_slice.start != 0:
        rows = rows - y_slice.start
    weights = np.zeros(subdef.shape, dtype=weights_dtype)
    accums = np.zeros(subdef.shape, dtype=accums_dtype)
    try:
        got_points = fornav_weights_and_sums_wrapper(
            cols, rows, data, weights, accums, np.nan, np.nan,
            **kwargs)
    except RuntimeError:
        return empty_weights, empty_accums
    if got_points:
        return empty_weights, empty_accums
    return weights, accums


def chunk_callable(x_chunk, axis, keepdims, **kwargs):
    """No-op for reduction call."""
    return x_chunk


def combine_fornav(x_chunk, axis, keepdims, computing_meta=False,
                   maximum_weight_mode=False):
    if isinstance(x_chunk, tuple) and len(x_chunk) == 2 and isinstance(x_chunk[0], tuple):
        # single "empty" chunk
        return x_chunk
    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]
    if computing_meta or not len(x_chunk):
        # computing metadata
        return x_chunk
    # if the first element is not an array it is either:
    #   1. (empty_tuple_description, empty_tuple_description)
    #   2. ('missing_chunk', i, j, k)
    valid_chunks = [x for x in x_chunk if not isinstance(x[0], (str, tuple))]
    if not len(valid_chunks):
        if keepdims:
            # split step - return "empty" chunk placeholder
            return x_chunk[0]
        else:
            return np.full(*x_chunk[0][0]), np.full(*x_chunk[0][1])
    weights = [x[0] for x in valid_chunks]
    accums = [x[1] for x in valid_chunks]
    if maximum_weight_mode:
        weights = np.array(weights)
        accums = np.array(accums)
        max_indexes = np.expand_dims(np.argmax(weights, axis=0), axis=0)
        weights = np.take_along_axis(weights, max_indexes, axis=0).squeeze(axis=0)
        accums = np.take_along_axis(accums, max_indexes, axis=0).squeeze(axis=0)
        return weights, accums
    # NOTE: We use the builtin "sum" function below because it does not copy
    #       the numpy arrays. Using numpy.sum would do that.
    return sum(weights), sum(accums)


def average_fornav(x_chunk, axis, keepdims, computing_meta=False, dtype=None,
                   fill_value=None,
                   weight_sum_min=-1.0, maximum_weight_mode=False):
    if not len(x_chunk):
        return x_chunk
    # combine the arrays one last time
    res = combine_fornav(x_chunk, axis, keepdims,
                         computing_meta=computing_meta,
                         maximum_weight_mode=maximum_weight_mode)
    # if we have only "empty" arrays at this point then the target chunk
    # has no valid input data in it.
    if isinstance(res[0], tuple):
        return np.full(res[0], fill_value, dtype)
    weights, accums = res
    out = np.full(weights.shape, fill_value, dtype=dtype)
    write_grid_image_single(out, weights, accums, np.nan,
                            weight_sum_min=weight_sum_min,
                            maximum_weight_mode=maximum_weight_mode)
    return out


class DaskEWAResampler(BaseResampler):
    """Resample using an elliptical weighted averaging algorithm.

    This algorithm does **not** use caching or any externally provided data
    mask (unlike the 'nearest' resampler).

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
        """Initialize in-memory cache."""
        super(DaskEWAResampler, self).__init__(source_geo_def, target_geo_def)
        assert isinstance(source_geo_def, SwathDefinition), \
            "EWA resampling can only operate on SwathDefinitions"
        self.cache = {}

    def resample(self, *args, **kwargs):
        """Run precompute and compute methods.

        .. note::

            This sets the default of 'mask_area' to False since it is
            not needed in EWA resampling currently.

        """
        kwargs.setdefault('mask_area', False)
        return super(DaskEWAResampler, self).resample(*args, **kwargs)

    def _new_chunks(self, in_arr, rows_per_scan):
        """Determine a good scan-based chunk size."""
        if len(in_arr.shape) != 2:
            raise ValueError("Can only rechunk 2D arrays for EWA resampling.")

        # assume (y, x)
        num_cols = in_arr.shape[1]
        num_row_chunks = in_arr.chunks[0][0]
        if num_row_chunks % rows_per_scan == 0:
            row_chunks = num_row_chunks
        else:
            row_chunks = 'auto'
        # what do dask's settings give us for full width chunks
        auto_chunks = normalize_chunks({0: row_chunks, 1: num_cols},
                                       shape=in_arr.shape, dtype=in_arr.dtype,
                                       previous_chunks=in_arr.chunks)
        # let's make them scan-aligned
        chunk_rows = max(math.floor(auto_chunks[0][0] / rows_per_scan), 1) * rows_per_scan
        return {0: chunk_rows, 1: num_cols}

    def _get_rows_per_scan(self, kwargs):
        rows_per_scan = kwargs.get('rows_per_scan')
        if rows_per_scan is None and xr is not None and \
                isinstance(self.source_geo_def.lons, xr.DataArray):
            rows_per_scan = self.source_geo_def.lons.attrs.get('rows_per_scan')
        if rows_per_scan is None:
            # TODO: Should I allow full array as one scanline cases?
            raise ValueError("'rows_per_scan' keyword argument required if "
                             "not found in geolocation (i.e. "
                             "DataArray.attrs['rows_per_scan']).")
        return rows_per_scan

    def precompute(self, cache_dir=None, swath_usage=0, **kwargs):
        """Generate row and column arrays and store it for later use."""
        if self.cache:
            # this resampler should be used for one SwathDefinition
            # no need to recompute ll2cr output again
            return None

        if kwargs.get('mask') is not None:
            logger.warning("'mask' parameter has no affect during EWA "
                           "resampling")

        source_geo_def = self.source_geo_def
        target_geo_def = self.target_geo_def
        if cache_dir:
            logger.warning("'cache_dir' is not used by EWA resampling")

        # Satpy/Pyresample don't support dynamic grids out of the box yet
        rows_per_scan = self._get_rows_per_scan(kwargs)
        new_chunks = self._new_chunks(source_geo_def.lons.data, rows_per_scan)
        lons, lats = source_geo_def.get_lonlats(chunks=new_chunks)
        # run ll2cr to get column/row indexes
        # if chunk does not overlap target area then None is returned
        # otherwise a 3D array (2, y, x) of cols, rows are returned
        ll2cr_result = call_ll2cr(lons, lats, target_geo_def)
        persist = kwargs.get('persist', False)
        if persist:
            ll2cr_delayeds = ll2cr_result.to_delayed()
            ll2cr_delayeds = dask.persist(*ll2cr_delayeds.tolist())

        block_cache = {}
        for in_row_idx in range(lons.numblocks[0]):
            for in_col_idx in range(lons.numblocks[1]):
                key = (ll2cr_result.name, in_row_idx, in_col_idx)
                if persist:
                    this_delayed = ll2cr_delayeds[in_row_idx][in_col_idx]
                    result = dask.compute(this_delayed)[0]
                    if not isinstance(result[0], tuple):
                        block_cache[key] = this_delayed.key
                else:
                    block_cache[key] = key

        # save the dask arrays in the class instance cache
        self.cache = {
            'll2cr_result': ll2cr_result,
            'll2cr_blocks': block_cache,
        }
        return None

    def _get_input_tuples(self, data, kwargs):
        if xr is not None and isinstance(data, xr.DataArray):
            xr_obj = data
            if data.ndim == 3 and 'bands' in data.dims:
                data_in = tuple(data.sel(bands=band).data
                                for band in data['bands'])
            elif data.ndim == 2:
                data_in = (data.data,)
            elif data.ndim >= 3:
                # TODO: Create tuple of 2D arrays to operate on. For example:
                #       non_xy = [x for x in a.dims if x not in ['y', 'x']]
                #       b = a.stack({'z': non_xy})
                #       iter_list = [b[..., idx] for idx in range(b.sizes['z'])]
                #       new_arr = xr.DataArray(iter_list, dims=('z', 'y', 'x'),
                #           coords={'z': b.coords['z']}, indexes={'z': b.indexes['z']})
                #       new_arr.unstack('z')
                raise NotImplementedError("EWA support for dimensions other "
                                          "than y, x, and bands is not "
                                          "implemented.")
            else:
                raise ValueError("EWA cannot handle 1D arrays.")
        else:
            xr_obj = None
            if data.ndim != 2:
                raise ValueError("Can only support 2D arrays unless "
                                 "provided as an xarray DataArray object.")
            data_in = (data,)
        return data_in, xr_obj

    def _convert_to_dask(self, data_in, rows_per_scan):
        new_chunks = self._new_chunks(self.source_geo_def.lons, rows_per_scan)
        for data in data_in:
            if not isinstance(data, da.Array):
                yield da.from_array(data, chunks=new_chunks)
            else:
                yield data.rechunk(new_chunks)

    def _run_fornav_single(self, data, out_chunks, target_geo_def, fill_value, **kwargs):
        y_start = 0
        output_stack = {}
        ll2cr_result = self.cache['ll2cr_result']
        ll2cr_blocks = self.cache['ll2cr_blocks'].items()
        ll2cr_numblocks = ll2cr_result.shape if isinstance(ll2cr_result, np.ndarray) else ll2cr_result.numblocks
        name = "fornav-{}".format(data.name)
        maximum_weight_mode = kwargs.get('maximum_weight_mode', False)
        weight_sum_min = kwargs.get('weight_sum_min', -1.0)
        fill_value = kwargs.pop('fill_value', 0)
        for out_row_idx in range(len(out_chunks[0])):
            y_end = y_start + out_chunks[0][out_row_idx]
            x_start = 0
            for out_col_idx in range(len(out_chunks[1])):
                x_end = x_start + out_chunks[1][out_col_idx]
                y_slice = slice(y_start, y_end)
                x_slice = slice(x_start, x_end)
                for z_idx, ((ll2cr_name, in_row_idx, in_col_idx), ll2cr_block) in enumerate(ll2cr_blocks):
                    key = (name, z_idx, out_row_idx, out_col_idx)
                    output_stack[key] = (_delayed_fornav,
                                         ll2cr_block,
                                         target_geo_def, y_slice, x_slice,
                                         (data.name, in_row_idx, in_col_idx), kwargs)
                x_start = x_end
            y_start = y_end

        dsk_graph = HighLevelGraph.from_collections(name, output_stack, dependencies=[data, ll2cr_result])
        stack_chunks = ((1,) * (ll2cr_numblocks[0] * ll2cr_numblocks[1]),) + out_chunks
        out_stack = da.Array(dsk_graph, name, stack_chunks, data.dtype)
        combine_fornav_with_kwargs = partial(
            average_fornav, maximum_weight_mode=maximum_weight_mode)
        average_fornav_with_kwargs = partial(
            average_fornav, maximum_weight_mode=maximum_weight_mode,
            weight_sum_min=weight_sum_min, dtype=data.dtype,
            fill_value=fill_value)
        out = da.reduction(out_stack, chunk_callable,
                           average_fornav_with_kwargs,
                           combine=combine_fornav_with_kwargs, axis=(0,),
                           dtype=data.dtype, concatenate=False)
        return out

    def compute(self, data, cache_id=None, chunks=None, fill_value=None,
                weight_count=10000, weight_min=0.01, weight_distance_max=1.0,
                weight_delta_max=1.0, weight_sum_min=-1.0,
                maximum_weight_mode=False, grid_coverage=0, **kwargs):
        """Resample the data according to the precomputed X/Y coordinates."""
        # not used in this step
        kwargs.pop("persist", None)
        data_in, xr_obj = self._get_input_tuples(data, kwargs)
        rows_per_scan = self._get_rows_per_scan(kwargs)
        data_in = tuple(self._convert_to_dask(data_in, rows_per_scan))
        out_chunks = normalize_chunks(chunks or 'auto',
                                      shape=self.target_geo_def.shape,
                                      dtype=data.dtype)
        fornav_kwargs = kwargs.copy()
        fornav_kwargs.update(dict(
            weight_count=weight_count,
            weight_min=weight_min,
            weight_distance_max=weight_distance_max,
            weight_delta_max=weight_delta_max,
            weight_sum_min=weight_sum_min,
            maximum_weight_mode=maximum_weight_mode,
            rows_per_scan=rows_per_scan,
        ))

        # determine a fill value if they didn't tell us what they have as a
        # fill value in the numpy arrays
        if fill_value is None:
            if np.issubdtype(data_in[0].dtype, np.floating):
                fill_value = np.nan
            elif np.issubdtype(data_in[0].dtype, np.integer):
                fill_value = -999
            else:
                raise ValueError(
                    "Unsupported input data type for EWA Resampling: {}".format(data_in[0].dtype))

        data_out = []
        for data_subarr in data_in:
            res = self._run_fornav_single(data_subarr, out_chunks,
                                          self.target_geo_def,
                                          fill_value,
                                          **fornav_kwargs)
            data_out.append(res)
        if data.ndim == 2:
            out = data_out[0]
        else:
            out = da.concatenate([arr[None, ...] for arr in data_out], axis=0)

        if xr_obj is not None:
            dims = [d for d in xr_obj.dims if d not in ('y', 'x')] + ['y', 'x']
            out = xr.DataArray(out, attrs=xr_obj.attrs.copy(),
                               dims=dims)
            out = update_resampled_coords(xr_obj, out, self.target_geo_def)
        return out
