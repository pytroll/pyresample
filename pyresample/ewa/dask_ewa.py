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
import logging
import math
from functools import partial

import dask
import dask.array as da
import numpy as np
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph

from pyresample.ewa import ll2cr
from pyresample.ewa._fornav import fornav_weights_and_sums_wrapper, write_grid_image_single
from pyresample.geometry import SwathDefinition
from pyresample.resampler import BaseResampler

from ..future.resamplers.resampler import update_resampled_coords

try:
    import xarray as xr
except ImportError:
    # only used for some use cases
    xr = None

logger = logging.getLogger(__name__)


def _call_ll2cr(lons, lats, target_geo_def):
    """Wrap ll2cr() for handling dask delayed calls better."""
    new_src = SwathDefinition(lons, lats)
    swath_points_in_grid, cols, rows = ll2cr(new_src, target_geo_def)
    if swath_points_in_grid == 0:
        return (lons.shape, np.nan, lons.dtype), (lats.shape, np.nan, lats.dtype)
    return np.stack([cols, rows], axis=0)


def _call_mapped_ll2cr(lons, lats, target_geo_def):
    res = da.map_blocks(_call_ll2cr, lons, lats,
                        target_geo_def,
                        meta=np.array((), dtype=lons.dtype),
                        dtype=lons.dtype)
    return res


def _delayed_fornav(ll2cr_result, target_geo_def, y_slice, x_slice, data, fill_value, kwargs):
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
            cols, rows, data, weights, accums, fill_value, fill_value,
            **kwargs)
    except RuntimeError:
        return empty_weights, empty_accums
    if not got_points:
        return empty_weights, empty_accums
    return weights, accums


def _chunk_callable(x_chunk, axis, keepdims, **kwargs):
    """No-op for reduction call."""
    return x_chunk


def _combine_fornav(x_chunk, axis, keepdims, computing_meta=False,
                    maximum_weight_mode=False):
    if computing_meta or _is_empty_chunk(x_chunk):
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


def _is_empty_chunk(x_chunk):
    return isinstance(x_chunk, tuple) and len(x_chunk) == 2 and isinstance(x_chunk[0], tuple)


def _average_fornav(x_chunk, axis, keepdims, computing_meta=False, dtype=None,
                    fill_value=None,
                    weight_sum_min=-1.0, maximum_weight_mode=False):
    if computing_meta or not len(x_chunk):
        return x_chunk
    # combine the arrays one last time
    res = _combine_fornav(x_chunk, axis, keepdims,
                          computing_meta=computing_meta,
                          maximum_weight_mode=maximum_weight_mode)
    # if we have only "empty" arrays at this point then the target chunk
    # has no valid input data in it.
    if isinstance(res[0], tuple):
        # res is (weights_info, accums_info)
        # weights_info is (shape, fill, dtype)
        return np.full(res[0][0], fill_value, dtype)
    weights, accums = res
    out = np.full(weights.shape, fill_value, dtype=dtype)
    write_grid_image_single(out, weights, accums, fill_value,
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

    """

    def __init__(self, source_geo_def, target_geo_def):
        """Initialize in-memory cache."""
        super(DaskEWAResampler, self).__init__(source_geo_def, target_geo_def)
        if not isinstance(source_geo_def, SwathDefinition):
            raise ValueError("EWA resampling can only operate on SwathDefinitions")
        self.cache = {}

    def _new_chunks(self, in_arr, rows_per_scan):
        """Determine a good scan-based chunk size."""
        if len(in_arr.shape) != 2:
            raise ValueError("Can only rechunk 2D arrays for EWA resampling.")
        if xr is not None and isinstance(in_arr, xr.DataArray):
            # get the dask or numpy array underneath
            in_arr = in_arr.data

        # assume (y, x)
        num_cols = in_arr.shape[1]
        prev_chunks = getattr(in_arr, 'chunks',
                              tuple((x,) for x in in_arr.shape))
        num_row_chunks = prev_chunks[0][0]
        if num_row_chunks % rows_per_scan == 0:
            row_chunks = num_row_chunks
        else:
            row_chunks = 'auto'
        # what do dask's settings give us for full width chunks
        auto_chunks = normalize_chunks({0: row_chunks, 1: num_cols},
                                       shape=in_arr.shape, dtype=in_arr.dtype,
                                       previous_chunks=prev_chunks)
        # let's make them scan-aligned
        chunk_rows = max(math.floor(auto_chunks[0][0] / rows_per_scan), 1) * rows_per_scan
        return {0: chunk_rows, 1: num_cols}

    def _get_rows_per_scan(self, rows_per_scan=None):
        if rows_per_scan is None and xr is not None and \
                isinstance(self.source_geo_def.lons, xr.DataArray):
            rows_per_scan = self.source_geo_def.lons.attrs.get('rows_per_scan')
        if rows_per_scan is None:
            raise ValueError("'rows_per_scan' keyword argument required if "
                             "not found in geolocation (i.e. "
                             "DataArray.attrs['rows_per_scan']).")
        if rows_per_scan == 0:
            rows_per_scan = self.source_geo_def.shape[0]
        return rows_per_scan

    def _fill_block_cache_with_ll2cr_results(self, ll2cr_result,
                                             num_row_blocks,
                                             num_col_blocks,
                                             persist):
        if persist:
            ll2cr_delayeds = ll2cr_result.to_delayed()
            ll2cr_delayeds = dask.persist(*ll2cr_delayeds.tolist())

        block_cache = {}
        for in_row_idx in range(num_row_blocks):
            for in_col_idx in range(num_col_blocks):
                key = (ll2cr_result.name, in_row_idx, in_col_idx)
                if persist:
                    this_delayed = ll2cr_delayeds[in_row_idx][in_col_idx]
                    result = dask.compute(this_delayed)[0]
                    # XXX: Is this optimization lost because the persisted keys
                    #  in `ll2cr_delayeds` are used in future computations?
                    if not isinstance(result[0], tuple):
                        block_cache[key] = this_delayed.key
                else:
                    block_cache[key] = key
        return block_cache

    def precompute(self, cache_dir=None, rows_per_scan=None, persist=False,
                   **kwargs):
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

        rows_per_scan = self._get_rows_per_scan(rows_per_scan)
        new_chunks = self._new_chunks(source_geo_def.lons, rows_per_scan)
        lons, lats = source_geo_def.get_lonlats(chunks=new_chunks)
        # run ll2cr to get column/row indexes
        # if chunk does not overlap target area then None is returned
        # otherwise a 3D array (2, y, x) of cols, rows are returned
        ll2cr_result = _call_mapped_ll2cr(lons, lats, target_geo_def)
        block_cache = self._fill_block_cache_with_ll2cr_results(
            ll2cr_result, lons.numblocks[0], lons.numblocks[1], persist)

        # save the dask arrays in the class instance cache
        self.cache = {
            'll2cr_result': ll2cr_result,
            'll2cr_blocks': block_cache,
        }
        return None

    def _get_input_tuples(self, data):
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

    @staticmethod
    def _generate_fornav_dask_tasks(out_chunks, ll2cr_blocks, task_name,
                                    input_name, target_geo_def, fill_value, kwargs):
        y_start = 0
        output_stack = {}
        for out_row_idx in range(len(out_chunks[0])):
            y_end = y_start + out_chunks[0][out_row_idx]
            x_start = 0
            for out_col_idx in range(len(out_chunks[1])):
                x_end = x_start + out_chunks[1][out_col_idx]
                y_slice = slice(y_start, y_end)
                x_slice = slice(x_start, x_end)
                for z_idx, ((_, in_row_idx, in_col_idx), ll2cr_block) in enumerate(ll2cr_blocks):
                    key = (task_name, z_idx, out_row_idx, out_col_idx)
                    output_stack[key] = (_delayed_fornav,
                                         ll2cr_block,
                                         target_geo_def, y_slice, x_slice,
                                         (input_name, in_row_idx, in_col_idx), fill_value, kwargs)
                x_start = x_end
            y_start = y_end
        return output_stack

    def _run_fornav_single(self, data, out_chunks, target_geo_def, fill_value, **kwargs):
        ll2cr_result = self.cache['ll2cr_result']
        ll2cr_blocks = self.cache['ll2cr_blocks'].items()
        ll2cr_numblocks = ll2cr_result.shape if isinstance(ll2cr_result, np.ndarray) else ll2cr_result.numblocks
        fornav_task_name = f"fornav-{data.name}-{ll2cr_result.name}"
        maximum_weight_mode = kwargs.setdefault('maximum_weight_mode', False)
        weight_sum_min = kwargs.setdefault('weight_sum_min', -1.0)
        output_stack = self._generate_fornav_dask_tasks(out_chunks,
                                                        ll2cr_blocks,
                                                        fornav_task_name,
                                                        data.name,
                                                        target_geo_def,
                                                        fill_value,
                                                        kwargs)

        dsk_graph = HighLevelGraph.from_collections(fornav_task_name,
                                                    output_stack,
                                                    dependencies=[data, ll2cr_result])
        stack_chunks = ((1,) * (ll2cr_numblocks[0] * ll2cr_numblocks[1]),) + out_chunks
        out_stack = da.Array(dsk_graph, fornav_task_name, stack_chunks, data.dtype)
        combine_fornav_with_kwargs = partial(
            _combine_fornav, maximum_weight_mode=maximum_weight_mode)
        average_fornav_with_kwargs = partial(
            _average_fornav, maximum_weight_mode=maximum_weight_mode,
            weight_sum_min=weight_sum_min, dtype=data.dtype,
            fill_value=fill_value)
        out = da.reduction(out_stack, _chunk_callable,
                           average_fornav_with_kwargs,
                           combine=combine_fornav_with_kwargs, axis=(0,),
                           dtype=data.dtype, concatenate=False)
        return out

    def compute(self, data, cache_id=None, rows_per_scan=None, chunks=None, fill_value=None,
                weight_count=10000, weight_min=0.01, weight_distance_max=1.0,
                weight_delta_max=10.0, weight_sum_min=-1.0,
                maximum_weight_mode=None, **kwargs):
        """Resample the data according to the precomputed X/Y coordinates."""
        # not used in this step
        kwargs.pop("persist", None)
        data_in, xr_obj = self._get_input_tuples(data)
        rows_per_scan = self._get_rows_per_scan(rows_per_scan)
        data_in = tuple(self._convert_to_dask(data_in, rows_per_scan))
        out_chunks = normalize_chunks(chunks or 'auto',
                                      shape=self.target_geo_def.shape,
                                      dtype=data.dtype)
        fornav_kwargs = kwargs.copy()
        maximum_weight_mode = self._handle_mwm(data, maximum_weight_mode)
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
            fill_value = self._get_default_fill(data_in[0])

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
        if isinstance(data, np.ndarray):
            return out.compute()
        return out

    @staticmethod
    def _handle_mwm(data, maximum_weight_mode):
        if np.issubdtype(data.dtype, np.integer):
            if maximum_weight_mode is None:
                return True
            elif not maximum_weight_mode:
                logger.warning("'maximum_weight_mode' is 'False' for integer "
                               "data. This is not recommended and integer "
                               "overflow may occur.")
        return maximum_weight_mode or False

    @staticmethod
    def _get_default_fill(data):
        if np.issubdtype(data.dtype, np.floating):
            fill_value = np.nan
        elif np.issubdtype(data.dtype, np.integer):
            fill_value = np.iinfo(data.dtype).max
        else:
            raise ValueError(
                "Unsupported input data type for EWA Resampling: {}".format(data.dtype))
        return fill_value

    def resample(self, data, cache_dir=None, mask_area=None,
                 rows_per_scan=None, persist=False, chunks=None, fill_value=None,
                 weight_count=10000, weight_min=0.01, weight_distance_max=1.0,
                 weight_delta_max=10.0, weight_sum_min=-1.0,
                 maximum_weight_mode=None):
        """Resample using an elliptical weighted averaging algorithm.

        This algorithm does **not** use caching or any externally provided data
        mask (unlike the 'nearest' resampler).
        See the :class:`~satpy.ewa.dask_ewa.DaskEWAResampler` class docstring
        for more information on how the algorithm works.

        .. note::

            This sets the default of 'mask_area' to False since it is
            not needed in EWA resampling currently.

        Args:
            data (numpy.ndarray, dask.array.Array, xarray.DataArray):
                Raster data to be resampled. Can be a numpy array, dask array,
                or xarray DataArray backed by a numpy or dask array. If the
                data is a numpy or dask array then only 2D (y, x) arrays are
                permitted. DataArray objects may be 2D or 3D where the third
                dimension is named "bands". Note that regardless of the input
                type, data is converted to a dask array for internal
                processing and converted back to the original data type on
                return.
            cache_dir (str, None): Not used by this resampler.
            mask_area (bool, None): Not used by this resampler.
            rows_per_scan (int, None): Number of array rows that represent a
                single scan of the instrument. If ``None`` (default), then
                the ``.attrs`` of the source swath longitude and latitude data
                is checked for this value if they are DataArray objects.
                Otherwise, this value must be provided. Decent results may be
                possible if this value is set to the total number of rows in
                the array. As a convenience, providing ``0`` will result in
                the total number of rows being used.
            persist (bool): Whether to persist (as in dask) the computations
                during precompute or compute them on the fly during compute.
                Persisting allows the resampler to determine which input
                chunks will overlap with the target area. This can greatly
                reduce the number of tasks and checks that will need to be
                computed in cases where it is known that only a small amount
                of input data will fall into the output area.
            chunks (tuple, int, dict, string): Chunk size of resulting dask
                array. See :func:`~dask.array.core.normalize_chunks` for more
                information.
            fill_value (int, float): Output value when no data is present.
                Defaults to ``numpy.nan`` for float types or the maximum
                value for any integer types.
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
        mask_area = False if mask_area is None else mask_area
        return super().resample(data, cache_dir=cache_dir,
                                mask_area=mask_area,
                                rows_per_scan=rows_per_scan,
                                persist=persist,
                                chunks=chunks,
                                fill_value=fill_value,
                                weight_count=weight_count,
                                weight_min=weight_min,
                                weight_distance_max=weight_distance_max,
                                weight_delta_max=weight_delta_max,
                                weight_sum_min=weight_sum_min,
                                maximum_weight_mode=maximum_weight_mode
                                )
