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
from collections import defaultdict
import dask.array as da
from dask.array import map_blocks
from dask.array.core import normalize_chunks
from .ewa import ll2cr, fornav
from ._fornav import fornav_weights_and_sums_wrapper
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.resampler import BaseResampler
from pyresample import CHUNK_SIZE
import dask
import numpy as np

try:
    import xarray as xr
except ImportError:
    # only used for some use cases
    xr = None

logger = logging.getLogger(__name__)


def _run_proj(data_x, data_y, crs_wkt):
    """Perform inverse projection."""
    target_proj = Proj(crs_wkt)
    return np.dstack(target_proj(data_x, data_y))


def get_target_coords(lons, lats, target_geo_def, dtype=np.float64):
    crs_wkt = target_geo_def.crs_wkt
    res = map_blocks(_run_proj, lons, lats, crs_wkt,
                     chunks=(lons.chunks[0], lons.chunks[1], 2),
                     new_axis=[2]).astype(dtype)
    return res[:, :, 0], res[:, :, 1]


def _call_ll2cr(lons, lats, target_geo_def, swath_usage=0):
    """Wrap ll2cr() for handling dask delayed calls better."""
    new_src = SwathDefinition(lons, lats)
    swath_points_in_grid, cols, rows = ll2cr(new_src, target_geo_def)
    if swath_points_in_grid == 0:
        return None
    return np.stack([cols, rows], axis=0)


def call_ll2cr(lons, lats, target_geo_def):
    res = da.map_blocks(_call_ll2cr, lons, lats,
                        target_geo_def, 0,
                        dtype=lons.dtype)
    return res


def _delayed_fornav(ll2cr_result, target_geo_def, y_slice, x_slice, data, kwargs):
    if ll2cr_result is None:
        # this source data doesn't fit in the target area at all
        return None
    cols = ll2cr_result[0]
    rows = ll2cr_result[1]
    # Adjust cols and rows for this sub-area
    subdef = target_geo_def[y_slice, x_slice]
    if x_slice.start != 0:
        cols = cols - x_slice.start
    if y_slice.start != 0:
        rows = rows - y_slice.start
    weights = np.zeros(subdef.shape, dtype=np.float32)
    accums = np.zeros(subdef.shape, dtype=np.float32)
    try:
        #         got_points, res = fornav(cols, rows, subdef, data, **kwargs)
        got_points = fornav_weights_and_sums_wrapper(
            cols, rows, data, weights, accums, np.nan, np.nan,
            **kwargs)
    except RuntimeError:
        return None
    if got_points:
        return None
    return weights, accums


def merge_fornav(out_chunk, fill_value, dtype, *output_stack, maximum_weight_mode=False):
    # TODO: Actually do stuff with this to sum and average pixels
    valid_stack = [x for x in output_stack if x is not None]
    if not valid_stack:
        return np.full(out_chunk, fill_value, dtype=dtype)
    weights = np.array([x[0] for x in valid_stack], copy=False)
    accums = np.array([x[1] for x in valid_stack], copy=False)
    #     stack = np.array(valid_stack, copy=False)
    #     res = np.nanmax(stack, axis=0)
    if maximum_weight_mode:
        raise NotImplementedError("maximum_weight_mode is not implemented yet")
    else:
        weights = np.nansum(weights, axis=0)
        accums = np.nansum(accums, axis=0)
    res = (accums / weights).astype(dtype)
    return res


def run_chunked_ewa(src_data_arr, target_geo_def, **kwargs):
    from pyresample.geometry import SwathDefinition, AreaDefinition
    src_arr = src_data_arr.data
    src_geo_def = src_data_arr.attrs['area']
    assert isinstance(src_geo_def, SwathDefinition)
    assert isinstance(target_geo_def, AreaDefinition)
    kwargs['rows_per_scan'] = src_data_arr.attrs['rows_per_scan']

    out_chunks = normalize_chunks(CHUNK_SIZE, target_geo_def.shape)
    output_stack = {}
    # get lons and lats and assume the source data is chunked per scan
    lons = src_geo_def.lons.data
    lons = lons.rechunk(src_arr.chunks)
    lats = src_geo_def.lats.data
    lats = lats.rechunk(src_arr.chunks)
    ll2cr_result = call_ll2cr(lons, lats, target_geo_def)
    y_start = 0
    dsk = {}
    for out_row_idx in range(len(out_chunks[0])):
        y_end = y_start + out_chunks[0][out_row_idx]
        x_start = 0
        for out_col_idx in range(len(out_chunks[1])):
            x_end = x_start + out_chunks[1][out_col_idx]
            y_slice = slice(y_start, y_end)
            x_slice = slice(x_start, x_end)
            z_idx = 0
            out_chunk_list = []
            for in_row_idx in range(src_arr.numblocks[0]):
                for in_col_idx in range(src_arr.numblocks[1]):
                    key = ("out_stack", z_idx, out_row_idx, out_col_idx)
                    # TODO: Call an updated fornav to get weights, image pixels
                    output_stack[key] = (_delayed_fornav,
                                         (ll2cr_result.name, in_row_idx, in_col_idx),
                                         target_geo_def, y_slice, x_slice,
                                         (src_arr.name, in_row_idx, in_col_idx), kwargs)
                    out_chunk_list.append(key)
                    z_idx += 1
            # TODO: Create two output stacks here, one for each nansum
            #       Then the final step will be a divide
            dsk[("out", out_row_idx, out_col_idx)] = (
            merge_fornav, (out_chunks[0][out_row_idx], out_chunks[1][out_col_idx]), np.nan, src_data_arr.dtype,
            *tuple(out_chunk_list))
            x_start = x_end
        y_start = y_end

    from dask.highlevelgraph import HighLevelGraph
    dsk.update(output_stack)
    dsk_graph = HighLevelGraph.from_collections('out', dsk, dependencies=[scn['I04'].data, ll2cr_rows, ll2cr_cols])
    name = 'out'
    out = da.Array(dsk_graph, name, out_chunks, src_arr.dtype)
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

        block_cache = {}
        for in_row_idx in range(lons.numblocks[0]):
            for in_col_idx in range(lons.numblocks[1]):
                ll2cr_block = ll2cr_result.blocks[in_row_idx, in_col_idx]
                key = (ll2cr_block.name, in_row_idx, in_col_idx)
                # XXX: instead of using the block, use the original name to reduce duplicate dask graph entries
                if kwargs.get('persist', False):
                    ll2cr_block = ll2cr_block.persist()
                # FUTURE: Compute early and don't include source blocks that are None
                result = ll2cr_block
                if result is not None:
                    block_cache[key] = result

        # save the dask arrays in the class instance cache
        self.cache = {
            'll2cr_result': ll2cr_result,
            'll2cr_blocks': block_cache,
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
        logger.debug("EWA resampling found %f%% of the grid covered" %
                     (grid_covered_ratio * 100))

        return res

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

    def compute(self, data, cache_id=None, chunks=None, fill_value=0,
                weight_count=10000, weight_min=0.01, weight_distance_max=1.0,
                weight_delta_max=1.0, weight_sum_min=-1.0,
                maximum_weight_mode=False, grid_coverage=0, **kwargs):
        """Resample the data according to the precomputed X/Y coordinates."""
        data_in, xr_obj = self._get_input_tuples(data, kwargs)
        rows_per_scan = self._get_rows_per_scan(kwargs)
        data_in = tuple(self._convert_to_dask(data_in, rows_per_scan))
        out_chunks = normalize_chunks(chunks or 'auto',
                                      shape=self.target_geo_def.shape,
                                      dtype=data.dtype)
        # TODO: iterate over tuple of 2D arrays
        data = data_in[0]
        y_start = 0
        dsk = {}
        output_stack = {}
        ll2cr_result = self.cache['ll2cr_result']
        ll2cr_blocks = self.cache['ll2cr_blocks'].items()
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
        for out_row_idx in range(len(out_chunks[0])):
            y_end = y_start + out_chunks[0][out_row_idx]
            x_start = 0
            for out_col_idx in range(len(out_chunks[1])):
                x_end = x_start + out_chunks[1][out_col_idx]
                y_slice = slice(y_start, y_end)
                x_slice = slice(x_start, x_end)
                z_idx = 0
                out_chunk_list = []
                for (ll2cr_name, in_row_idx, in_col_idx), ll2cr_block in ll2cr_blocks:
                    key = ("out_stack", z_idx, out_row_idx, out_col_idx)
                    # TODO: Call an updated fornav to get weights, image pixels
                    output_stack[key] = (_delayed_fornav,
                                         (ll2cr_result.name, in_row_idx, in_col_idx),
                                         self.target_geo_def, y_slice, x_slice,
                                         (data.name, in_row_idx, in_col_idx), fornav_kwargs)
                    out_chunk_list.append(key)
                    z_idx += 1
                # TODO: Create two output stacks here, one for each nansum
                #       Then the final step will be a divide
                dsk[("out", out_row_idx, out_col_idx)] = (
                    merge_fornav, (out_chunks[0][out_row_idx], out_chunks[1][out_col_idx]), np.nan, data.dtype,
                    *tuple(out_chunk_list))
                x_start = x_end
            y_start = y_end

        from dask.highlevelgraph import HighLevelGraph
        dsk.update(output_stack)
        dsk_graph = HighLevelGraph.from_collections('out', dsk, dependencies=[data, ll2cr_result])
        name = 'out'
        out = da.Array(dsk_graph, name, out_chunks, data.dtype)
        # if xr_obj is not None:
        #     # TODO: Rebuild xarray object
        #     out = xr.DataArray(out, attrs=xr_obj.attrs.copy())
        return out

# ewa_out = run_chunked_ewa(scn['I04'], area_def, weight_delta_max=40.0, weight_distance_max=2.0)