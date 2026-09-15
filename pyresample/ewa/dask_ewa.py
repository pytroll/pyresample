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
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from functools import partial
from typing import Any

import dask.array as da
import numpy as np
from dask.array.core import normalize_chunks
from dask.base import tokenize

from pyresample.ewa import ll2cr
from pyresample.ewa._fornav import fornav_weights_and_sums_wrapper, write_grid_image_single
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.resampler import BaseResampler

from ..future.resamplers.resampler import update_resampled_coords

try:
    import xarray as xr
except ImportError:
    # only used for some use cases
    xr = None

try:
    # Not public API, but stable since 2021 and what dask's own map_blocks
    # uses to pass per-block metadata (``block_info``) into a Blockwise layer.
    from dask.layers import ArrayValuesDep
except ImportError as err:  # pragma: no cover
    raise ImportError("pyresample's Dask EWA resampler requires 'dask.layers.ArrayValuesDep' "
                      "which is not available in the installed dask version") from err

logger = logging.getLogger(__name__)

# (row_min, row_max, col_min, col_max) in continuous grid-cell coordinates
Extent = tuple[float, float, float, float]
# chunk sizes along each output dimension (y, x), as returned by normalize_chunks
OutChunks = tuple[tuple[int, ...], ...]
# per fornav stack block (input row block, output row chunk, output column chunk):
# (output row slice, output column slice, whether the input block can overlap the output chunk)
FornavBlockMeta = tuple[slice, slice, bool]
# (shape, fill value, dtype) description of an array that has not been allocated
EmptyArrayInfo = tuple[tuple[int, ...], float, Any]


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


def _ll2cr_block_extent(
        ll2cr_block: np.ndarray | tuple,
        grid_shape: tuple[int, int],
        margin: float,
) -> Extent | None:
    """Compute row/column bounds for a single ll2cr block.

    Only points that can contribute to the target grid are considered. A
    swath pixel reaches at most ``margin`` grid cells from its ll2cr
    position (see ``weight_delta_max``/``weight_distance_max`` in fornav), so
    points outside ``[-margin, grid_size + margin]`` in either dimension are
    ignored. This matters for blocks that straddle the grid edge:
    ``ll2cr`` keeps out-of-grid points as finite far-off values, which would
    otherwise inflate the extent to cover the whole grid.

    Args:
        ll2cr_block: ll2cr output block as ``(cols, rows)`` arrays, or the
            empty sentinel returned by ``_call_ll2cr``.
        grid_shape: ``(rows, cols)`` shape of the target grid.
        margin: Non-negative margin in grid cells around the target grid.

    Returns:
        ``(row_min, row_max, col_min, col_max)`` as floats, or ``None`` when
        the block contains no finite coordinates within the padded grid.
    """
    # Empty ll2cr results: ((shape, fill, dtype), (shape, fill, dtype))
    if isinstance(ll2cr_block[0], tuple):
        return None

    cols = ll2cr_block[0]
    rows = ll2cr_block[1]
    grid_rows, grid_cols = grid_shape
    # NaN comparisons are False so non-finite points are excluded too
    valid = (
        (rows >= -margin) & (rows <= grid_rows + margin) &
        (cols >= -margin) & (cols <= grid_cols + margin)
    )
    if not np.any(valid):
        return None

    # 'where' avoids copying the valid points out of the block
    row_min = float(rows.min(where=valid, initial=np.inf))
    row_max = float(rows.max(where=valid, initial=-np.inf))
    col_min = float(cols.min(where=valid, initial=np.inf))
    col_max = float(cols.max(where=valid, initial=-np.inf))
    return row_min, row_max, col_min, col_max


def _ll2cr_block_extent_array(ll2cr_block: np.ndarray | tuple, grid_shape: tuple[int, int],
                              margin: float) -> np.ndarray:
    """Wrap ``_ll2cr_block_extent`` for ``map_blocks`` returning a ``(1, 1, 4)`` array (all NaN when empty)."""
    extent = _ll2cr_block_extent(ll2cr_block, grid_shape, margin)
    if extent is None:
        return np.full((1, 1, 4), np.nan, dtype=np.float64)
    return np.array(extent, dtype=np.float64).reshape((1, 1, 4))


def _compute_ll2cr_extents(ll2cr_result: da.Array, grid_shape: tuple[int, int], margin: float) -> np.ndarray:
    """Compute the extent of every ll2cr block in a single batched dask computation.

    Returns:
        ``(num_row_blocks, num_col_blocks, 4)`` float64 array of
        ``(row_min, row_max, col_min, col_max)`` for each block. Blocks with
        no usable points are all NaN.
    """
    num_row_blocks, num_col_blocks = ll2cr_result.numblocks
    extents = da.map_blocks(
        _ll2cr_block_extent_array, ll2cr_result, grid_shape, margin,
        chunks=((1,) * num_row_blocks, (1,) * num_col_blocks, (4,)),
        new_axis=2, dtype=np.float64, meta=np.array((), dtype=np.float64),
    )
    return extents.compute()


def _row_block_extent(extents: np.ndarray) -> Extent | None:
    """Union of the ``(num_col_blocks, 4)`` extents of one row of input blocks, or ``None`` if all are empty."""
    if np.isnan(extents[:, 0]).all():
        return None
    row_min, col_min = np.nanmin(extents[:, [0, 2]], axis=0)
    row_max, col_max = np.nanmax(extents[:, [1, 3]], axis=0)
    return float(row_min), float(row_max), float(col_min), float(col_max)


def _pad_bounds(bounds: Extent | None, overlap_margin: float) -> Extent | None:
    """Pad ll2cr bounds by a constant overlap margin.

    Args:
        bounds: ll2cr bounds tuple ``(row_min, row_max, col_min, col_max)``
            in continuous grid-cell coordinates as returned by ``ll2cr``,
            or ``None`` when the bounds are unknown.
        overlap_margin: Non-negative overlap margin in grid cells.

    Returns:
        Padded bounds tuple, or ``None`` when input bounds is ``None``.
    """
    if bounds is None:
        return None
    row_min, row_max, col_min, col_max = bounds
    return (
        row_min - overlap_margin,
        row_max + overlap_margin,
        col_min - overlap_margin,
        col_max + overlap_margin,
    )


def _chunk_intersects_bounds(bounds: Extent | None, y_slice: slice, x_slice: slice) -> bool:
    """Check whether a target chunk overlaps pre-padded ll2cr bounds.

    Args:
        bounds: ll2cr bounds tuple ``(row_min, row_max, col_min, col_max)``
            in continuous grid-cell coordinates, already padded for overlap,
            or ``None`` when the bounds are unknown (always intersects).
        y_slice: Output chunk rows as a ``[start, stop)`` slice in integer
            grid cells.
        x_slice: Output chunk columns as a ``[start, stop)`` slice in integer
            grid cells.

    Returns:
        ``True`` if the chunk intersects the bounds.
    """
    if bounds is None:
        return True
    row_min, row_max, col_min, col_max = bounds
    return (
        y_slice.stop > row_min and y_slice.start <= row_max and
        x_slice.stop > col_min and x_slice.start <= col_max
    )


def _fornav_block(
        ll2cr_blocks: list[np.ndarray | tuple],
        data_blocks: list[np.ndarray],
        block_meta: FornavBlockMeta,
        fill_value: float | int,
        kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | tuple[EmptyArrayInfo, EmptyArrayInfo]:
    """Compute fornav weights and accumulations of one row of input blocks for one output chunk.

    Args:
        ll2cr_blocks: ll2cr output blocks along the input column axis, each
            either a ``(2, y, x)`` array of ``cols, rows`` or the empty
            sentinel returned by ``_call_ll2cr``.
        data_blocks: Input data blocks matching ``ll2cr_blocks``.
        block_meta: ``(y_slice, x_slice, overlaps)`` for this output chunk.
            When ``overlaps`` is ``False`` the input blocks were determined
            to not contribute to this output chunk and nothing is computed.
        fill_value: Fill value of the input data.
        kwargs: Keyword arguments for ``fornav_weights_and_sums_wrapper``.

    Returns:
        ``(weights, accums)`` arrays, or a pair of ``(shape, 0, dtype)``
        descriptions when no input data landed in this output chunk.
    """
    y_slice, x_slice, overlaps = block_meta
    shape = (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start)
    weights_dtype = np.float32
    accums_dtype = np.float32
    empty_weights = (shape, 0, weights_dtype)
    empty_accums = (shape, 0, accums_dtype)
    if not overlaps:
        return empty_weights, empty_accums

    weights = np.zeros(shape, dtype=weights_dtype)
    accums = np.zeros(shape, dtype=accums_dtype)
    got_points = False
    for ll2cr_block, data_block in zip(ll2cr_blocks, data_blocks, strict=True):
        # Empty ll2cr results: ((shape, fill, dtype), (shape, fill, dtype))
        if isinstance(ll2cr_block[0], tuple):
            # this source data doesn't fit in the target area at all
            continue
        cols = ll2cr_block[0]
        rows = ll2cr_block[1]
        if x_slice.start != 0:
            cols = cols - x_slice.start
        if y_slice.start != 0:
            rows = rows - y_slice.start
        try:
            # accumulates in-place so multiple input blocks can share the output arrays
            got_points |= fornav_weights_and_sums_wrapper(
                cols, rows, data_block, weights, accums, fill_value, fill_value,
                **kwargs)
        except RuntimeError:
            continue
    if not got_points:
        return empty_weights, empty_accums
    return weights, accums


def _chunk_callable(x_chunk, axis, keepdims, **kwargs):
    """No-op for reduction call."""
    return x_chunk


def _sum_arrays(arrays):
    """Sum arrays with one initial copy and in-place accumulation.

    Args:
        arrays: Non-empty sequence of NumPy arrays with compatible shapes.

    Returns:
        Element-wise sum as a NumPy array.
    """
    total = arrays[0].copy()
    for arr in arrays[1:]:
        total += arr
    return total


def _combine_fornav(x_chunk, axis, keepdims, computing_meta=False,
                    maximum_weight_mode=False):
    if computing_meta or _is_empty_chunk(x_chunk):
        # single "empty" chunk
        return x_chunk
    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]
    if not len(x_chunk):
        return x_chunk
    # if the first element is not an array it is:
    # (empty_tuple_description, empty_tuple_description)
    valid_chunks = [x for x in x_chunk if not isinstance(x[0], tuple)]
    if not len(valid_chunks):
        if keepdims:
            # split step - return "empty" chunk placeholder
            return x_chunk[0]
        return np.full(*x_chunk[0][0]), np.full(*x_chunk[0][1])
    if len(valid_chunks) == 1:
        return valid_chunks[0]
    weights = [x[0] for x in valid_chunks]
    accums = [x[1] for x in valid_chunks]
    if maximum_weight_mode:
        weights = np.array(weights)
        accums = np.array(accums)
        max_indexes = np.expand_dims(np.argmax(weights, axis=0), axis=0)
        weights = np.take_along_axis(weights, max_indexes, axis=0).squeeze(axis=0)
        accums = np.take_along_axis(accums, max_indexes, axis=0).squeeze(axis=0)
        return weights, accums
    return _sum_arrays(weights), _sum_arrays(accums)


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

    def _ll2cr_cache_matches(self, rows_per_scan: int, persist: bool, extent_margin: float) -> bool:
        return (
            self.cache.get('rows_per_scan') == rows_per_scan and
            self.cache.get('persist') == persist and
            self.cache.get('extent_margin') == extent_margin
        )

    def precompute(
            self,
            cache_dir: str | None = None,
            rows_per_scan: int | None = None,
            persist: bool = False,
            weight_distance_max: float = 1.0,
            weight_delta_max: float = 10.0,
            **kwargs: Any,
    ) -> None:
        """Generate row and column arrays and store it for later use."""
        rows_per_scan = self._get_rows_per_scan(rows_per_scan)
        extent_margin = _get_extent_margin(weight_delta_max, weight_distance_max)
        if self._ll2cr_cache_matches(rows_per_scan, persist, extent_margin):
            # this resampler should be used for one SwathDefinition
            # no need to recompute matching ll2cr output again
            return None

        if kwargs.get('mask') is not None:
            logger.warning("'mask' parameter has no affect during EWA "
                           "resampling")

        if cache_dir:
            logger.warning("'cache_dir' is not used by EWA resampling")

        new_chunks = self._new_chunks(self.source_geo_def.lons, rows_per_scan)
        lons, lats = self.source_geo_def.get_lonlats(chunks=new_chunks)
        # run ll2cr to convert input lon/lat coordinates to column/row indexes in the target area
        # if a chunk does not overlap the target area then a pair of
        # (shape, fill, dtype) tuples is returned instead of an array
        # otherwise a 3D array (2, y, x) of cols, rows is returned
        ll2cr_result = _call_mapped_ll2cr(lons, lats, self.target_geo_def)
        extents = None
        if persist:
            # keeps the same name and block keys, but the blocks are now in memory
            ll2cr_result = ll2cr_result.persist()
            # a cheap pass over the in-memory blocks that returns 4 floats per block
            extents = _compute_ll2cr_extents(ll2cr_result, self.target_geo_def.shape, extent_margin)

        # save the dask arrays in the class instance cache
        self.cache = {
            'll2cr_result': ll2cr_result,
            'extents': extents,
            'rows_per_scan': rows_per_scan,
            'persist': persist,
            'extent_margin': extent_margin,
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

    def _convert_to_dask(self, data_in):
        # match the chunks of the cached ll2cr result so input data blocks
        # and ll2cr blocks correspond one-to-one
        new_chunks = self.cache['ll2cr_result'].chunks
        for data in data_in:
            if not isinstance(data, da.Array):
                yield da.from_array(data, chunks=new_chunks)
            else:
                yield data.rechunk(new_chunks)

    def _run_fornav_single(
            self,
            data: da.Array,
            out_chunks: OutChunks,
            target_geo_def: AreaDefinition,
            fill_value: float | int,
            **kwargs: Any,
    ) -> da.Array:
        ll2cr_result = self.cache['ll2cr_result']
        extents = self.cache['extents']
        overlap_margin = _get_extent_margin(
            kwargs.get("weight_delta_max", 0.0),
            kwargs.get("weight_distance_max", 0.0),
        )
        if overlap_margin > self.cache['extent_margin']:
            # block extents were computed with a smaller margin; precompute
            # invalidates its cache on a margin change so this is a bug
            raise RuntimeError(
                "Cached ll2cr block extents used a margin of "
                f"{self.cache['extent_margin']} but fornav requires {overlap_margin}. "
                "Call 'precompute' with the same 'weight_delta_max' and "
                "'weight_distance_max' as 'compute'.")

        num_row_blocks = ll2cr_result.numblocks[0]
        if extents is None:
            # nothing is known about the ll2cr blocks until they are computed
            kept_row_blocks = list(range(num_row_blocks))
            block_extents: list[Extent | None] = [None] * num_row_blocks
        else:
            block_extents = [_row_block_extent(extents[row_idx]) for row_idx in range(num_row_blocks)]
            kept_row_blocks = [row_idx for row_idx, extent in enumerate(block_extents) if extent is not None]
            block_extents = [block_extents[row_idx] for row_idx in kept_row_blocks]
        if not kept_row_blocks:
            return da.full(target_geo_def.shape, fill_value, dtype=data.dtype,
                           chunks=out_chunks)
        if len(kept_row_blocks) != num_row_blocks:
            # pure alias layers referring to the kept blocks, no slicing tasks
            ll2cr_result = ll2cr_result.blocks[kept_row_blocks, :]
            data = data.blocks[kept_row_blocks, :]

        maximum_weight_mode = kwargs.setdefault('maximum_weight_mode', False)
        weight_sum_min = kwargs.setdefault('weight_sum_min', -1.0)
        block_meta = _fornav_block_meta(out_chunks, block_extents, overlap_margin)
        out_stack = _fornav_stack(ll2cr_result, data, out_chunks, block_meta, fill_value, kwargs)
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
        # the geolocation and data share one instrument scan layout;
        # use what precompute validated and cached
        rows_per_scan = self.cache['rows_per_scan']
        data_in = tuple(self._convert_to_dask(data_in))
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
                When ``True`` the ll2cr result is computed once during
                ``precompute`` and reused by every later ``compute``. Input
                chunks that do not overlap the target area are dropped and
                input/output chunk pairs that cannot overlap are skipped.
                This can greatly reduce the number of tasks and checks that
                will need to be computed in cases where it is known that
                only a small amount of input data will fall into the output
                area. The persisted result is invalidated (recomputed) if a
                later call uses different ``rows_per_scan``,
                ``weight_delta_max``, or ``weight_distance_max`` values.
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


def _get_extent_margin(weight_delta_max: float, weight_distance_max: float) -> float:
    """Get the maximum reach of a swath pixel in grid cells.

    See ``_fornav_templates.cpp``: the ellipse half-widths are clamped to
    ``weight_delta_max`` except for pixels with bad neighboring
    geolocation which use ``weight_distance_max``.
    """
    return max(float(weight_delta_max), float(weight_distance_max), 0.0)


def _fornav_block_meta(
        out_chunks: OutChunks,
        block_extents: Sequence[Extent | None],
        overlap_margin: float,
) -> dict[tuple[int, int, int], FornavBlockMeta]:
    """Determine output chunk slices and overlap for every fornav stack block.

    Args:
        out_chunks: Output chunk sizes ``((y0, y1, ...), (x0, x1, ...))``.
        block_extents: Extent of each input row block that will be included
            in the fornav stack. ``None`` means the extent is unknown and
            the block is assumed to overlap every output chunk.
        overlap_margin: Maximum reach of a swath pixel in grid cells (see
            ``_get_extent_margin``).

    Returns:
        Mapping from ``(input row block, output row chunk, output column chunk)``
        indexes to ``(y_slice, x_slice, overlaps)``. Every block has an
        entry: ``da.blockwise`` creates a task for every block regardless
        and a missing entry is silently replaced by the block index by dask.
        Blocks with ``overlaps=False`` are cheap tasks that return an empty
        result without touching the input blocks.
    """
    block_bounds = [_pad_bounds(extent, overlap_margin) for extent in block_extents]
    y_starts = np.cumsum((0,) + tuple(out_chunks[0]))
    x_starts = np.cumsum((0,) + tuple(out_chunks[1]))
    block_meta = {}
    for out_row_idx in range(len(out_chunks[0])):
        y_slice = slice(int(y_starts[out_row_idx]), int(y_starts[out_row_idx + 1]))
        for out_col_idx in range(len(out_chunks[1])):
            x_slice = slice(int(x_starts[out_col_idx]), int(x_starts[out_col_idx + 1]))
            for z_idx, bounds in enumerate(block_bounds):
                overlaps = _chunk_intersects_bounds(bounds, y_slice, x_slice)
                block_meta[(z_idx, out_row_idx, out_col_idx)] = (y_slice, x_slice, overlaps)
    return block_meta


def _fornav_stack(
        ll2cr_result: da.Array,
        data: da.Array,
        out_chunks: OutChunks,
        block_meta: dict[tuple[int, int, int], FornavBlockMeta],
        fill_value: float | int,
        kwargs: dict[str, Any],
) -> da.Array:
    """Build the ``(input row block, y, x)`` stack of fornav weights and accumulations.

    Each block of the result is one ``(weights, accums)`` pair (or a pair of
    empty array descriptions) for one input row block and one output chunk.
    The input column block axis is contracted so ``_fornav_block`` receives
    all of the blocks of one input row at once.
    """
    num_row_blocks = len(ll2cr_result.chunks[0])
    stack_chunks = ((1,) * num_row_blocks,) + tuple(out_chunks)
    meta_dep = ArrayValuesDep(stack_chunks, block_meta)
    name = "fornav-" + tokenize(ll2cr_result.name, data.name, out_chunks, block_meta, fill_value, kwargs)
    return da.blockwise(
        _fornav_block, 'ryx',
        ll2cr_result, 'rc',
        data, 'rc',
        meta_dep, 'ryx',
        new_axes={'y': out_chunks[0], 'x': out_chunks[1]},
        adjust_chunks={'r': 1},
        # blocks are (weights, accums) tuples, never concatenate them
        concatenate=False,
        # unify_chunks must not touch the ll2cr result whose blocks are not 2D
        align_arrays=False,
        dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
        name=name,
        fill_value=fill_value,
        kwargs=kwargs,
    )
