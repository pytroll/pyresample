#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-2019
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Implementation of the gradient search algorithm as described by Trishchenko."""
from __future__ import annotations

import logging
import warnings
from functools import wraps

import dask
import dask.array as da
import numpy as np
import pyproj
import xarray as xr
from shapely.geometry import Polygon

from pyresample import CHUNK_SIZE
from pyresample.geometry import AreaDefinition, SwathDefinition, get_geostationary_bounding_box_in_lonlats
from pyresample.gradient._gradient_search import one_step_gradient_indices, one_step_gradient_search
from pyresample.resampler import BaseResampler, resample_blocks

logger = logging.getLogger(__name__)


def GradientSearchResampler(source_geo_def, target_geo_def):
    """Create a gradient search resampler."""
    warnings.warn("`GradientSearchResampler` is deprecated, please use "
                  "`create_gradient_search_resampler` instead.",
                  DeprecationWarning, stacklevel=2)
    return create_gradient_search_resampler(source_geo_def, target_geo_def)


def create_gradient_search_resampler(source_geo_def, target_geo_def):
    """Create a gradient search resampler."""
    if (is_area_to_area(source_geo_def, target_geo_def) or
        is_swath_to_area(source_geo_def, target_geo_def) or
        is_area_to_swath(source_geo_def, target_geo_def)):
        return ResampleBlocksGradientSearchResampler(source_geo_def, target_geo_def)
    raise NotImplementedError


def is_area_to_area(source_geo_def, target_geo_def):
    """Check if source is area and target is area."""
    return isinstance(source_geo_def, AreaDefinition) and isinstance(target_geo_def, AreaDefinition)


def is_swath_to_area(source_geo_def, target_geo_def):
    """Check if source is swath and target is area."""
    return isinstance(source_geo_def, SwathDefinition) and isinstance(target_geo_def, AreaDefinition)


def is_area_to_swath(source_geo_def, target_geo_def):
    """Check if source is area and targed is swath."""
    return isinstance(source_geo_def, AreaDefinition) and isinstance(target_geo_def, SwathDefinition)


def _gradient_resample_data(src_data, src_x, src_y,
                            src_gradient_xl, src_gradient_xp,
                            src_gradient_yl, src_gradient_yp,
                            dst_x, dst_y,
                            method='bilinear'):
    """Resample using gradient search."""
    _check_input_coordinates(dst_x, dst_y,
                             src_gradient_xl, src_gradient_xp,
                             src_gradient_yl, src_gradient_yp,
                             src_x, src_y)
    if src_data.ndim != 3 or src_data.shape[1:] != src_x.shape:
        raise ValueError("Malformed input data.")

    image = one_step_gradient_search(src_data, src_x, src_y,
                                     src_gradient_xl, src_gradient_xp,
                                     src_gradient_yl, src_gradient_yp,
                                     dst_x, dst_y,
                                     method=method)
    return image


def _gradient_resample_indices(src_x, src_y,
                               src_gradient_xl, src_gradient_xp,
                               src_gradient_yl, src_gradient_yp,
                               dst_x, dst_y):
    """Return indices computed using gradient search."""
    _check_input_coordinates(dst_x, dst_y,
                             src_gradient_xl, src_gradient_xp,
                             src_gradient_yl, src_gradient_yp,
                             src_x, src_y)

    indices_xy = one_step_gradient_indices(src_x, src_y,
                                           src_gradient_xl, src_gradient_xp,
                                           src_gradient_yl, src_gradient_yp,
                                           dst_x, dst_y)
    return indices_xy


def _check_input_coordinates(dst_x, dst_y,
                             src_gradient_xl, src_gradient_xp,
                             src_gradient_yl, src_gradient_yp,
                             src_x, src_y):
    if (src_x.ndim != 2 or
            src_y.ndim != 2 or
            src_gradient_xl.ndim != 2 or
            src_gradient_xp.ndim != 2 or
            src_gradient_yl.ndim != 2 or
            src_gradient_yp.ndim != 2 or
            dst_x.ndim != 2 or
            dst_y.ndim != 2):
        raise ValueError("Wrong number of dimensions.")
    source_shapes_equal = (src_x.shape == src_y.shape ==
                           src_gradient_xl.shape == src_gradient_xp.shape ==
                           src_gradient_yl.shape == src_gradient_yp.shape)
    if not source_shapes_equal:
        raise ValueError("Source arrays should all have the same shape")

    target_shapes_equal = (dst_x.shape == dst_y.shape)
    if not target_shapes_equal:
        raise ValueError("Target arrays should all have the same shape")


def parallel_gradient_search(data, src_x, src_y, dst_x, dst_y,
                             src_gradient_xl, src_gradient_xp,
                             src_gradient_yl, src_gradient_yp,
                             dst_mosaic_locations, dst_slices,
                             **kwargs):
    """Run gradient search in parallel in input area coordinates."""
    method = kwargs.get('method', 'bilinear')
    # Determine the number of bands
    bands = np.array([arr.shape[0] for arr in data if arr is not None])
    num_bands = np.max(bands)
    if np.any(bands != num_bands):
        raise ValueError("All source data chunks have to have the same number of bands")
    chunks = {}
    is_pad = False
    # Collect co-located target chunks
    for i, arr in enumerate(data):
        if arr is None:
            is_pad = True
            res = da.full((num_bands, dst_slices[i][1] - dst_slices[i][0],
                           dst_slices[i][3] - dst_slices[i][2]), np.nan)
        else:
            is_pad = False
            res = dask.delayed(_gradient_resample_data)(
                arr,
                src_x[i], src_y[i],
                src_gradient_xl[i], src_gradient_xp[i],
                src_gradient_yl[i], src_gradient_yp[i],
                dst_x[i], dst_y[i],
                method=method)
            res = da.from_delayed(res, (num_bands, ) + dst_x[i].shape,
                                  meta=np.array((), dtype=arr.dtype),
                                  dtype=arr.dtype)
        if dst_mosaic_locations[i] in chunks:
            if not is_pad:
                chunks[dst_mosaic_locations[i]].append(res)
        else:
            chunks[dst_mosaic_locations[i]] = [res, ]

    return _concatenate_chunks(chunks)


def _concatenate_chunks(chunks):
    """Concatenate chunks to full output array."""
    # Form the full array
    col, res = [], []
    prev_y = 0
    for y, x in sorted(chunks):
        if len(chunks[(y, x)]) > 1:
            chunk = da.nanmax(da.stack(chunks[(y, x)], axis=-1), axis=-1)
        else:
            chunk = chunks[(y, x)][0]
        if y == prev_y:
            col.append(chunk)
            continue
        res.append(da.concatenate(col, axis=1))
        col = [chunk]
        prev_y = y
    res.append(da.concatenate(col, axis=1))

    res = da.concatenate(res, axis=2)

    return res


def _fill_in_coords(target_geo_def, data_coords, data_dims):
    try:
        x_coord, y_coord = target_geo_def.get_proj_vectors()
    except AttributeError:
        return None
    coords = []
    for key in data_dims:
        if key == 'x':
            coords.append(x_coord)
        elif key == 'y':
            coords.append(y_coord)
        else:
            coords.append(data_coords[key])
    return coords


def ensure_data_array(func):
    """Ensure the data is an instance of an xarray.DataArray with correct dimensions."""
    @wraps(func)
    def wrapper(self, data, *args, **kwargs):
        if not isinstance(data, xr.DataArray):
            if data.ndim != 2:
                raise TypeError("Use a xarray.DataArray to label the dimensions"
                                " of arrays with other than two dimensions.")
            else:
                data = xr.DataArray(data, dims=["y", "x"])
        dims = data.dims
        data = data.transpose(..., "y", "x")
        return func(self, data, *args, **kwargs).transpose(*dims)
    return wrapper


class ResampleBlocksGradientSearchResampler(BaseResampler):
    """Resample using gradient search based bilinear interpolation, using `resample_blocks` for lazy processing."""

    def __init__(self, source_geo_def, target_geo_def):
        """Init GradientResampler."""
        if isinstance(source_geo_def, SwathDefinition):
            source_geo_def.lons = source_geo_def.lons.persist()
            source_geo_def.lats = source_geo_def.lats.persist()
        super().__init__(source_geo_def, target_geo_def)
        self.indices_xy = None

    def precompute(self, transform_step=None, transform_tolerance=0.01, **kwargs):
        """Precompute resampling parameters.

        Args:
            transform_step: When the source and the target are both areas, transform only every
                ``transform_step``-th target row and column to the source projection and interpolate the rest.
                ``None`` (the default) transforms every target pixel exactly.
            transform_tolerance: Maximum accepted interpolation error, in source pixels, when ``transform_step``
                is used. Parts of the target where the error is larger, or which are close to invalid
                coordinates, are transformed exactly.
            kwargs: Ignored, accepted for compatibility with the other resamplers.

        The indices are computed only once, so the arguments of the first call are the ones that are used.
        """
        if self.indices_xy is None:
            self.indices_xy = resample_blocks(gradient_resampler_indices_block,
                                              self.source_geo_def, [], self.target_geo_def,
                                              chunk_size=(2, CHUNK_SIZE, CHUNK_SIZE), dtype=float,
                                              transform_step=transform_step,
                                              transform_tolerance=transform_tolerance)

    @ensure_data_array
    def compute(self, data, method="bilinear", cache_id=None, **kwargs):
        """Perform the resampling."""
        if method == "bilinear":
            fun = block_bilinear_interpolator
        elif method in ["nearest_neighbour", "nn"]:
            fun = block_nn_interpolator
        else:
            raise ValueError(f"Unrecognized interpolation method {method} for gradient resampling.")

        chunks = list(data.shape[:-2]) + [CHUNK_SIZE, CHUNK_SIZE]

        res = resample_blocks(fun, self.source_geo_def, [data.data], self.target_geo_def,
                              dst_arrays=[self.indices_xy],
                              chunk_size=chunks, dtype=data.dtype, **kwargs)

        coords = _fill_in_coords(self.target_geo_def, data.coords, data.dims)

        res = xr.DataArray(res, attrs=data.attrs.copy(), dims=data.dims, coords=coords)
        res.attrs["area"] = self.target_geo_def
        return res


def ensure_3d_data(func):
    """Ensure the data is in three dimensions."""
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        """Wrap around the original function."""
        if data.ndim == 2:
            data_3d = data[np.newaxis, :, :]
        else:
            data_3d = data

        resampled = func(data_3d, *args, **kwargs)

        if data.ndim == 2:
            resampled = resampled.squeeze(0)
        return resampled

    wrapper.__doc__ += "\n\nThe input data can be 2d, or 3d with the two last axes being respectively `y` and `x`."
    return wrapper


@ensure_3d_data
def gradient_resampler(data, source_area, target_area, method='bilinear'):
    """Do the gradient search resampling."""
    dst_coords, src_gradients, src_coords = _get_coordinates_in_same_projection(source_area, target_area)
    dst_x, dst_y = dst_coords
    src_gradient_xl, src_gradient_xp, src_gradient_yl, src_gradient_yp = src_gradients
    src_x, src_y = src_coords

    return _gradient_resample_data(data, src_x, src_y,
                                   src_gradient_xl, src_gradient_xp,
                                   src_gradient_yl, src_gradient_yp,
                                   dst_x, dst_y,
                                   method=method)


def gradient_resampler_indices_block(block_info, **kwargs):
    """Do the gradient search resampling using block_info for areas, returning the resulting indices."""
    source_area = block_info[0]["area"]
    target_area = block_info[None]["area"]
    return gradient_resampler_indices(source_area, target_area, block_info, **kwargs)


def gradient_resampler_indices(source_area, target_area, block_info=None, transform_step=None,
                               transform_tolerance=0.01, **kwargs):
    """Do the gradient search resampling, returning the resulting indices."""
    dst_coords, src_gradients, src_coords = _get_coordinates_in_same_projection(
        source_area, target_area, transform_step=transform_step, transform_tolerance=transform_tolerance)
    dst_x, dst_y = dst_coords
    src_gradient_xl, src_gradient_xp, src_gradient_yl, src_gradient_yp = src_gradients
    src_x, src_y = src_coords

    indices_xy = _gradient_resample_indices(src_x, src_y,
                                            src_gradient_xl, src_gradient_xp,
                                            src_gradient_yl, src_gradient_yp,
                                            dst_x, dst_y)

    if block_info:
        y_slice, x_slice = block_info[0]["array-location"][-2:]
        indices_xy[0, :, :] += x_slice.start
        indices_xy[1, :, :] += y_slice.start

    return indices_xy


def _get_coordinates_in_same_projection(source_area, target_area, transform_step=None, transform_tolerance=0.01):
    target_crs = target_area.crs
    try:
        src_coords, src_gradients = _get_area_coordinates_and_gradients(source_area)
        work_crs = source_area.crs
        # the target coordinates are transformed to the source projection, so the error can be given in source pixels
        abs_tolerance = (transform_tolerance * abs(source_area.resolution[0]),
                         transform_tolerance * abs(source_area.resolution[1]))
    except AttributeError:
        # source is a swath definition, use target crs instead
        lons, lats = source_area.get_lonlats()
        src_x, src_y = da.compute(lons, lats)
        trans = pyproj.Transformer.from_crs(source_area.crs, target_crs, always_xy=True)
        src_x, src_y = trans.transform(src_x, src_y)
        work_crs = target_crs
        src_gradient_xl, src_gradient_xp = np.gradient(src_x, axis=[0, 1])
        src_gradient_yl, src_gradient_yp = np.gradient(src_y, axis=[0, 1])
        src_coords = (src_x, src_y)
        src_gradients = (src_gradient_xl, src_gradient_xp, src_gradient_yl, src_gradient_yp)
        # the work crs is the target crs, the target coordinates don't need an expensive transform
        transform_step = None
    transformer = pyproj.Transformer.from_crs(target_crs, work_crs, always_xy=True)
    try:
        if transform_step is not None and transform_step > 1:
            dst_x, dst_y = _transform_area_coordinates_coarsely(transformer, target_area, transform_step,
                                                                abs_tolerance)
        else:
            dst_x, dst_y = transformer.transform(*target_area.get_proj_coords())
    except AttributeError:
        # target is a swath definition
        lons, lats = target_area.get_lonlats()
        dst_x, dst_y = transformer.transform(*da.compute(lons, lats))
    return (dst_x, dst_y), src_gradients, src_coords


def _get_area_coordinates_and_gradients(source_area):
    """Get the projection coordinates and their gradients for an area source.

    The coordinates of an area are an outer product of two 1D vectors, so the gradients along lines of x and along
    pixels of y are zero, and the other two only vary along one axis. Everything is therefore computed in 1D and
    returned as read-only broadcast views of the full 2D shape, which is identical to, but much cheaper than,
    calling ``np.gradient`` on the full ``get_proj_coords()`` arrays.
    """
    x_vec, y_vec = source_area.get_proj_vectors()
    shape = (y_vec.size, x_vec.size)
    src_x = np.broadcast_to(x_vec[np.newaxis, :], shape)
    src_y = np.broadcast_to(y_vec[:, np.newaxis], shape)
    zeros = np.broadcast_to(np.zeros((), dtype=x_vec.dtype), shape)
    src_gradient_xp = np.broadcast_to(np.gradient(x_vec)[np.newaxis, :], shape)
    src_gradient_yl = np.broadcast_to(np.gradient(y_vec)[:, np.newaxis], shape)
    return (src_x, src_y), (zeros, src_gradient_xp, src_gradient_yl, zeros)


def _transform_area_coordinates_coarsely(transformer, target_area, step, abs_tolerance):
    """Transform the projection coordinates of an area on a coarse grid and interpolate the rest.

    Every ``step``-th row and column (and the last ones) are transformed exactly, and the other pixels are
    bilinearly interpolated from them. The coarse grid splits the area in cells. The pixels of a cell are
    interpolated when all four corners of the cell are valid and the interpolated coordinates at the middle of the
    cell differ from the exactly transformed ones by at most ``abs_tolerance`` (x, y), in units of the destination
    projection of ``transformer``. The pixels of a cell are set to ``inf`` when none of its corners, nor any corner
    of its eight neighbours, is valid. All other cells are retried with half the step, down to a step of 1 where
    the remaining pixels are transformed exactly.
    """
    x_vec, y_vec = target_area.get_proj_vectors()
    if x_vec.size < 2 or y_vec.size < 2:
        return transformer.transform(*np.meshgrid(x_vec, y_vec))
    dst_x = np.empty((y_vec.size, x_vec.size))
    dst_y = np.empty((y_vec.size, x_vec.size))
    pending = None
    while step > 1:
        pending = _interpolate_coarse_level(transformer, x_vec, y_vec, step, abs_tolerance, dst_x, dst_y, pending)
        if not pending.any():
            return dst_x, dst_y
        step //= 2
    rows, cols = np.nonzero(pending)
    dst_x[rows, cols], dst_y[rows, cols] = transformer.transform(x_vec[cols], y_vec[rows])
    return dst_x, dst_y


def _interpolate_coarse_level(transformer, x_vec, y_vec, step, abs_tolerance, dst_x, dst_y, pending):
    """Fill in the pending pixels that can be resolved with the given step, and return the ones still pending.

    ``pending`` is a boolean mask of the pixels to resolve, ``None`` meaning all of them.
    """
    x_axis = _get_coarse_cells(x_vec.size, step)
    y_axis = _get_coarse_cells(y_vec.size, step)
    x_cells, x_weights, x_corners = x_axis
    y_cells, y_weights, y_corners = y_axis
    if pending is None:
        needed = np.ones((y_corners.size - 1, x_corners.size - 1), dtype=bool)
    else:
        needed = np.logical_or.reduceat(np.logical_or.reduceat(pending, y_corners[:-1], axis=0),
                                        x_corners[:-1], axis=1)

    coarse_x, coarse_y = _transform_needed_corners(transformer, x_vec[x_corners], y_vec[y_corners], _dilate(needed))
    valid_corners = np.isfinite(coarse_x) & np.isfinite(coarse_y)
    num_valid_corners = (valid_corners[:-1, :-1].astype(int) + valid_corners[1:, :-1] +
                         valid_corners[:-1, 1:] + valid_corners[1:, 1:])
    accepted = needed & (num_valid_corners == 4)
    accepted[accepted] = _is_accurate_at_middles(transformer, x_vec, y_vec, coarse_x, coarse_y, abs_tolerance,
                                                 np.nonzero(accepted), x_axis, y_axis)
    invalid = needed & ~_dilate(num_valid_corners > 0)

    def to_pixels(cell_mask):
        return _expand_cells_to_pixels(cell_mask, x_corners, y_corners, dst_x.shape)

    interpolated = to_pixels(accepted)
    invalid_pixels = to_pixels(invalid)
    if pending is not None:
        interpolated &= pending
        invalid_pixels &= pending
    num_interpolated = np.count_nonzero(interpolated)
    if num_interpolated > dst_x.size // 4:
        with np.errstate(invalid="ignore"):
            for coarse, dst in ((coarse_x, dst_x), (coarse_y, dst_y)):
                full = _interpolate_separably(coarse, x_cells, x_weights, y_cells, y_weights)
                if pending is None:
                    dst[:] = full
                else:
                    dst[interpolated] = full[interpolated]
    elif num_interpolated:
        rows, cols = np.nonzero(interpolated)
        cell_rows, cell_cols = y_cells[rows], x_cells[cols]
        wy, wx = y_weights[rows], x_weights[cols]
        for coarse, dst in ((coarse_x, dst_x), (coarse_y, dst_y)):
            dst[rows, cols] = _interpolate_in_cells(coarse, cell_rows, cell_cols, wy, wx)
    dst_x[invalid_pixels] = np.inf
    dst_y[invalid_pixels] = np.inf
    still_pending = ~(interpolated | invalid_pixels)
    if pending is not None:
        still_pending &= pending
    return still_pending


def _expand_cells_to_pixels(cell_mask, x_corners, y_corners, shape):
    """Expand a mask of coarse cells to a mask of the pixels in them."""
    y_counts = np.diff(np.r_[y_corners[:-1], shape[0]])
    x_counts = np.diff(np.r_[x_corners[:-1], shape[1]])
    return np.repeat(np.repeat(cell_mask, y_counts, axis=0), x_counts, axis=1)


def _interpolate_in_cells(coarse, cell_rows, cell_cols, wy, wx):
    """Bilinearly interpolate within the given cells of a coarse grid."""
    return ((1 - wy) * ((1 - wx) * coarse[cell_rows, cell_cols] + wx * coarse[cell_rows, cell_cols + 1]) +
            wy * ((1 - wx) * coarse[cell_rows + 1, cell_cols] + wx * coarse[cell_rows + 1, cell_cols + 1]))


def _transform_needed_corners(transformer, x_corner_vec, y_corner_vec, cells):
    """Transform the corners of the given cells, leaving the other corners as NaN."""
    corners = np.zeros((y_corner_vec.size, x_corner_vec.size), dtype=bool)
    corners[:-1, :-1] |= cells
    corners[1:, :-1] |= cells
    corners[:-1, 1:] |= cells
    corners[1:, 1:] |= cells
    coarse_x = np.full(corners.shape, np.nan)
    coarse_y = np.full(corners.shape, np.nan)
    corner_rows, corner_cols = np.nonzero(corners)
    coarse_x[corners], coarse_y[corners] = transformer.transform(x_corner_vec[corner_cols], y_corner_vec[corner_rows])
    return coarse_x, coarse_y


def _is_accurate_at_middles(transformer, x_vec, y_vec, coarse_x, coarse_y, abs_tolerance, cells, x_axis, y_axis):
    """Check if the interpolation at the middle pixel of each cell is within the tolerance of the exact value."""
    cell_rows, cell_cols = cells
    _, x_weights, x_corners = x_axis
    _, y_weights, y_corners = y_axis
    middle_cols = (x_corners[cell_cols] + x_corners[cell_cols + 1]) // 2
    middle_rows = (y_corners[cell_rows] + y_corners[cell_rows + 1]) // 2
    exact_x, exact_y = transformer.transform(x_vec[middle_cols], y_vec[middle_rows])
    wx, wy = x_weights[middle_cols], y_weights[middle_rows]
    accurate = np.ones(cell_rows.shape, dtype=bool)
    for coarse, exact, tolerance in ((coarse_x, exact_x, abs_tolerance[0]), (coarse_y, exact_y, abs_tolerance[1])):
        interpolated = _interpolate_in_cells(coarse, cell_rows, cell_cols, wy, wx)
        with np.errstate(invalid="ignore"):
            accurate &= np.abs(interpolated - exact) <= tolerance
    return accurate


def _get_coarse_cells(size, step):
    """Get the coarse corner indices, and the cell index and interpolation weight of every pixel along one axis."""
    corners = np.r_[np.arange(0, size - 1, step), size - 1]
    pixels = np.arange(size)
    cells = np.minimum(pixels // step, corners.size - 2)
    weights = (pixels - corners[cells]) / (corners[cells + 1] - corners[cells])
    return cells, weights, corners


def _interpolate_separably(coarse, x_cells, x_weights, y_cells, y_weights):
    """Bilinearly interpolate a coarse grid to full resolution, first along x and then along y."""
    along_x = coarse[:, x_cells] * (1 - x_weights) + coarse[:, x_cells + 1] * x_weights
    return (along_x[y_cells, :] * (1 - y_weights)[:, np.newaxis] +
            along_x[y_cells + 1, :] * y_weights[:, np.newaxis])


def _dilate(mask):
    """Grow a 2D boolean mask by one element in all eight directions."""
    padded = np.pad(mask, 1)
    rows, cols = mask.shape
    res = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            res |= padded[dy:dy + rows, dx:dx + cols]
    return res


def block_bilinear_interpolator(data, indices_xy, fill_value=np.nan, block_info=None, **kwargs):
    """Bilinear interpolation implementation for resample_blocks."""
    mask, x_indices, y_indices = _get_mask_and_adjusted_indices(indices_xy, block_info)

    weight_l, l_start = np.modf(y_indices.clip(0, data.shape[-2] - 1))
    weight_p, p_start = np.modf(x_indices.clip(0, data.shape[-1] - 1))

    weight_l = weight_l.astype(data.dtype)
    weight_p = weight_p.astype(data.dtype)

    l_start = l_start.astype(int)
    p_start = p_start.astype(int)
    l_end = np.clip(l_start + 1, 1, data.shape[-2] - 1)
    p_end = np.clip(p_start + 1, 1, data.shape[-1] - 1)

    res = ((1 - weight_l) * (1 - weight_p) * data[..., l_start, p_start] +
           (1 - weight_l) * weight_p * data[..., l_start, p_end] +
           weight_l * (1 - weight_p) * data[..., l_end, p_start] +
           weight_l * weight_p * data[..., l_end, p_end])
    res = np.where(mask, fill_value, res)
    return res


def block_nn_interpolator(data, indices_xy, fill_value=np.nan, block_info=None, **kwargs):
    """Nearest neighbour 'interpolator' for resample_blocks."""
    mask, x_indices, y_indices = _get_mask_and_adjusted_indices(indices_xy, block_info)

    x_indices = np.clip(np.rint(x_indices), 0, data.shape[-1] - 1).astype(int)
    y_indices = np.clip(np.rint(y_indices), 0, data.shape[-2] - 1).astype(int)

    res = data[..., y_indices, x_indices]
    return np.where(mask, fill_value, res)


def _get_mask_and_adjusted_indices(indices_xy, block_info):
    """Get a mask for valid data and adjusted x and y indices."""
    x_indices, y_indices = indices_xy
    if block_info:
        y_slice, x_slice = block_info[0]["array-location"][-2:]
        x_indices = x_indices - x_slice.start
        y_indices = y_indices - y_slice.start
    mask = np.isnan(y_indices)
    x_indices = np.nan_to_num(x_indices, 0)
    y_indices = np.nan_to_num(y_indices, 0)
    return mask, x_indices, y_indices
