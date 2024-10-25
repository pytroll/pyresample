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

    def precompute(self, **kwargs):
        """Precompute resampling parameters."""
        if self.indices_xy is None:
            self.indices_xy = resample_blocks(gradient_resampler_indices_block,
                                              self.source_geo_def, [], self.target_geo_def,
                                              chunk_size=(2, CHUNK_SIZE, CHUNK_SIZE), dtype=float)

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


def gradient_resampler_indices_block(block_info=None, **kwargs):
    """Do the gradient search resampling using block_info for areas, returning the resulting indices."""
    source_area = block_info[0]["area"]
    target_area = block_info[None]["area"]
    return gradient_resampler_indices(source_area, target_area, block_info, **kwargs)


def gradient_resampler_indices(source_area, target_area, block_info=None, **kwargs):
    """Do the gradient search resampling, returning the resulting indices."""
    dst_coords, src_gradients, src_coords = _get_coordinates_in_same_projection(source_area, target_area)
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


def _get_coordinates_in_same_projection(source_area, target_area):
    try:
        src_x, src_y = source_area.get_proj_coords()
        work_crs = source_area.crs
    except AttributeError:
        # source is a swath definition, use target crs instead
        lons, lats = source_area.get_lonlats()
        src_x, src_y = da.compute(lons, lats)
        trans = pyproj.Transformer.from_crs(source_area.crs, target_area.crs, always_xy=True)
        src_x, src_y = trans.transform(src_x, src_y)
        work_crs = target_area.crs
    transformer = pyproj.Transformer.from_crs(target_area.crs, work_crs, always_xy=True)
    try:
        dst_x, dst_y = transformer.transform(*target_area.get_proj_coords())
    except AttributeError:
        # target is a swath definition
        lons, lats = target_area.get_lonlats()
        dst_x, dst_y = transformer.transform(*da.compute(lons, lats))
    src_gradient_xl, src_gradient_xp = np.gradient(src_x, axis=[0, 1])
    src_gradient_yl, src_gradient_yp = np.gradient(src_y, axis=[0, 1])
    return (dst_x, dst_y), (src_gradient_xl, src_gradient_xp, src_gradient_yl, src_gradient_yp), (src_x, src_y)


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
