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
    if isinstance(source_geo_def, AreaDefinition) and isinstance(target_geo_def, AreaDefinition):
        return ResampleBlocksGradientSearchResampler(source_geo_def, target_geo_def)
    elif isinstance(source_geo_def, SwathDefinition) and isinstance(target_geo_def, AreaDefinition):
        return StackingGradientSearchResampler(source_geo_def, target_geo_def)
    raise NotImplementedError


@da.as_gufunc(signature='(),()->(),()')
def transform(x_coords, y_coords, src_prj=None, dst_prj=None):
    """Calculate projection coordinates."""
    transformer = pyproj.Transformer.from_crs(src_prj, dst_prj)
    return transformer.transform(x_coords, y_coords)


class StackingGradientSearchResampler(BaseResampler):
    """Resample using gradient search based bilinear interpolation, using stacking for dask processing."""

    def __init__(self, source_geo_def, target_geo_def):
        """Init GradientResampler."""
        super().__init__(source_geo_def, target_geo_def)
        import warnings
        warnings.warn("You are using the Gradient Search Resampler, which is still EXPERIMENTAL.", stacklevel=2)
        self.use_input_coords = None
        self._src_dst_filtered = False
        self.prj = None
        self.src_x = None
        self.src_y = None
        self.src_slices = None
        self.dst_x = None
        self.dst_y = None
        self.dst_slices = None
        self.src_gradient_xl = None
        self.src_gradient_xp = None
        self.src_gradient_yl = None
        self.src_gradient_yp = None
        self.dst_polys = {}
        self.dst_mosaic_locations = None
        self.coverage_status = None

    def _get_projection_coordinates(self, datachunks):
        """Get projection coordinates."""
        if self.use_input_coords is None:
            try:
                self.src_x, self.src_y = self.source_geo_def.get_proj_coords(
                    chunks=datachunks)
                src_crs = self.source_geo_def.crs
                self.use_input_coords = True
            except AttributeError:
                self.src_x, self.src_y = self.source_geo_def.get_lonlats(
                    chunks=datachunks)
                src_crs = pyproj.CRS.from_string("+proj=longlat")
                self.use_input_coords = False
            try:
                self.dst_x, self.dst_y = self.target_geo_def.get_proj_coords(
                    chunks=CHUNK_SIZE)
                dst_crs = self.target_geo_def.crs
            except AttributeError as err:
                if self.use_input_coords is False:
                    raise NotImplementedError('Cannot resample lon/lat to lon/lat with gradient search.') from err
                self.dst_x, self.dst_y = self.target_geo_def.get_lonlats(
                    chunks=CHUNK_SIZE)
                dst_crs = pyproj.CRS.from_string("+proj=longlat")
            if self.use_input_coords:
                self.dst_x, self.dst_y = transform(
                    self.dst_x, self.dst_y,
                    src_prj=dst_crs, dst_prj=src_crs)
                self.prj = pyproj.Proj(self.source_geo_def.crs)
            else:
                self.src_x, self.src_y = transform(
                    self.src_x, self.src_y,
                    src_prj=src_crs, dst_prj=dst_crs)
                self.prj = pyproj.Proj(self.target_geo_def.crs)

    def _get_prj_poly(self, geo_def):
        # - None if out of Earth Disk
        # - False is SwathDefinition
        if isinstance(geo_def, SwathDefinition):
            return False
        try:
            poly = get_polygon(self.prj, geo_def)
        except (NotImplementedError, ValueError):  # out-of-earth disk area or any valid projected boundary coordinates
            poly = None
        return poly

    def _get_src_poly(self, src_y_start, src_y_end, src_x_start, src_x_end):
        """Get bounding polygon for source chunk."""
        geo_def = self.source_geo_def[src_y_start:src_y_end,
                                      src_x_start:src_x_end]
        return self._get_prj_poly(geo_def)

    def _get_dst_poly(self, idx,
                      dst_x_start, dst_x_end,
                      dst_y_start, dst_y_end):
        """Get target chunk polygon."""
        dst_poly = self.dst_polys.get(idx, None)
        if dst_poly is None:
            geo_def = self.target_geo_def[dst_y_start:dst_y_end,
                                          dst_x_start:dst_x_end]
            dst_poly = self._get_prj_poly(geo_def)
            self.dst_polys[idx] = dst_poly
        return dst_poly

    def get_chunk_mappings(self):
        """Map source and target chunks together if they overlap."""
        src_y_chunks, src_x_chunks = self.src_x.chunks
        dst_y_chunks, dst_x_chunks = self.dst_x.chunks

        coverage_status = []
        src_slices, dst_slices = [], []
        dst_mosaic_locations = []

        src_x_start = 0
        for src_x_step in src_x_chunks:
            src_x_end = src_x_start + src_x_step
            src_y_start = 0
            for src_y_step in src_y_chunks:
                src_y_end = src_y_start + src_y_step
                # Get source chunk polygon
                src_poly = self._get_src_poly(src_y_start, src_y_end,
                                              src_x_start, src_x_end)

                dst_x_start = 0
                for x_step_number, dst_x_step in enumerate(dst_x_chunks):
                    dst_x_end = dst_x_start + dst_x_step
                    dst_y_start = 0
                    for y_step_number, dst_y_step in enumerate(dst_y_chunks):
                        dst_y_end = dst_y_start + dst_y_step
                        # Get destination chunk polygon
                        dst_poly = self._get_dst_poly((x_step_number, y_step_number),
                                                      dst_x_start, dst_x_end,
                                                      dst_y_start, dst_y_end)

                        covers = check_overlap(src_poly, dst_poly)

                        coverage_status.append(covers)
                        src_slices.append((src_y_start, src_y_end,
                                           src_x_start, src_x_end))
                        dst_slices.append((dst_y_start, dst_y_end,
                                           dst_x_start, dst_x_end))
                        dst_mosaic_locations.append((x_step_number, y_step_number))

                        dst_y_start = dst_y_end
                    dst_x_start = dst_x_end
                src_y_start = src_y_end
            src_x_start = src_x_end

        self.src_slices = src_slices
        self.dst_slices = dst_slices
        self.dst_mosaic_locations = dst_mosaic_locations
        self.coverage_status = coverage_status

    def _filter_data(self, data, is_src=True, add_dim=False):
        """Filter unused chunks from the given array."""
        if add_dim:
            if data.ndim not in [2, 3]:
                raise NotImplementedError('Gradient search resampling only '
                                          'supports 2D or 3D arrays.')
            if data.ndim == 2:
                data = data[np.newaxis, :, :]

        data_out = []
        for i, covers in enumerate(self.coverage_status):
            if covers:
                if is_src:
                    y_start, y_end, x_start, x_end = self.src_slices[i]
                else:
                    y_start, y_end, x_start, x_end = self.dst_slices[i]
                try:
                    val = data[:, y_start:y_end, x_start:x_end]
                except IndexError:
                    val = data[y_start:y_end, x_start:x_end]
            else:
                val = None
            data_out.append(val)

        return data_out

    def _get_gradients(self):
        """Get gradients in X and Y directions."""
        self.src_gradient_xl, self.src_gradient_xp = np.gradient(
            self.src_x, axis=[0, 1])
        self.src_gradient_yl, self.src_gradient_yp = np.gradient(
            self.src_y, axis=[0, 1])

    def _filter_src_dst(self):
        """Filter source and target chunks."""
        self.src_x = self._filter_data(self.src_x)
        self.src_y = self._filter_data(self.src_y)
        self.src_gradient_yl = self._filter_data(self.src_gradient_yl)
        self.src_gradient_yp = self._filter_data(self.src_gradient_yp)
        self.src_gradient_xl = self._filter_data(self.src_gradient_xl)
        self.src_gradient_xp = self._filter_data(self.src_gradient_xp)
        self.dst_x = self._filter_data(self.dst_x, is_src=False)
        self.dst_y = self._filter_data(self.dst_y, is_src=False)
        self._src_dst_filtered = True

    def compute(self, data, fill_value=None, **kwargs):
        """Resample the given data using gradient search algorithm."""
        if 'bands' in data.dims:
            datachunks = data.sel(bands=data.coords['bands'][0]).chunks
        else:
            datachunks = data.chunks
        data_dims = data.dims
        data_coords = data.coords

        self._get_projection_coordinates(datachunks)

        if self.src_gradient_xl is None:
            self._get_gradients()
        if self.coverage_status is None:
            self.get_chunk_mappings()
        if not self._src_dst_filtered:
            self._filter_src_dst()

        data = self._filter_data(data.data, add_dim=True)

        res = parallel_gradient_search(data,
                                       self.src_x, self.src_y,
                                       self.dst_x, self.dst_y,
                                       self.src_gradient_xl,
                                       self.src_gradient_xp,
                                       self.src_gradient_yl,
                                       self.src_gradient_yp,
                                       self.dst_mosaic_locations,
                                       self.dst_slices,
                                       **kwargs)

        coords = _fill_in_coords(self.target_geo_def, data_coords, data_dims)

        if fill_value is not None:
            res = da.where(np.isnan(res), fill_value, res)
        if res.ndim > len(data_dims):
            res = res.squeeze()

        res = xr.DataArray(res, dims=data_dims, coords=coords)
        return res


def check_overlap(src_poly, dst_poly):
    """Check if the two polygons overlap."""
    # swath definition case
    if dst_poly is False or src_poly is False:
        covers = True
    # area / area case
    elif dst_poly is not None and src_poly is not None:
        covers = src_poly.intersects(dst_poly)
    # out of earth disk case
    else:
        covers = False
    return covers


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


def get_border_lonlats(geo_def: AreaDefinition):
    """Get the border x- and y-coordinates."""
    if geo_def.is_geostationary:
        lon_b, lat_b = get_geostationary_bounding_box_in_lonlats(geo_def, 3600)
    else:
        lons, lats = geo_def.get_boundary_lonlats()
        lon_b = np.concatenate((lons.side1, lons.side2, lons.side3, lons.side4))
        lat_b = np.concatenate((lats.side1, lats.side2, lats.side3, lats.side4))

    return lon_b, lat_b


def get_polygon(prj, geo_def):
    """Get border polygon from area definition in projection *prj*."""
    lon_b, lat_b = get_border_lonlats(geo_def)
    x_borders, y_borders = prj(lon_b, lat_b)
    boundary = [(x_borders[i], y_borders[i]) for i in range(len(x_borders))
                if np.isfinite(x_borders[i]) and np.isfinite(y_borders[i])]
    poly = Polygon(boundary)
    if np.isfinite(poly.area) and poly.area > 0.0:
        return poly
    return None


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
                arr.astype(np.float64),
                src_x[i], src_y[i],
                src_gradient_xl[i], src_gradient_xp[i],
                src_gradient_yl[i], src_gradient_yp[i],
                dst_x[i], dst_y[i],
                method=method)
            res = da.from_delayed(res, (num_bands, ) + dst_x[i].shape,
                                  dtype=np.float64)
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
    x_coord, y_coord = target_geo_def.get_proj_vectors()
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
        if isinstance(target_geo_def, SwathDefinition):
            raise NotImplementedError("Cannot resample to a SwathDefinition.")
        super().__init__(source_geo_def, target_geo_def)
        logger.debug("/!\\ Instantiating an experimental GradientSearch resampler /!\\")
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
        transformer = pyproj.Transformer.from_crs(target_area.crs, source_area.crs, always_xy=True)
    except AttributeError as err:
        raise NotImplementedError("Cannot resample from Swath for now.") from err

    try:
        dst_x, dst_y = transformer.transform(*target_area.get_proj_coords())
    except AttributeError as err:
        raise NotImplementedError("Cannot resample to Swath for now.") from err
    src_gradient_xl, src_gradient_xp = np.gradient(src_x, axis=[0, 1])
    src_gradient_yl, src_gradient_yp = np.gradient(src_y, axis=[0, 1])
    return (dst_x, dst_y), (src_gradient_xl, src_gradient_xp, src_gradient_yl, src_gradient_yp), (src_x, src_y)


def block_bilinear_interpolator(data, indices_xy, fill_value=np.nan, block_info=None, **kwargs):
    """Bilinear interpolation implementation for resample_blocks."""
    mask, x_indices, y_indices = _get_mask_and_adjusted_indices(indices_xy, block_info)

    weight_l, l_start = np.modf(y_indices.clip(0, data.shape[-2] - 1))
    weight_p, p_start = np.modf(x_indices.clip(0, data.shape[-1] - 1))

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
