#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013-2019

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

import logging

import dask.array as da
import numpy as np
import pyproj
import xarray as xr
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

from pyresample import CHUNK_SIZE
from pyresample.gradient._gradient_search import one_step_gradient_search
from pyresample.resampler import BaseResampler
from pyresample.geometry import get_geostationary_bounding_box

logger = logging.getLogger(__name__)


@da.as_gufunc(signature='(),()->(),()')
def transform(x_coords, y_coords, src_prj=None, dst_prj=None):
    """Calculate projection coordinates."""
    return pyproj.transform(src_prj, dst_prj, x_coords, y_coords)


class GradientSearchResampler(BaseResampler):
    """Resample using gradient search based bilinear interpolation."""

    def __init__(self, source_geo_def, target_geo_def):
        """Init GradientResampler."""
        super(GradientSearchResampler, self).__init__(source_geo_def, target_geo_def)
        import warnings
        warnings.warn("You are using the Gradient Search Resampler, which is still EXPERIMENTAL.")
        self.use_input_coords = None
        self.src_x = None
        self.src_y = None
        self.dst_x = None
        self.dst_y = None
        self.src_gradient_xl = None
        self.src_gradient_xp = None
        self.src_gradient_yl = None
        self.src_gradient_yp = None
        self._chunk_covers = None

    def _get_projection_coordinates(self, datachunks):
        """Get projection coordinates."""
        if self.use_input_coords is None:
            try:
                self.src_x, self.src_y = self.source_geo_def.get_proj_coords(
                    chunks=datachunks)
                self.src_prj = pyproj.Proj(**self.source_geo_def.proj_dict)
                self.use_input_coords = True
            except AttributeError:
                self.src_x, self.src_y = self.source_geo_def.get_lonlats(
                    chunks=datachunks)
                self.src_prj = pyproj.Proj("+proj=longlat")
                self.use_input_coords = False
            try:
                self.dst_x, self.dst_y = self.target_geo_def.get_proj_coords(
                    chunks=CHUNK_SIZE)
                self.dst_prj = pyproj.Proj(**self.target_geo_def.proj_dict)
            except AttributeError:
                if self.use_input_coords is False:
                    raise NotImplementedError('Cannot resample lon/lat to lon/lat with gradient search.')
                self.dst_x, self.dst_y = self.target_geo_def.get_lonlats(
                    chunks=CHUNK_SIZE)
                self.dst_prj = pyproj.Proj("+proj=longlat")
            if self.use_input_coords:
                self.dst_x, self.dst_y = transform(
                    self.dst_x, self.dst_y,
                    src_prj=self.dst_prj, dst_prj=self.src_prj)
            else:
                self.src_x, self.src_y = transform(
                    self.src_x, self.src_y,
                    src_prj=self.src_prj, dst_prj=self.dst_prj)

    def _find_covering_chunks(self):
        """Find chunks that cover the target area."""
        dst_border_lon, dst_border_lat = get_border_lonlats(self.target_geo_def)
        if self.use_input_coords:
            prj = pyproj.Proj(self.source_geo_def.crs)
        else:
            prj = pyproj.Proj(self.target_geo_def.crs)
        dst_border_x, dst_border_y = prj(dst_border_lon, dst_border_lat)

        dst_polygon = get_polygon(dst_border_x, dst_border_y)

        shp = self.src_x.shape
        num_chunks = shp[2]
        x_start, y_start = 0, 0
        src_polys = []
        for i in range(num_chunks):
            x_end = x_start + shp[1]
            y_end = y_start + shp[0]

            chunk_geo_def = self.source_geo_def[y_start:y_end, x_start:x_end]
            b_lon, b_lat = get_border_lonlats(chunk_geo_def)
            b_x, b_y = prj(b_lon, b_lat)
            src_polys.append(get_polygon(b_x, b_y))

            x_start += shp[1]
            if x_end == shp[1]:
                x_start = 0
                y_start += shp[0]

        covers = []
        for poly in src_polys:
            try:
                # Destination area has all corners/sides in space
                if dst_polygon.area == 0.0:
                    cov = True
                else:
                    cov = dst_polygon.intersects(poly)
            # This happens if the target area "goes over the edge" of the
            # GEO disk and the border isn't a closed curve
            except TopologicalError:
                cov = True
            covers.append(cov)

        self._chunk_covers = covers

    def _remove_extra_chunks(self, data):
        """Remove chunks that don't cover the target area."""
        if self._chunk_covers is None:
            self._find_covering_chunks()
            arrays = (data, self.src_x, self.src_y,
                      self.src_gradient_xl, self.src_gradient_xp,
                      self.src_gradient_yl, self.src_gradient_yp)
        else:
            arrays = (data, )

        res = []
        for arr in arrays:
            res.append(np.take(arr, self._chunk_covers, axis=-1))

        try:
            (data, self.src_x, self.src_y,
             self.src_gradient_xl, self.src_gradient_xp,
             self.src_gradient_yl, self.src_gradient_yp) = res
        except ValueError:
            data, = res

        return data

    def _prepare_arrays(self, data):
        if data.ndim not in [2, 3]:
            raise NotImplementedError('Gradient search resampling only '
                                      'supports 2D or 3D arrays.')
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        if self._chunk_covers is None:
            arrays = reshape_arrays_in_stacked_chunks(
                (self.src_x, self.src_y,
                 self.src_gradient_xl, self.src_gradient_xp,
                 self.src_gradient_yl, self.src_gradient_yp),
                self.src_x.chunks)
            (self.src_x, self.src_y, self.src_gradient_xl,
             self.src_gradient_xp, self.src_gradient_yl,
             self.src_gradient_yp) = arrays

        data = reshape_to_stacked_3d(data)
        data = self._remove_extra_chunks(data)

        return data

    def _get_gradients(self):
        """Get gradients in X and Y directions."""
        self.src_gradient_xl, self.src_gradient_xp = np.gradient(
            self.src_x, axis=[0, 1])
        self.src_gradient_yl, self.src_gradient_yp = np.gradient(
            self.src_y, axis=[0, 1])

    def compute(self, data, fill_value=None, **kwargs):
        """Resample the given data using gradient search algorithm."""
        if 'bands' in data.dims:
            datachunks = data.sel(bands=data.coords['bands'][0]).chunks
        else:
            datachunks = data.chunks
        data_dims = data.dims
        data_coords = data.coords

        self._get_projection_coordinates(datachunks)
        self._get_gradients()
        data = self._prepare_arrays(data.data)

        res = parallel_gradient_search(data,
                                       self.src_x, self.src_y,
                                       self.dst_x, self.dst_y,
                                       self.src_gradient_xl,
                                       self.src_gradient_xp,
                                       self.src_gradient_yl,
                                       self.src_gradient_yp,
                                       **kwargs)
        # TODO: this will crash wen the target geo definition is a swath def.
        x_coord, y_coord = self.target_geo_def.get_proj_vectors()
        coords = []
        for key in data_dims:
            if key == 'x':
                coords.append(x_coord)
            elif key == 'y':
                coords.append(y_coord)
            else:
                coords.append(data_coords[key])
        res = xr.DataArray(res, dims=data_dims, coords=coords)
        return res


def _gradient_resample_data(src_data, src_x, src_y,
                            src_gradient_xl, src_gradient_xp,
                            src_gradient_yl, src_gradient_yp,
                            dst_x, dst_y, method='bilinear'):
    image = one_step_gradient_search(
        src_data[0][0][:, :, :, 0],
        src_x[0][0][:, :, 0],
        src_y[0][0][:, :, 0],
        src_gradient_xl[0][0][:, :, 0],
        src_gradient_xp[0][0][:, :, 0],
        src_gradient_yl[0][0][:, :, 0],
        src_gradient_yp[0][0][:, :, 0],
        dst_x,
        dst_y,
        method=method)
    return image[:, :, :, np.newaxis]


def vsplit(arr, n):
    """Split the array vertically."""
    res = arr.reshape((n, -1) + arr.shape[1:])
    return [np.take(res, x, axis=0) for x in range(n)]


def hsplit(arr, n):
    """Split the array horizontally."""
    res = arr.reshape((arr.shape[0], n, -1) + arr.shape[2:])
    return [np.take(res, x, axis=1) for x in range(n)]


def split(arr, n, axis):
    """Split an array in n pieces along axis."""
    shape = arr.shape
    ax_shape = shape[axis]
    if axis < 0:
        rest_idx = len(shape) + axis + 1
    else:
        rest_idx = axis + 1
    new_shape = shape[:axis] + (n, int(ax_shape / n)) + shape[rest_idx:]
    res = arr.reshape(new_shape)
    return [np.take(res, x, axis=rest_idx - 1) for x in range(n)]


def reshape_arrays_in_stacked_chunks(arrays, chunks):
    """Reshape the arrays such that all the chunks are stacked along the last dimension.

    In effect, this will make the arrays have only one chunk over the first dimensions.
    """
    h_fac = len(chunks[1])
    v_fac = len(chunks[0])
    res = []
    for array in arrays:
        cols = hsplit(array, h_fac)
        layers = []
        for col in cols:
            layers.extend(vsplit(col, v_fac))
        res.append(np.stack(layers, axis=2))

    return res


def reshape_to_stacked_3d(array):
    """Reshape a 3d array so that all chunks on the x and y dimensions are stacked along the last dimension.

    This relies on y and x being the two last dimensions of the input array.
    """
    chunks = array.chunks

    x_fac = len(chunks[-1])
    y_fac = len(chunks[-2])
    cols = split(array, x_fac, -1)
    layers = []
    for col in cols:
        layers.extend(split(col, y_fac, -2))
    return np.stack(layers, axis=-1)


def get_border_lonlats(geo_def, x_stride=1, y_stride=1):
    """Get the border x- and y-coordinates."""
    if geo_def.proj_dict['proj'] == 'geos':
        lon_b, lat_b = get_geostationary_bounding_box(geo_def)
    else:
        lons, lats = geo_def.get_boundary_lonlats()
        lon_b = np.concatenate((lons.side1, lons.side2, lons.side3, lons.side4))
        lat_b = np.concatenate((lats.side1, lats.side2, lats.side3, lats.side4))

    return lon_b, lat_b


def get_polygon(x_borders, y_borders):
    """Get border polygon from *x_borders* and *y_borders*."""
    boundary = [(x_borders[i], y_borders[i]) for i in range(len(x_borders))
                if np.isfinite(x_borders[i]) and np.isfinite(y_borders[i])]

    return Polygon(boundary)


def parallel_gradient_search(data, src_x, src_y, dst_x, dst_y,
                             src_gradient_xl, src_gradient_xp,
                             src_gradient_yl, src_gradient_yp, **kwargs):
    """Run gradient search in parallel in input area coordinates."""
    if data.ndim != 4:
        raise NotImplementedError(
            'Gradient search resampling only supports 4D arrays.')
    res = da.blockwise(_gradient_resample_data, 'bmnz',
                       data.astype(np.float64), 'bijz',
                       src_x, 'ijz', src_y, 'ijz',
                       src_gradient_xl, 'ijz', src_gradient_xp, 'ijz',
                       src_gradient_yl, 'ijz', src_gradient_yp, 'ijz',
                       dst_x, 'mn', dst_y, 'mn',
                       dtype=np.float64,
                       method=kwargs.get('method', 'bilinear'))

    return da.nanmax(res, axis=-1).squeeze()
