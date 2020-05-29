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

import logging

import dask.array as da
import dask
import numpy as np
import pyproj
import xarray as xr
from shapely.geometry import Polygon

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
                src_prj = pyproj.Proj(**self.source_geo_def.proj_dict)
                self.use_input_coords = True
            except AttributeError:
                self.src_x, self.src_y = self.source_geo_def.get_lonlats(
                    chunks=datachunks)
                src_prj = pyproj.Proj("+proj=longlat")
                self.use_input_coords = False
            try:
                self.dst_x, self.dst_y = self.target_geo_def.get_proj_coords(
                    chunks=CHUNK_SIZE)
                dst_prj = pyproj.Proj(**self.target_geo_def.proj_dict)
            except AttributeError:
                if self.use_input_coords is False:
                    raise NotImplementedError('Cannot resample lon/lat to lon/lat with gradient search.')
                self.dst_x, self.dst_y = self.target_geo_def.get_lonlats(
                    chunks=CHUNK_SIZE)
                dst_prj = pyproj.Proj("+proj=longlat")
            if self.use_input_coords:
                self.dst_x, self.dst_y = transform(
                    self.dst_x, self.dst_y,
                    src_prj=dst_prj, dst_prj=src_prj)
                self.prj = pyproj.Proj(**self.source_geo_def.proj_dict)
            else:
                self.src_x, self.src_y = transform(
                    self.src_x, self.src_y,
                    src_prj=src_prj, dst_prj=dst_prj)
                self.prj = pyproj.Proj(**self.target_geo_def.proj_dict)

    def _get_src_poly(self, src_y_start, src_y_end, src_x_start, src_x_end):
        """Get bounding polygon for source chunk."""
        geo_def = self.source_geo_def[src_y_start:src_y_end,
                                      src_x_start:src_x_end]
        try:
            src_poly = get_polygon(self.prj, geo_def)
        except AttributeError:
            # Can't create polygons for SwathDefinition
            src_poly = False

        return src_poly

    def _get_dst_poly(self, idx, dst_x_start, dst_x_end,
                      dst_y_start, dst_y_end):
        """Get target chunk polygon."""
        dst_poly = self.dst_polys.get(idx, None)
        if dst_poly is None:
            geo_def = self.target_geo_def[dst_y_start:dst_y_end,
                                          dst_x_start:dst_x_end]
            try:
                dst_poly = get_polygon(self.prj, geo_def)
            except AttributeError:
                # Can't create polygons for SwathDefinition
                dst_poly = False
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
                for k, dst_x_step in enumerate(dst_x_chunks):
                    dst_x_end = dst_x_start + dst_x_step
                    dst_y_start = 0
                    for l, dst_y_step in enumerate(dst_y_chunks):
                        dst_y_end = dst_y_start + dst_y_step
                        # Get destination chunk polygon
                        dst_poly = self._get_dst_poly((k, l),
                                                      dst_x_start, dst_x_end,
                                                      dst_y_start, dst_y_end)

                        covers = check_overlap(src_poly, dst_poly)

                        coverage_status.append(covers)
                        src_slices.append((src_y_start, src_y_end,
                                           src_x_start, src_x_end))
                        dst_slices.append((dst_y_start, dst_y_end,
                                           dst_x_start, dst_x_end))
                        dst_mosaic_locations.append((k, l))

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

        if fill_value is not None:
            res = da.where(np.isnan(res), fill_value, res)
        res = xr.DataArray(res, dims=data_dims, coords=coords)

        return res


def check_overlap(src_poly, dst_poly):
    """Check if the two polygons overlap."""
    if dst_poly is False or src_poly is False:
        covers = True
    elif dst_poly is not None and src_poly is not None:
        covers = src_poly.intersects(dst_poly)
    else:
        covers = False

    return covers


def _gradient_resample_data(src_data, src_x, src_y,
                            src_gradient_xl, src_gradient_xp,
                            src_gradient_yl, src_gradient_yp,
                            dst_x, dst_y,
                            method='bilinear'):
    """Resample using gradient search."""
    assert src_data.ndim == 3
    assert src_x.ndim == 2
    assert src_y.ndim == 2
    assert src_gradient_xl.ndim == 2
    assert src_gradient_xp.ndim == 2
    assert src_gradient_yl.ndim == 2
    assert src_gradient_yp.ndim == 2
    assert dst_x.ndim == 2
    assert dst_y.ndim == 2
    assert (src_data.shape[1:] == src_x.shape == src_y.shape ==
            src_gradient_xl.shape == src_gradient_xp.shape ==
            src_gradient_yl.shape == src_gradient_yp.shape)
    assert dst_x.shape == dst_y.shape

    image = one_step_gradient_search(src_data, src_x, src_y,
                                     src_gradient_xl, src_gradient_xp,
                                     src_gradient_yl, src_gradient_yp,
                                     dst_x, dst_y,
                                     method=method)

    return image


def get_border_lonlats(geo_def):
    """Get the border x- and y-coordinates."""
    if geo_def.proj_dict['proj'] == 'geos':
        lon_b, lat_b = get_geostationary_bounding_box(geo_def, 3600)
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

    res = da.concatenate(res, axis=2).squeeze()

    return res
