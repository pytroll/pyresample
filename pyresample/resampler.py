#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Pyresample developers
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
"""Base resampler class made for subclassing."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from numbers import Number
from typing import Union

import numpy as np

try:
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
except ImportError:
    da = None

try:
    import xarray as xr
except ImportError:
    xr = None

from pyproj.transformer import Transformer

from pyresample.geometry import (
    AreaDefinition,
    CoordinateDefinition,
    IncompatibleAreas,
    InvalidArea,
    SwathDefinition,
    get_geostationary_bounding_box_in_proj_coords,
)

from .future.resamplers.resampler import hash_dict

logger = logging.getLogger(__name__)


class BaseResampler:
    """Base abstract resampler class."""

    def __init__(self,
                 source_geo_def: Union[SwathDefinition, AreaDefinition],
                 target_geo_def: Union[CoordinateDefinition, AreaDefinition],
                 ):
        """Initialize resampler with geolocation information.

        Args:
            source_geo_def:
                Geolocation definition for the data to be resampled
            target_geo_def:
                Geolocation definition for the area to resample data to.

        """
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def

    def get_hash(self, source_geo_def=None, target_geo_def=None, **kwargs):
        """Get hash for the current resample with the given *kwargs*."""
        if source_geo_def is None:
            source_geo_def = self.source_geo_def
        if target_geo_def is None:
            target_geo_def = self.target_geo_def
        the_hash = source_geo_def.update_hash()
        target_geo_def.update_hash(the_hash)
        hash_dict(kwargs, the_hash)
        return the_hash.hexdigest()

    def precompute(self, **kwargs):
        """Do the precomputation.

        This is an optional step if the subclass wants to implement more
        complex features like caching or can share some calculations
        between multiple datasets to be processed.

        """
        return None

    def compute(self, data, **kwargs):
        """Do the actual resampling.

        This must be implemented by subclasses.

        """
        raise NotImplementedError

    def resample(self, data, cache_dir=None, mask_area=None, **kwargs):
        """Resample `data` by calling `precompute` and `compute` methods.

        Only certain resampling classes may use `cache_dir` and the `mask`
        provided when `mask_area` is True. The return value of calling the
        `precompute` method is passed as the `cache_id` keyword argument
        of the `compute` method, but may not be used directly for caching. It
        is up to the individual resampler subclasses to determine how this
        is used.

        Args:
            data (xarray.DataArray): Data to be resampled
            cache_dir (str): directory to cache precomputed results
                             (default False, optional)
            mask_area (bool): Mask geolocation data where data values are
                              invalid. This should be used when data values
                              may affect what neighbors are considered valid.

        Returns (xarray.DataArray): Data resampled to the target area

        """
        # default is to mask areas for SwathDefinitions
        if mask_area is None and isinstance(
                self.source_geo_def, SwathDefinition):
            mask_area = True

        if mask_area:
            if isinstance(self.source_geo_def, SwathDefinition):
                geo_dims = self.source_geo_def.lons.dims
            else:
                geo_dims = ('y', 'x')
            flat_dims = [dim for dim in data.dims if dim not in geo_dims]
            if np.issubdtype(data.dtype, np.integer):
                kwargs['mask'] = data == data.attrs.get('_FillValue', np.iinfo(data.dtype.type).max)
            else:
                kwargs['mask'] = data.isnull()
            kwargs['mask'] = kwargs['mask'].all(dim=flat_dims)

        cache_id = self.precompute(cache_dir=cache_dir, **kwargs)
        return self.compute(data, cache_id=cache_id, **kwargs)

    def _create_cache_filename(self, cache_dir=None, prefix='',
                               fmt='.zarr', **kwargs):
        """Create filename for the cached resampling parameters."""
        cache_dir = cache_dir or '.'
        hash_str = self.get_hash(**kwargs)

        return os.path.join(cache_dir, prefix + hash_str + fmt)


def resample_blocks(src_area, dst_area, funk, *args, dst_arrays=(), chunks=None, dtype=None, name=None, **kwargs):
    """Resample blockwise.

    Args:
        src_area: a source geo definition
        dst_area:
        funk: a function to use. If func has a block_info keyword argument, the chunk info is passed, as in map_blocks
        args: data to use
        dst_arrays: arrays to use that are already in dst_area space. If the array has more than 2 dimensions,
            the last two are expected to be y, x.
        chunks: Has to be provided
        dtype: Has to be provided
        kwargs:

    """
    if args:
        data = args[0]
    else:
        data = None
    if dst_area == src_area:
        return data
    from dask.base import tokenize
    from dask.utils import apply, funcname, has_keyword

    name = f"{name or funcname(funk)}-{tokenize(funk, src_area, dst_area, dst_arrays, dtype, chunks, *args, **kwargs)}"
    dask_graph = dict()
    dependencies = []

    for target_block_info, target_geo_def in _enumerate_dst_area_chunks(dst_area, chunks):
        position = target_block_info["chunk-location"]
        output_shape = target_block_info["shape"]
        try:
            smaller_data, source_geo_def, source_block_info = crop_data_around_area(src_area, data, target_geo_def)
        except IncompatibleAreas:  # no relevant data matching
            if np.issubdtype(dtype, np.integer):
                fill_value = np.iinfo(dtype).min
            else:
                fill_value = np.nan
            task = [np.full, target_block_info["chunk-shape"], fill_value]
        else:
            args = [source_geo_def, target_geo_def]
            if data is not None:
                args.append((smaller_data.name, 0, 0))
                dependencies.append(smaller_data)

            for dst_array in dst_arrays:
                dst_position = [0] * (dst_array.ndim - 2) + list(position)
                args.append((dst_array.name, *dst_position))
                dependencies.append(dst_array)
            funk_kwargs = kwargs.copy()
            if has_keyword(funk, "block_info"):
                funk_kwargs["block_info"] = {0: source_block_info,
                                             None: target_block_info}
            task = [apply, funk, args, funk_kwargs]

        dask_graph[(name, *position)] = tuple(task)

    dask_graph = HighLevelGraph.from_collections(name, dask_graph, dependencies=dependencies)
    return da.Array(dask_graph, name, chunks=chunks, dtype=dtype, shape=output_shape)


@lru_cache(None)
def crop_source_area(source_geo_def, target_geo_def):
    """Crop a source area around a provided a target area."""
    slicer = Slicer(source_geo_def, target_geo_def)
    x_slice, y_slice = slicer.get_slices()
    small_source_geo_def = source_geo_def[y_slice, x_slice]
    if isinstance(small_source_geo_def, SwathDefinition):
        small_source_geo_def.lons.data = small_source_geo_def.lons.data.rechunk((-1, -1))
        small_source_geo_def.lats.data = small_source_geo_def.lats.data.rechunk((-1, -1))
    return small_source_geo_def, x_slice, y_slice


def crop_data_around_area(source_geo_def, data, target_geo_def):
    """Crop the data around the provided area."""
    small_source_geo_def, x_slice, y_slice = crop_source_area(source_geo_def, target_geo_def)
    if data is not None:
        smaller_data = data[..., y_slice, x_slice].rechunk(data.chunks[:-2] + (-1, -1))
    else:
        smaller_data = None
    block_info = {"shape": source_geo_def.shape, "array-location": (y_slice, x_slice)}
    return smaller_data, small_source_geo_def, block_info


def _enumerate_dst_area_chunks(dst_area, chunks):
    """Enumerate the chunks in function of the dst_area."""
    rest_shape = []
    if not isinstance(chunks, Number) and len(chunks) > len(dst_area.shape):
        rest_chunks = chunks[:-len(dst_area.shape)]
        for elt in rest_chunks:
            try:
                rest_shape.append(sum(elt))
            except TypeError:
                rest_shape.append(elt)
    output_shape = tuple(rest_shape) + dst_area.shape
    dst_chunks = da.core.normalize_chunks(chunks, output_shape)

    for position, slices in _enumerate_chunk_slices(dst_chunks):
        chunk_shape = tuple(chunk[pos] for pos, chunk in zip(position, dst_chunks))
        target_geo_def = dst_area[slices[-2:]]
        block_info = {"shape": output_shape,
                      "num-chunks": [len(chunks) for chunks in dst_chunks],
                      "chunk-location": position,
                      "array-location": slices,
                      "chunk-shape": chunk_shape,
                      }
        yield block_info, target_geo_def


def _enumerate_chunk_slices(chunks):
    """Enumerate chunks with slices."""
    for position in np.ndindex(tuple(map(len, (chunks)))):
        slices = []
        for pos, chunk in zip(position, chunks):
            chunk_size = chunk[pos]
            offset = sum(chunk[:pos])
            slices.append(slice(offset, offset + chunk_size))

        yield (position, slices)


def bil2(src_area, dst_area, data, indices_xy, block_info=None):
    """Bilinear interpolation implementation for resample_blocks."""
    del src_area, dst_area
    x_indices, y_indices = indices_xy
    if block_info:
        y_slice, x_slice = block_info[0]["array-location"][-2:]
        x_indices -= x_slice.start
        y_indices -= y_slice.start
    mask = np.isnan(y_indices)
    x_indices = np.nan_to_num(x_indices, 0)
    y_indices = np.nan_to_num(y_indices, 0)

    w_l, l_a = np.modf(y_indices.clip(0, data.shape[-2] - 1))
    w_p, p_a = np.modf(x_indices.clip(0, data.shape[-1] - 1))

    l_a = l_a.astype(int)
    p_a = p_a.astype(int)
    l_b = np.clip(l_a + 1, 1, data.shape[-2] - 1)
    p_b = np.clip(p_a + 1, 1, data.shape[-1] - 1)

    res = ((1 - w_l) * (1 - w_p) * data[..., l_a, p_a] +
           (1 - w_l) * w_p * data[..., l_a, p_b] +
           w_l * (1 - w_p) * data[..., l_b, p_a] +
           w_l * w_p * data[..., l_b, p_b])
    res = np.where(mask, np.nan, res)
    return res


class Slicer(ABC):
    """Abstract Slicer and Slicer factory class, returning a AreaSlicer or a SwathSlicer based on the first area type.

    Provided an Area-to-crop and an Area-to-contain, a Slicer provides methods
    to find slices that enclose Area-to-contain inside Area-to-crop.
    """

    def __new__(cls, area_to_crop, area_to_contain):
        """Create a Slicer for cropping *area_to_crop* based on *area_to_contain*."""
        if cls is Slicer:
            if isinstance(area_to_crop, SwathDefinition):
                return SwathSlicer(area_to_crop, area_to_contain)
            elif isinstance(area_to_crop, AreaDefinition):
                return AreaSlicer(area_to_crop, area_to_contain)
            else:
                raise NotImplementedError("Don't know how to slice a " + str(type(area_to_crop)))
        else:
            return super().__new__(cls)

    def __init__(self, area_to_crop, area_to_contain):
        """Set up the Slicer."""
        self.area_to_crop = area_to_crop
        self.area_to_contain = area_to_contain
        self._transformer = Transformer.from_crs(self.area_to_contain.crs, self.area_to_crop.crs)

    def get_slices(self):
        """Get the slices to crop *area_to_crop* enclosing *area_to_contain*."""
        poly = self.get_polygon()
        return self.get_slices_from_polygon(poly)

    @abstractmethod
    def get_polygon(self):
        """Get the shapely Polygon corresponding to *area_to_contain*."""
        raise NotImplementedError

    @abstractmethod
    def get_slices_from_polygon(self, poly):
        """Get the slices based on the polygon."""
        raise NotImplementedError


class SwathSlicer(Slicer):
    """A Slicer for cropping SwathDefinitions."""

    def get_polygon(self):
        """Get the shapely Polygon corresponding to *area_to_contain* in lon/lat coordinates."""
        from shapely.geometry import Polygon
        x, y = self.area_to_contain.get_bbox_coords(10)
        poly = Polygon(zip(*self._transformer.transform(x, y)))
        return poly

    def get_slices_from_polygon(self, poly):
        """Get the slices based on the polygon."""
        intersecting_chunk_slices = []
        for smaller_poly, slices in _get_chunk_polygons_for_area_to_crop(self.area_to_crop):
            if smaller_poly.intersects(poly):
                intersecting_chunk_slices.append(slices)
        if not intersecting_chunk_slices:
            raise IncompatibleAreas
        return self._assemble_slices(intersecting_chunk_slices)

    @staticmethod
    def _assemble_slices(chunk_slices):
        """Assemble slices to one slice per dimension."""
        lines, cols = zip(*chunk_slices)
        line_slice = slice(min(slc.start for slc in lines), max(slc.stop for slc in lines))
        col_slice = slice(min(slc.start for slc in cols), max(slc.stop for slc in cols))
        slices = col_slice, line_slice
        return slices


@lru_cache(maxsize=10)
def _get_chunk_polygons_for_area_to_crop(area_to_crop):
    """Get the polygons for each chunk of the area_to_crop."""
    res = []
    from shapely.geometry import Polygon
    chunks = np.array(area_to_crop.lons.data.chunksize) // 2
    src_chunks = da.core.normalize_chunks(chunks, area_to_crop.shape)
    for _position, (line_slice, col_slice) in _enumerate_chunk_slices(src_chunks):
        line_slice = expand_slice(line_slice)
        col_slice = expand_slice(col_slice)
        smaller_swath = area_to_crop[line_slice, col_slice]
        lons, lats = smaller_swath.get_edge_lonlats(10)
        lons = np.hstack(lons)
        lats = np.hstack(lats)
        smaller_poly = Polygon(zip(lons, lats))
        res.append((smaller_poly, (line_slice, col_slice)))
    return res


def expand_slice(small_slice):
    """Expand slice by one."""
    return slice(max(small_slice.start - 1, 0), small_slice.stop + 1, small_slice.step)


class AreaSlicer(Slicer):
    """A Slicer for cropping AreaDefinitions."""

    def get_polygon(self):
        """Get the shapely Polygon corresponding to *area_to_contain* in projection coordinates."""
        from shapely.geometry import Polygon
        x, y = self.area_to_contain.get_bbox_coords(frequency=10)
        # before_poly = Polygon(zip(x, y)).buffer(np.max(target_area.resolution))
        # x, y = zip(*before_poly.exterior.coords)
        if self.area_to_crop.is_geostationary:
            x_geos, y_geos = get_geostationary_bounding_box_in_proj_coords(self.area_to_crop, 360)
            x_geos, y_geos = self._transformer.transform(x_geos, y_geos, direction='INVERSE')
            geos_poly = Polygon(zip(x_geos, y_geos))
            poly = Polygon(zip(x, y))
            poly = poly.intersection(geos_poly)
            if poly.is_empty:
                raise IncompatibleAreas('No slice on area.')
            x, y = zip(*poly.exterior.coords)
        poly = Polygon(zip(*self._transformer.transform(x, y)))
        return poly

    def get_slices_from_polygon(self, poly):
        """Get the slices based on the polygon."""
        # We take a little margin around the polygon to ensure all needed pixels will be included.
        try:
            bounds = poly.buffer(np.max(self.area_to_contain.resolution)).bounds
        except ValueError as err:
            raise InvalidArea(str(err))
        bounds = self._sanitize_polygon_bounds(bounds)
        slice_x, slice_y = self._create_slices_from_bounds(bounds)
        return slice_x, slice_y

    def _sanitize_polygon_bounds(self, bounds):
        """Reset the bounds within the shape of the area."""
        try:
            (minx, miny, maxx, maxy) = bounds
        except ValueError:
            raise IncompatibleAreas('No slice on area.')
        x_bounds, y_bounds = self.area_to_crop.get_array_coordinates_from_projection_coordinates(np.array([minx, maxx]),
                                                                                                 np.array([miny, maxy]))
        y_size, x_size = self.area_to_crop.shape
        if np.all(x_bounds < 0) or np.all(y_bounds < 0) or np.all(x_bounds >= x_size) or np.all(y_bounds >= y_size):
            raise IncompatibleAreas('No slice on area.')
        return x_bounds, y_bounds

    @staticmethod
    def _create_slices_from_bounds(bounds):
        """Create slices from bounds."""
        x_bounds, y_bounds = bounds
        try:
            slice_x = slice(int(np.floor(max(np.min(x_bounds), 0))),
                            int(np.ceil(np.max(x_bounds))))
            slice_y = slice(int(np.floor(max(np.min(y_bounds), 0))),
                            int(np.ceil(np.max(y_bounds))))
        except OverflowError:
            raise IncompatibleAreas("Area not within finite bounds.")
        return expand_slice(slice_x), expand_slice(slice_y)
