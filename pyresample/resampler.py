#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019

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
"""Base resampler class made for subclassing."""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache, partial
from uuid import uuid4

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

from pyresample.geometry import (SwathDefinition, AreaDefinition, IncompatibleAreas,
                                 get_geostationary_bounding_box_in_proj_coords, InvalidArea)
from pyproj.transformer import Transformer


logger = logging.getLogger(__name__)


def hash_dict(the_dict, the_hash=None):
    """Calculate a hash for a dictionary."""
    if the_hash is None:
        the_hash = hashlib.sha1()
    the_hash.update(json.dumps(the_dict, sort_keys=True).encode('utf-8'))
    return the_hash


def _data_arr_needs_xy_coords(data_arr, area):
    coords_exist = 'x' in data_arr.coords and 'y' in data_arr.coords
    no_xy_dims = 'x' not in data_arr.dims or 'y' not in data_arr.dims
    has_proj_vectors = hasattr(area, 'get_proj_vectors')
    return not (coords_exist or no_xy_dims or not has_proj_vectors)


def _add_xy_units(crs, x_attrs, y_attrs):
    if crs is not None:
        units = crs.axis_info[0].unit_name
        # fix udunits/CF standard units
        units = units.replace('metre', 'meter')
        if units == 'degree':
            y_attrs['units'] = 'degrees_north'
            x_attrs['units'] = 'degrees_east'
        else:
            y_attrs['units'] = units
            x_attrs['units'] = units


def add_xy_coords(data_arr, area, crs=None):
    """Assign x/y coordinates to DataArray from provided area.

    If 'x' and 'y' coordinates already exist then they will not be added.

    Args:
        data_arr (xarray.DataArray): data object to add x/y coordinates to
        area (pyresample.geometry.AreaDefinition): area providing the
            coordinate data.
        crs (pyproj.crs.CRS or None): CRS providing additional information
            about the area's coordinate reference system if available.
            Requires pyproj 2.0+.

    Returns (xarray.DataArray): Updated DataArray object

    """
    if not _data_arr_needs_xy_coords(data_arr, area):
        return data_arr

    x, y = area.get_proj_vectors()
    # convert to DataArrays
    y_attrs = {}
    x_attrs = {}
    _add_xy_units(crs, x_attrs, y_attrs)
    y = xr.DataArray(y, dims=('y',), attrs=y_attrs)
    x = xr.DataArray(x, dims=('x',), attrs=x_attrs)
    return data_arr.assign_coords(y=y, x=x)


def _find_and_assign_crs(data_arr, area):
    # add CRS object if pyproj 2.0+
    try:
        from pyproj import CRS
    except ImportError:
        logger.debug("Could not add 'crs' coordinate with pyproj<2.0")
        crs = None
    else:
        # default lat/lon projection
        latlon_proj = "+proj=latlong +datum=WGS84 +ellps=WGS84"
        # otherwise get it from the area definition
        if hasattr(area, 'crs'):
            crs = area.crs
        else:
            proj_str = getattr(area, 'proj_str', latlon_proj)
            crs = CRS.from_string(proj_str)
        data_arr = data_arr.assign_coords(crs=crs)
    return data_arr, crs


def _update_swath_lonlat_attrs(area):
    # add lon/lat arrays for swath definitions
    # SwathDefinitions created by Satpy should be assigning DataArray
    # objects as the lons/lats attributes so use those directly to
    # maintain original .attrs metadata (instead of converting to dask
    # array).
    lons = area.lons
    lats = area.lats
    lons.attrs.setdefault('standard_name', 'longitude')
    lons.attrs.setdefault('long_name', 'longitude')
    lons.attrs.setdefault('units', 'degrees_east')
    lats.attrs.setdefault('standard_name', 'latitude')
    lats.attrs.setdefault('long_name', 'latitude')
    lats.attrs.setdefault('units', 'degrees_north')
    # See https://github.com/pydata/xarray/issues/3068
    # data_arr = data_arr.assign_coords(longitude=lons, latitude=lats)


def add_crs_xy_coords(data_arr, area):
    """Add :class:`pyproj.crs.CRS` and x/y or lons/lats to coordinates.

    For SwathDefinition or GridDefinition areas this will add a
    `crs` coordinate and coordinates for the 2D arrays of `lons` and `lats`.

    For AreaDefinition areas this will add a `crs` coordinate and the
    1-dimensional `x` and `y` coordinate variables.

    Args:
        data_arr (xarray.DataArray): DataArray to add the 'crs'
            coordinate.
        area (pyresample.geometry.AreaDefinition): Area to get CRS
            information from.

    """
    data_arr, crs = _find_and_assign_crs(data_arr, area)

    # Add x/y coordinates if possible
    if isinstance(area, SwathDefinition):
        _update_swath_lonlat_attrs(area)
    else:
        # Gridded data (AreaDefinition/StackedAreaDefinition)
        data_arr = add_xy_coords(data_arr, area, crs=crs)
    return data_arr


def update_resampled_coords(old_data, new_data, new_area):
    """Add coordinate information to newly resampled DataArray.

    Args:
        old_data (xarray.DataArray): Old data before resampling.
        new_data (xarray.DataArray): New data after resampling.
        new_area (pyresample.geometry.BaseDefinition): Area definition
            for the newly resampled data.

    """
    # copy over other non-x/y coordinates
    # this *MUST* happen before we set 'crs' below otherwise any 'crs'
    # coordinate in the coordinate variables we are copying will overwrite the
    # 'crs' coordinate we just assigned to the data
    ignore_coords = ('y', 'x', 'crs')
    new_coords = {}
    for cname, cval in old_data.coords.items():
        # we don't want coordinates that depended on the old x/y dimensions
        has_ignored_dims = any(dim in cval.dims for dim in ignore_coords)
        if cname in ignore_coords or has_ignored_dims:
            continue
        new_coords[cname] = cval
    new_data = new_data.assign_coords(**new_coords)

    # add crs, x, and y coordinates
    new_data = add_crs_xy_coords(new_data, new_area)
    # make sure the new area is assigned to the attributes
    new_data.attrs['area'] = new_area
    return new_data


class BaseResampler(object):
    """Base abstract resampler class."""

    def __init__(self, source_geo_def, target_geo_def):
        """Initialize resampler with geolocation information.

        Args:
            source_geo_def (SwathDefinition, AreaDefinition):
                Geolocation definition for the data to be resampled
            target_geo_def (CoordinateDefinition, AreaDefinition):
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


def _enumerate_chunk_slices(chunks):
    """Enumerate chunks with slices."""
    for position in np.ndindex(tuple(map(len, (chunks)))):
        slices = []
        for pos, chunk in zip(position, chunks):
            chunk_size = chunk[pos]
            offset = sum(chunk[:pos])
            slices.append(slice(offset, offset + chunk_size))

        yield (position, slices)


class DaskResampler:
    """Resampler that uses dask for processing the data chunkwise.

    This works by generating a dask graph for the destination array (the array
    that will contain the resampled data), and using cropped out versions of the
    input data as dependencies.
    """

    def __init__(self, source_geo_def, target_geo_def, resampler, **resampler_kwargs):
        """Initialize the class."""
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def
        self.resampler = partial(resampler, **resampler_kwargs)
        self.resampler.__name__ = resampler.__name__

    def resample(self, data: da.Array, chunks=None) -> da.Array:
        """Resample the provided dask array.

        The input array has to be at least two-dimensional and expects the last
        two dimensions to be respectively y and x.
        """
        name = self.resampler.__name__ + '-' + uuid4().hex
        output_shape = data.shape[:-2] + self.target_geo_def.shape
        dst_chunks = da.core.normalize_chunks(chunks or data.chunksize, output_shape)
        dask_graph = dict()
        deps = [data]
        for position, slices in _enumerate_chunk_slices(dst_chunks):
            target_geo_def = self.target_geo_def[slices[-2:]]
            try:
                smaller_data, source_geo_def = self.crop_data_around_area(data, target_geo_def)
            except IncompatibleAreas:  # no relevant data matching
                chunk_shape = [chunk[pos] for pos, chunk in zip(position, dst_chunks)]
                dask_graph[(name, *position)] = (
                    np.full,
                    chunk_shape,
                    np.nan
                )
            else:
                deps.append(smaller_data)
                if isinstance(source_geo_def, SwathDefinition):
                    deps.append(source_geo_def.lons.data)
                    deps.append(source_geo_def.lats.data)
                    source_geo_def = [(source_geo_def.lons.data.name, 0, 0), (source_geo_def.lats.data.name, 0, 0)]
                dask_graph[(name, *position)] = (
                    self.resampler,
                    (smaller_data.name, *position[:-2], 0, 0),
                    source_geo_def,
                    target_geo_def,
                )

        dask_graph = HighLevelGraph.from_collections(name, dask_graph, dependencies=deps)

        return da.Array(dask_graph, name, chunks=dst_chunks, dtype=data.dtype, shape=output_shape)

    def crop_data_around_area(self, data, target_geo_def):
        """Crop the data around the provided area."""
        slicer = Slicer(self.source_geo_def, target_geo_def)
        x_slice, y_slice = slicer.get_slices()

        source_geo_def = self.source_geo_def[y_slice, x_slice]
        if isinstance(source_geo_def, SwathDefinition):
            source_geo_def.lons.data = source_geo_def.lons.data.rechunk((-1, -1))
            source_geo_def.lats.data = source_geo_def.lats.data.rechunk((-1, -1))
        smaller_data = data[..., y_slice, x_slice].rechunk(data.chunks[:-2] + (-1, -1))
        return smaller_data, source_geo_def


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
        x, y = self.area_to_contain.get_bbox_coords(10)
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
