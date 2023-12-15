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

import abc
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Optional, Union

try:
    import xarray as xr
except ImportError:
    xr = None

from pyresample.geometry import AreaDefinition, CoordinateDefinition

if TYPE_CHECKING:
    # defined in typeshed to hide private C-level type
    from hashlib import _Hash

logger = logging.getLogger(__name__)


def hash_dict(the_dict: dict, existing_hash: Optional[_Hash] = None) -> _Hash:
    """Calculate a hash for a dictionary and optionally update an existing hash."""
    if existing_hash is None:
        existing_hash = hashlib.sha1()  # nosec: B324
    existing_hash.update(json.dumps(the_dict, sort_keys=True).encode('utf-8'))
    return existing_hash


def hash_resampler_geometries(source_geo_def, target_geo_def, **kwargs) -> str:
    """Get hash for the geometries used by a resampler with extra *kwargs*."""
    resampler_hash = source_geo_def.update_hash()
    resampler_hash = target_geo_def.update_hash(resampler_hash)
    resampler_hash = hash_dict(kwargs, resampler_hash)
    return resampler_hash.hexdigest()


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
    if lons.ndim > 2:
        return
    if not isinstance(lons, xr.DataArray):
        dims = ('y', 'x') if lons.ndim == 2 else ('y',)
        lons = xr.DataArray(lons, dims=dims)
        lats = xr.DataArray(lats, dims=dims)
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
    if isinstance(area, CoordinateDefinition):
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


class Resampler:
    """Base abstract resampler class."""

    def __init__(self,
                 source_geo_def: Union[CoordinateDefinition, AreaDefinition],
                 target_geo_def: Union[CoordinateDefinition, AreaDefinition],
                 cache=None,
                 ):
        """Initialize resampler with geolocation information.

        Args:
            source_geo_def:
                Geolocation definition for the data to be resampled
            target_geo_def:
                Geolocation definition for the area to resample data to.
            cache:
                :class:`~pyresample.cache.ResampleCache` instance used by
                the resampler to cache.

        """
        self.source_geo_def = source_geo_def
        self.target_geo_def = target_geo_def
        if isinstance(cache, str):
            # TODO: convenience for file based caching
            raise NotImplementedError()
        self.cache = cache

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Get version number string of the current resampler for hashing and caching.

        In cases where a resampler has changed enough that past cached files
        are not usable then this version number will be incremented to
        distinguish between the cache entries.

        """

    def _get_hash(self, **kwargs) -> str:
        """Get hash for the current resampler with the given *kwargs*."""
        kwargs.update({
            "resampler_version": self.version,
            "resampler_name": self.__class__.__name__,
        })
        return hash_resampler_geometries(self.source_geo_def, self.target_geo_def, **kwargs)

    def precompute(self, **kwargs):
        """Do the precomputation.

        This is an optional step if the subclass wants to implement more
        complex features like caching or can share some calculations
        between multiple datasets to be processed.

        """
        return None

    def resample(self, data):
        """Resample input ``data`` from the source geometry to the target geometry.

        Args:
            data (ArrayLike): Data to be resampled

        Returns (ArrayLike): Data resampled to the target area

        """
        self.precompute()
        raise NotImplementedError("This method must be implemented by a subclass.")
