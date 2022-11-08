#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-2022 Pyresample Developers
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
"""Definition of SExtent class."""
import numbers

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def bounds_from_extent(extent):
    """Get shapely bounds from a matplotlib/cartopy extent.

    Shapely bounds: [x_min, y_min, x_max, y_max]
    Matplotlib/Cartopy extent: [x_min, x_max, y_min, y_max]
    """
    bounds = [extent[0], extent[2], extent[1], extent[3]]
    return bounds


def _get_list_extents_from_args(args):
    """Get list_extent from SExtent input arguments."""
    args = list(args)
    # Check something is passed to SExtent
    if len(args) == 0:
        raise ValueError("No extent passed to SExtent.")
    # Check that the first args element is a list, tuple of np.array
    if np.any([not isinstance(arg, (list, tuple, np.ndarray)) for arg in args]):
        raise TypeError("Specify the extent(s) using list, tuple or np.array.")
    # Get list extent
    # - If SExtent(extent) or SExtent(list_extent)
    if len(args) == 1:
        list_extents = list(*args)
        if len(list_extents) == 0:  # input case: [], or ()
            raise ValueError("No extent passed to SExtent.")
    else:
        list_extents = args
    # If a single list, check correct length and contain a number
    if len(list_extents) == 4 and isinstance(list_extents[0], numbers.Number):
        list_extents = [list_extents]
    # Ensure that here after is a list of extents
    if not isinstance(list_extents[0], (tuple, list)):
        raise ValueError("SExtent expects a single extent or a list of extents.")
    return list_extents


def _check_extent_order(extent):
    """Check extent order validity."""
    if extent[0] > extent[1]:
        raise ValueError('extent[0] (lon_min) must be smaller than extent[1] (lon_max).')
    if extent[2] > extent[3]:
        raise ValueError('extent[2] (lat_min) must be smaller than extent[3] (lat_max).')


def _check_extent_values(extent, use_radians=False):
    """Check extent value validity."""
    lat_valid_range = [-90.0, 90.0]
    lon_valid_range = [-180.0, 180.0]
    if use_radians:
        lat_valid_range = np.deg2rad(lat_valid_range)
        lon_valid_range = np.deg2rad(lon_valid_range)
    if np.any(np.logical_or(extent[0:2] < lon_valid_range[0], extent[0:2] > lon_valid_range[1])):
        raise ValueError(f'extent longitude values must be within {lon_valid_range}.')
    if np.any(np.logical_or(extent[2:] < lat_valid_range[0], extent[2:] > lat_valid_range[1])):
        raise ValueError(f'extent latitude values must be within {lat_valid_range}.')


def _check_valid_extent(extent, use_radians=False):
    """Check lat/lon extent validity."""
    # Check extent dtype
    if not isinstance(extent, (tuple, list, np.ndarray)):
        raise TypeError("'extent' must be a list, tuple or np.array. [lon_min, lon_max, lat_min, lat_max].")
    # Check extent length
    if len(extent) != 4:
        raise ValueError("'extent' must have length 4: [lon_min, lon_max, lat_min, lat_max].")
    # Convert to np.array
    extent = np.array(extent)
    # Check order and values
    _check_extent_order(extent)
    _check_extent_values(extent, use_radians=use_radians)
    # Ensure is a list
    extent = extent.tolist()
    return extent


def _check_topology_validity(polygons):
    """Check that the extents polygons do not overlap."""
    from shapely.topology import TopologicalError
    try:
        # TODO: improve to raise error also when duplicate geometries or within on another
        # TopologicalError: The operation 'GEOSIntersects_r' could not be performed.
        # Likely cause is invalidity of the geometry
        polygons.intersects(polygons)
    except TopologicalError:
        raise ValueError("The extent list is not valid. The composing extents must not overlap each other.")


class SExtent(object):
    """Spherical Extent.

    SExtent longitudes are defined between -180 and 180 degree.
    A spherical geometry crossing the anti-meridian will have an SExtent
     composed of [lon_start, 180, ...,...] and [-180, lon_end, ..., ...]

    The extents composing an SExtent:
    - can not intersect/overlap each other
    - can touch each other

    There is not an upper limit on the number of extents composing SExtent.
    The only conditions is that the extents do not intersect/overlap.

    Examples of valid SExtent inputs
      extent = [x_min, x_max, y_min, ymax]
      sext = SExtent(extent)
      sext = SExtent([extent, extent])
      sext = SExtent(extent, extent, ...)

    """

    def __init__(self, *args):
        # Get list of extents
        list_extents = _get_list_extents_from_args(args)
        # Check extents format and value validity
        self.list = [_check_valid_extent(ext) for ext in list_extents]
        # Pre-compute shapely polygon
        list_polygons = [Polygon.from_bounds(*bounds_from_extent(ext)) for ext in self.list]
        self.polygons = MultiPolygon(list_polygons)
        # Check topological validitiy of extent polygons
        _check_topology_validity(self.polygons)

    def to_shapely(self):
        """Return the shapely extent rectangle(s) polygon(s)."""
        return self.polygons

    def __str__(self):
        """Get simplified representation of SExtent."""
        return str(self.list)

    def __repr__(self):
        """Get simplified representation of SExtent."""
        return str(self.list)

    def __iter__(self):
        """Get extents iterator."""
        return self.list.__iter__()

    @property
    def is_global(self):
        """Check if the extent is global."""
        if len(self.list) != 1:
            return False
        if self.list[0] == [-180, 180, -90, 90]:
            return True
        else:
            return False

    def intersects(self, other):
        """Check if SExtent is intersecting the other SExtent.

        Touching extent are considered to not intersect !
        """
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.intersects() expects a SExtent class instance.")
        bl = (self.to_shapely().intersects(other.to_shapely()) and
              not self.to_shapely().touches(other.to_shapely()))
        return bl

    def disjoint(self, other):
        """Check if SExtent does not intersect (and do not touch) the other SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.intersects() expects a SExtent class instance.")
        return self.to_shapely().disjoint(other.to_shapely())

    def within(self, other):
        """Check if SExtent is within another SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.within() expects a SExtent class instance.")
        return self.to_shapely().within(other.to_shapely())

    def contains(self, other):
        """Check if the SExtent contains another SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.contains() expects a SExtent class instance.")
        return self.to_shapely().contains(other.to_shapely())

    def touches(self, other):
        """Check if SExtent external touches another SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.touches() expects a SExtent class instance.")
        return self.to_shapely().touches(other.to_shapely())

    def equals(self, other):
        """Check if SExtent is equals to SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.equals() expects a SExtent class instance.")
        return self.to_shapely().equals(other.to_shapely())

    # TODO: equals_exact
    # - Choose whether to use shapely for equals and equals_exact

    def plot(self, ax=None, **plot_kwargs):
        """Plot the SLine using Cartopy."""
        import matplotlib.pyplot as plt
        try:
            import cartopy.crs as ccrs
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install cartopy to plot spherical geometries.")

        # Retrieve shapely polygon
        geom = self.to_shapely()

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Add extent polygon
        ax.add_geometries([geom], crs=ccrs.PlateCarree(), **plot_kwargs)
        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
        return ax
