#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2023 Pyresample developers
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
"""Define the GeographicBoundary class."""

import logging

import numpy as np
from pyproj import CRS

from pyresample.boundary.area_boundary import Boundary as OldBoundary
from pyresample.boundary.base_boundary import BaseBoundary

logger = logging.getLogger(__name__)


def _is_corner_is_clockwise(lon1, lat1, corner_lon, corner_lat, lon2, lat2):
    """Determine if coordinates follow a clockwise path.

    This uses :class:`pyresample.spherical.Arc` to determine the angle
    between the first line segment (Arc) from (lon1, lat1) to
    (corner_lon, corner_lat) and the second line segment from
    (corner_lon, corner_lat) to (lon2, lat2). A straight line would
    produce an angle of 0, a clockwise path would have a negative angle,
    and a counter-clockwise path would have a positive angle.

    """
    import math

    from pyresample.spherical import Arc, SCoordinate
    point1 = SCoordinate(math.radians(lon1), math.radians(lat1))
    point2 = SCoordinate(math.radians(corner_lon), math.radians(corner_lat))
    point3 = SCoordinate(math.radians(lon2), math.radians(lat2))
    arc1 = Arc(point1, point2)
    arc2 = Arc(point2, point3)
    angle = arc1.angle(arc2)
    is_clockwise = -np.pi < angle < 0
    return is_clockwise


def _is_boundary_clockwise(sides_lons, sides_lats):
    """Determine if the boundary sides are clockwise."""
    is_clockwise = _is_corner_is_clockwise(
        lon1=sides_lons[0][-2],
        lat1=sides_lats[0][-2],
        corner_lon=sides_lons[0][-1],
        corner_lat=sides_lats[0][-1],
        lon2=sides_lons[1][1],
        lat2=sides_lats[1][1])
    return is_clockwise


class GeographicBoundary(BaseBoundary, OldBoundary):
    """GeographicBoundary object.

    The inputs must be the list of longitude and latitude boundary sides.
    """
    # NOTES:
    # - Boundary provide the ancient method contour_poly and draw
    #   from the old interface for compatibility to AreaBoundary

    @classmethod
    def _check_is_boundary_clockwise(cls, sides_x, sides_y):
        """GeographicBoundary specific implementation."""
        return _is_boundary_clockwise(sides_lons=sides_x, sides_lats=sides_y)

    def __init__(self, sides_lons, sides_lats, order=None, crs=None):
        super().__init__(sides_x=sides_lons, sides_y=sides_lats, order=order)

        self.sides_lons = self._sides_x
        self.sides_lats = self._sides_y
        self.crs = crs or CRS(proj="longlat", ellps="WGS84")
        self._contour_poly = None  # Backcompatibility with old AreaBoundary

    @property
    def lons(self):
        """Retrieve boundary longitude vertices."""
        return self._x

    @property
    def lats(self):
        """Retrieve boundary latitude vertices."""
        return self._y

    def _to_spherical_polygon(self):
        self = self.set_clockwise()  # TODO: add exception for pole wrapping polygons
        raise NotImplementedError("This will return a SPolygon in pyresample 2.0")

    def polygon(self, shapely=False):
        """Return the boundary polygon."""
        # ALTERNATIVE:
        # - shapely: to_shapely_polygon(), to_shapely_line(),
        # - pyresample spherical: to_polygon(), to_line(), polygon, line
        if shapely:
            return self._to_shapely_polygon()
        else:
            return self._to_spherical_polygon()

    def plot(self, ax=None, subplot_kw=None, **kwargs):
        """Plot the the boundary."""
        import cartopy.crs as ccrs

        from pyresample.visualization.geometries import plot_geometries

        geom = self.polygon(shapely=True)
        crs = ccrs.Geodetic()
        p = plot_geometries(geometries=[geom], crs=crs,
                            ax=ax, subplot_kw=subplot_kw, **kwargs)
        return p
