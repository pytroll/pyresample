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

from pyresample.boundary.sides import BoundarySides
from pyresample.spherical import SphPolygon

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


class GeographicBoundary():
    """GeographicBoundary object.

    The inputs must be the list of longitude and latitude boundary sides.
    """

    def __init__(self, sides_lons, sides_lats, order=None):

        self.sides_lons = BoundarySides(sides_lons)
        self.sides_lats = BoundarySides(sides_lats)

        # Old interface for compatibility to AreaBoundary
        self._contour_poly = None

        # Check if it is clockwise/counterclockwise
        self.is_clockwise = _is_boundary_clockwise(sides_lons=sides_lons,
                                                   sides_lats=sides_lats)
        self.is_counterclockwise = not self.is_clockwise

        # Define wished order
        if self.is_clockwise:
            self._actual_order = "clockwise"
        else:
            self._actual_order = "counterclockwise"

        if order is None:
            self._wished_order = self._actual_order
        else:
            if order not in ["clockwise", "counterclockwise"]:
                raise ValueError("Valid 'order' is 'clockwise' or 'counterclockwise'")
            self._wished_order = order

    def set_clockwise(self):
        """Set clockwise order for vertices retrieval."""
        self._wished_order = "clockwise"
        return self

    def set_counterclockwise(self):
        """Set counterclockwise order for vertices retrieval."""
        self._wished_order = "counterclockwise"
        return self

    @property
    def sides(self):
        """Return the boundary sides as a tuple of (sides_lons, sides_lats) arrays."""
        return self.sides_lons, self.sides_lats

    @property
    def lons(self):
        """Retrieve boundary longitude vertices."""
        lons = np.concatenate([lns[:-1] for lns in self.sides_lons])
        if self._wished_order == self._actual_order:
            return lons
        else:
            return lons[::-1]

    @property
    def lats(self):
        """Retrieve boundary latitude vertices."""
        lats = np.concatenate([lts[:-1] for lts in self.sides_lats])
        if self._wished_order == self._actual_order:
            return lats
        else:
            return lats[::-1]

    @property
    def vertices(self):
        """Return boundary vertices 2D array [lon, lat]."""
        vertices = np.vstack((self.lons, self.lats)).T
        vertices = vertices.astype(np.float64, copy=False)  # Important for spherical ops.
        return vertices

    def contour(self, closed=False):
        """Return the (lons, lats) tuple of the boundary object.

        If excludes the last element of each side because it's included in the next side.
        If closed=False (the default), the last vertex is not equal to the first vertex
        If closed=True, the last vertex is set to be equal to the first
        closed=True is required for shapely Polygon creation.
        closed=False is required for pyresample SPolygon creation.
        """
        lons = self.lons
        lats = self.lats
        if closed:
            lons = np.hstack((lons, lons[0]))
            lats = np.hstack((lats, lats[0]))
        return lons, lats

    def _to_shapely_polygon(self):
        from shapely.geometry import Polygon
        self = self.set_counterclockwise()  # TODO: add exception for pole wrapping polygons
        lons, lats = self.contour(closed=True)
        return Polygon(zip(lons, lats))

    def _to_spherical_polygon(self):
        self = self.set_clockwise()  # TODO: add exception for pole wrapping polygons
        raise NotImplementedError("This will return a SPolygon in pyresample 2.0")

    def polygon(self, shapely=False):
        """Return the boundary polygon."""
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

    # For backward compatibility !
    def decimate(self, ratio):
        """Remove some points in the boundaries, but never the corners."""
        # TODO: to update --> used by AreaDefBoundary
        for i in range(len(self.sides_lons)):
            length = len(self.sides_lons[i])
            start = int((length % ratio) / 2)
            points = np.concatenate(([0], np.arange(start, length, ratio),
                                     [length - 1]))
            if points[1] == 0:
                points = points[1:]
            if points[-2] == (length - 1):
                points = points[:-1]
            self.sides_lons[i] = self.sides_lons[i][points]
            self.sides_lats[i] = self.sides_lats[i][points]

    @property
    def contour_poly(self):
        """Return the pyresample SphPolygon."""
        if self._contour_poly is None:
            self._contour_poly = SphPolygon(np.deg2rad(self.vertices))
        return self._contour_poly

    def draw(self, mapper, options, **more_options):
        """Draw the current boundary on the *mapper*."""
        self.contour_poly.draw(mapper, options, **more_options)
