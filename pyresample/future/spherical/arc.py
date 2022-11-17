#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Pyresample developers
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
"""Define a great-circle arc between two points on the sphere through the SArc class."""

import numpy as np
import pyproj
from shapely.geometry import LineString

from pyresample.future.spherical import SPoint
from pyresample.spherical import Arc

EPSILON = 0.0000001


def _check_valid_arc(start_point, end_point):
    """Check arc validity."""
    if start_point == end_point:
        raise ValueError("An SArc can not be represented by the same start and end SPoint.")
    if start_point.is_pole() and end_point.is_pole():
        raise ValueError("An SArc can not be uniquely defined between the two poles.")
    if start_point.is_on_equator and end_point.is_on_equator and abs(start_point.lon - end_point.lon) == np.pi:
        raise ValueError(
            "An SArc can not be uniquely defined on the equator if start and end points are 180 degrees apart.")


class SArc(Arc):
    """Object representing a great-circle arc between two points on a sphere.

    The ``start_point`` and ``end_point`` must be SPoint objects.
    The great-circle arc is defined as the shortest track(s) between the two points.
    Between the north and south pole there are an infinite number of great-circle arcs.
    """

    def __init__(self, start_point, end_point):
        _check_valid_arc(start_point, end_point)
        super().__init__(start_point, end_point)

    def __hash__(self):
        """Define SArc hash."""
        return hash((float(self.start.lon), float(self.start.lat),
                     float(self.end.lon), float(self.end.lat)))

    def is_on_equator(self):
        """Check if the SArc lies on the equator."""
        if self.start.lat == 0 and self.end.lat == 0:
            return True
        return False

    def __eq__(self, other):
        """Check equality."""
        if self.start == other.start and self.end == other.end:
            return True
        return False

    def reverse_direction(self):
        """Reverse SArc direction."""
        return SArc(self.end, self.start)

    @property
    def vertices(self):
        """Get start SPoint and end SPoint (radians) vertices array."""
        return self.start.vertices, self.end.vertices

    @property
    def vertices_in_degrees(self):
        """Get start SPoint and end SPoint (degrees) vertices array."""
        return self.start.vertices_in_degrees, self.end.vertices_in_degrees

    def get_next_intersection(self, other_arc):
        """Overwrite a Arc deprecated function. Inherited from Arc class."""
        raise ValueError("This function is deprecated.")

    def intersections(self, other_arc):
        """Overwritea Arc deprecated function. Inherited from Arc class."""
        raise ValueError("'SArc.intersections' is deprecated. Use '_great_circle_intersections' instead.")

    def _great_circle_intersections(self, other_arc):
        """Compute the intersections points of the greats circles over which the arcs lies.

        A great circle divides the sphere in two equal hemispheres.
        """
        end_lon = self.end.lon
        other_end_lon = other_arc.end.lon

        if self.end.lon - self.start.lon > np.pi:
            end_lon -= 2 * np.pi
        if other_arc.end.lon - other_arc.start.lon > np.pi:
            other_end_lon -= 2 * np.pi
        if self.end.lon - self.start.lon < -np.pi:
            end_lon += 2 * np.pi
        if other_arc.end.lon - other_arc.start.lon < -np.pi:
            other_end_lon += 2 * np.pi

        end_point = SPoint(end_lon, self.end.lat)
        other_end_point = SPoint(other_end_lon, other_arc.end.lat)

        ea_ = self.start.cross2cart(end_point).normalize()
        eb_ = other_arc.start.cross2cart(other_end_point).normalize()

        cross = ea_.cross(eb_)
        lat = np.arctan2(cross.cart[2],
                         np.sqrt(cross.cart[0] ** 2 + cross.cart[1] ** 2))
        lon = np.arctan2(cross.cart[1], cross.cart[0])

        return (SPoint(lon, lat), SPoint(lon + np.pi, -lat))

    def intersection(self, other_arc):
        """Overwrite a Arc deprecated function. Inherited from Arc class."""
        raise ValueError("'SArc.intersection' is deprecated. Use 'intersection_point' instead.")

    def intersection_point(self, other_arc):
        """Compute the intersection point between two arcs.

        If arc and *other_arc* intersect, it returns the intersection SPoint.
        If arc and *other_arc* does not intersect, it returns None.
        If same arc (also same direction), it returns None.
        """
        # If same arc (same direction), return None
        if self == other_arc:
            return None

        great_circles_intersection_spoints = self._great_circle_intersections(other_arc)

        for spoint in great_circles_intersection_spoints:
            a = self.start
            b = self.end
            c = other_arc.start
            d = other_arc.end

            ab_dist = a.hdistance(b)
            cd_dist = c.hdistance(d)
            ap_dist = a.hdistance(spoint)
            bp_dist = b.hdistance(spoint)
            cp_dist = c.hdistance(spoint)
            dp_dist = d.hdistance(spoint)

            if (((spoint in (a, b)) or (abs(ap_dist + bp_dist - ab_dist) < EPSILON)) and
                    ((spoint in (c, d)) or (abs(cp_dist + dp_dist - cd_dist) < EPSILON))):
                return spoint
        return None

    def intersects(self, other_arc):
        """Check if the current Sarc and another SArc intersect."""
        return bool(self.intersection_point(other_arc))

    def midpoint(self, ellips='sphere'):
        """Return the SArc midpoint SPoint."""
        geod = pyproj.Geod(ellps=ellips)
        lon_start = self.start.lon
        lon_end = self.end.lon
        lat_start = self.start.lat
        lat_end = self.end.lat
        lon_mid, lat_mid = geod.npts(lon_start, lat_start, lon_end, lat_end, npts=1, radians=True)[0]
        return SPoint(lon_mid, lat_mid)

    def to_shapely(self):
        """Convert to Shapely LineString."""
        start_coord, end_coord = self.vertices_in_degrees
        return LineString((start_coord.tolist()[0], end_coord.tolist()[0]))

    # def segmentize(self, npts=0, max_distance=0, ellips='sphere'):
    #     """Segmentize the great-circle arc.

    #     It returns an SLine.
    #     npts or max_distance are mutually exclusively. Specify one of them.
    #     max_distance must be provided in kilometers.
    #     """
    #     if npts != 0:
    #         npts = npts + 2  # + 2 to account for initial and terminus
    #     geod = pyproj.Geod(ellps=ellips)
    #     lon_start = self.start.lon
    #     lon_end = self.end.lon
    #     lat_start = self.start.lat
    #     lat_end = self.end.lat
    #     points = geod.inv_intermediate(lon_start, lat_start, lon_end, lat_end,
    #                                    del_s=max_distance,
    #                                    npts=npts,
    #                                    radians=True,
    #                                    initial_idx=0, terminus_idx=0)
    #     lons, lats = (points.lons, points.lats)
    #     lons = np.asarray(lons)
    #     lats = np.asarray(lats)
    #     vertices = np.stack((lons, lats)).T
    #     return SLine(vertices)

    # def to_line(self):
    #     """Convert to SLine."""
    #     vertices = np.vstack(self.vertices)
    #     return SLine(vertices)

    # def plot(self, *args, **kwargs):
    #     """Convert to SLine."""
    #     self.to_line.plot(*args, **kwargs)
