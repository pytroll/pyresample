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

from pyresample.spherical import Arc

from .point import SPoint, create_spherical_point

EPSILON = 0.0000001


def _check_valid_arc(start_point, end_point):
    """Check arc validity."""
    if start_point == end_point:
        raise ValueError("An SArc can not be represented by the same start and end SPoint.")
    if start_point.is_pole() and end_point.is_pole():
        raise ValueError("An SArc can not be uniquely defined between the two poles.")
    if start_point.is_on_equator() and end_point.is_on_equator() and abs(start_point.lon - end_point.lon) == np.pi:
        raise ValueError(
            "An SArc can not be uniquely defined on the equator if start and end points are 180 degrees apart.")
    if start_point.antipode == end_point:
        raise ValueError("An SArc can not be uniquely defined between antipodal points.")


def _is_point_on_arc(point, arc):
    """Check if the point is on the arc."""
    # Define arc start and end points
    start = arc.start
    end = arc.end

    # Compute arc length
    arc_length = start.hdistance(end)

    # Distance from arc start & end points to the input point
    start_to_point_dist = start.hdistance(point)
    end_to_point_dist = end.hdistance(point)

    # Check if point is a start or end point of the arc
    point_is_on_arc_extremities = point in (start, end)

    # Check if point is on the arc segment
    point_is_on_arc_segment = abs(start_to_point_dist + end_to_point_dist - arc_length) < EPSILON

    # Assess if point is on the arc
    point_is_on_arc = point_is_on_arc_extremities or point_is_on_arc_segment
    return point_is_on_arc


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
        return self.start.lat == 0 and self.end.lat == 0

    def __eq__(self, other):
        """Check equality."""
        return self.start == other.start and self.end == other.end

    def __contains__(self, point):
        """Check if a point lies on the SArc."""
        if isinstance(point, SPoint):
            return _is_point_on_arc(point, arc=self)
        else:
            raise NotImplementedError("SArc.__contains__ currently accept only SPoint objects.")

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
        """Compute the intersection point between two great circle arcs.

        If arc and *other_arc* does not intersect, it returns None.
        If arc and *other_arc* are the same (whatever direction), it returns None.
        If arc and *other_arc* overlaps, within or contain, it returns None.
        If arc and *other_arc* intersect, it returns the intersection SPoint.
        If arc and *other_arc* touches in 1 point, it returns the intersection SPoint.

        """
        # If same arc (same direction), return None
        if self == other_arc:
            return None
        # Compute the great circle intersection points
        great_circles_intersection_spoints = self._great_circle_intersections(other_arc)
        # If a great circle intersection point lies on the arcs, it is the intersection point
        for point in great_circles_intersection_spoints:
            if point in self and point in other_arc:
                return point
        return None

    def intersects(self, other_arc):
        """Check if the current Sarc and another SArc intersect."""
        return bool(self.intersection_point(other_arc))

    # def midpoint(self):
    #     """Return the SArc midpoint SPoint."""
    #     # Retrieve start and end point in Cartesian coordinates
    #     start_xyz = self.start.to_cart().cart
    #     end_xyz = self.end.to_cart().cart
    #     # Find midpoint
    #     midpoint_xyz = (start_xyz + end_xyz) / 2.0
    #     # Normalize
    #     midpoint_xyz = CCoordinate(midpoint_xyz).normalize()
    #     # Convert back to SPoint(s)
    #     midpoint = midpoint_xyz.to_spherical(future=True)
    #     return midpoint

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

    def forward_points(self, distance, ellps="sphere"):
        """Get points at given distance(s) from SArc end point in the forward direction.

        The distance value(s) must be provided in meters.
        If the distance is positive, the point will be located outside the SArc.
        If the distance is negative, the point will be located inside the SArc.
        The function returns an SPoint or SMultiPoint.
        """
        # Define geoid
        geod = pyproj.Geod(ellps=ellps)
        # Retrieve forward and back azimuth
        fwd_az, back_az, _ = geod.inv(self.start.lon, self.start.lat,
                                      self.end.lon, self.end.lat, radians=True)
        # Retreve forward points
        distance = np.array(distance)
        lon, lat, back_az = geod.fwd(np.broadcast_to(np.array(self.end.lon), distance.shape),
                                     np.broadcast_to(np.array(self.end.lat), distance.shape),
                                     az=np.broadcast_to(fwd_az, distance.shape),
                                     dist=distance, radians=True)
        p = create_spherical_point(lon, lat)
        return p

    def backward_points(self, distance, ellps="sphere"):
        """Get points at given distance(s) from SArc start point in the backward direction.

        The distance value(s) must be provided in meters.
        If the distance is positive, the point will be located outside the SArc.
        If the distance is negative, the point will be located inside the SArc.
        The function returns an SPoint or SMultiPoint.
        """
        reverse_arc = self.reverse_direction()
        return reverse_arc.forward_points(distance=distance, ellps=ellps)

    def extend(self, distance, direction="both", ellps="sphere"):
        """Extend the SArc of a given distance in both, forward or backward directions.

        If the distance is positive, it extends the SArc.
        If the distance is negative, it shortens the SArc.
        """
        valid_direction = ["both", "forward", "backward"]
        if direction not in valid_direction:
            raise ValueError(f"Valid direction values are: {valid_direction}")
        if direction in ["both", "forward"]:
            end_point = self.forward_points(distance=distance, ellps="sphere")
        else:
            end_point = self.end

        if direction in ["both", "backward"]:
            start_point = self.backward_points(distance=distance, ellps="sphere")
        else:
            start_point = self.start
        arc = create_spherical_arcs(start_point, end_point)
        return arc

    def shorten(self, distance, direction="both", ellps="sphere"):
        """Short the SArc of a given distance in both, forward or backward directions.

        If the distance is positive, it shortens the SArc.
        If the distance is negative, it extends the SArc.
        """
        return self.extend(distance=-distance, direction=direction, ellps="sphere")

    # def segmentize(self, npts=0, max_distance=0, ellips='sphere'):
    #     """Segmentize the great-circle arc.

    #     It returns an SLine.
    #     npts or max_distance are mutually exclusively. Specify one of them.
    #     max_distance must be provided in meters.
    #     """
    #     if npts != 0:
    #         npts = npts + 2  # + 2 to account for initial and terminus points
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


def create_spherical_arcs(start_point, end_point):
    """Create a SArc or SArcs class depending on the number of points.

    If a Spoint is provided, it returns an SArc.
    If a SMultiPoint is provided, it returns an SArcs.
    """
    if isinstance(start_point, SPoint) and isinstance(end_point, SPoint):
        arc = SArc(start_point, end_point)
    else:
        raise NotImplementedError("SArcs class is not yet available.")
    return arc
