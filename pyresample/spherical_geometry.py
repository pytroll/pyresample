#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2021 Pyresample developers
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
"""Classes for spherical geometry operations."""

from __future__ import absolute_import

import math
import warnings

import numpy as np

warnings.warn("This module will be removed in pyresample 2.0, please use the "
              "`pyresample.spherical` module functions and class instead.",
              DeprecationWarning, stacklevel=2)

EPSILON = 0.0000001

# FIXME: this has not been tested with R != 1


class Coordinate(object):
    """Point on earth in terms of lat and lon.

    It expects lon,lat in degrees
    But self.lat and self.lon are returned in radians !
    """

    lat = None
    lon = None
    x__ = None
    y__ = None
    z__ = None

    def __init__(self, lon=None, lat=None,
                 x__=None, y__=None, z__=None, R__=1):
        self.R__ = R__
        if lat is not None and lon is not None:
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError('Illegal (lon, lat) coordinates: (%s, %s)'
                                 % (lon, lat))
            self.lat = math.radians(lat)
            self.lon = math.radians(lon)
            self._update_cart()
        else:
            self.x__ = x__
            self.y__ = y__
            self.z__ = z__
            self._update_lonlat()

    def _update_cart(self):
        """Convert lon/lat to cartesian coordinates."""
        self.x__ = math.cos(self.lat) * math.cos(self.lon)
        self.y__ = math.cos(self.lat) * math.sin(self.lon)
        self.z__ = math.sin(self.lat)

    def _update_lonlat(self):
        """Convert cartesian to lon/lat."""
        self.lat = math.degrees(math.asin(self.z__ / self.R__))
        self.lon = math.degrees(math.atan2(self.y__, self.x__))

    def __ne__(self, other):
        """Check inequality."""
        if (abs(self.lat - other.lat) < EPSILON and
                abs(self.lon - other.lon) < EPSILON):
            return 0
        else:
            return 1

    def __eq__(self, other):
        """Check equality."""
        return not self.__ne__(other)

    def __str__(self):
        """Get simplified representation of lon/lats in degrees."""
        return str((math.degrees(self.lon), math.degrees(self.lat)))

    def __repr__(self):
        """Get simplified representation of lon/lats in degrees."""
        return str((math.degrees(self.lon), math.degrees(self.lat)))

    def cross2cart(self, point):
        """Compute the cross product, and convert to cartesian coordinates (assuming radius 1)."""
        lat1 = self.lat
        lon1 = self.lon
        lat2 = point.lat
        lon2 = point.lon

        res = Coordinate(
            x__=(math.sin(lat1 - lat2) * math.sin((lon1 + lon2) / 2) *
                 math.cos((lon1 - lon2) / 2) - math.sin(lat1 + lat2) *
                 math.cos((lon1 + lon2) / 2) * math.sin((lon1 - lon2) / 2)),
            y__=(math.sin(lat1 - lat2) * math.cos((lon1 + lon2) / 2) *
                 math.cos((lon1 - lon2) / 2) + math.sin(lat1 + lat2) *
                 math.sin((lon1 + lon2) / 2) * math.sin((lon1 - lon2) / 2)),
            z__=(math.cos(lat1) * math.cos(lat2) * math.sin(lon1 - lon2)))

        return res

    def distance(self, point):
        """Get distance using Vincenty formula."""
        dlambda = self.lon - point.lon
        num = ((math.cos(point.lat) * math.sin(dlambda)) ** 2 +
               (math.cos(self.lat) * math.sin(point.lat) -
                math.sin(self.lat) * math.cos(point.lat) *
                math.cos(dlambda)) ** 2)
        den = (math.sin(self.lat) * math.sin(point.lat) +
               math.cos(self.lat) * math.cos(point.lat) * math.cos(dlambda))

        return math.atan2(math.sqrt(num), den)

    def norm(self):
        """Return the norm of the vector."""
        return math.sqrt(self.x__ ** 2 + self.y__ ** 2 + self.z__ ** 2)

    def normalize(self):
        """Normalize the vector."""
        norm = self.norm()
        self.x__ /= norm
        self.y__ /= norm
        self.z__ /= norm

        return self

    def cross(self, point):
        """Get cross product with another vector."""
        x__ = self.y__ * point.z__ - self.z__ * point.y__
        y__ = self.z__ * point.x__ - self.x__ * point.z__
        z__ = self.x__ * point.y__ - self.y__ * point.x__

        return Coordinate(x__=x__, y__=y__, z__=z__)

    def dot(self, point):
        """Get dot product with another vector."""
        return (self.x__ * point.x__ +
                self.y__ * point.y__ +
                self.z__ * point.z__)


class Arc(object):
    """An arc of the great circle between two points."""

    def __init__(self, start, end):
        self.start, self.end = start, end

    def center_angle(self):
        """Get angle of an arc at the center of the sphere."""
        val = (math.cos(self.start.lat - self.end.lat) +
               math.cos(self.start.lon - self.end.lon) - 1)

        if val > 1:
            val = 1
        elif val < -1:
            val = -1

        return math.acos(val)

    def __eq__(self, other):
        """Check equality."""
        if self.start == other.start and self.end == other.end:
            return 1
        return 0

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)

    def __str__(self):
        """Get simplified representation."""
        return str((str(self.start), str(self.end)))

    def angle(self, other_arc, snap=True):
        """Get oriented angle between two arcs.

        Parameters
        ----------
        other_arc : pyresample.spherical_geometry.Arc
        snap : boolean
            Snap small angles to 0. Allows for detecting colinearity. Disable
            snapping when calculating polygon areas as it might lead to
            negative area values.
        """
        if self.start == other_arc.start:
            a__ = self.start
            b__ = self.end
            c__ = other_arc.end
        elif self.start == other_arc.end:
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
        elif self.end == other_arc.end:
            a__ = self.end
            b__ = self.start
            c__ = other_arc.start
        elif self.end == other_arc.start:
            a__ = self.end
            b__ = self.start
            c__ = other_arc.end
        else:
            raise ValueError("No common point in angle computation.")

        ua_ = a__.cross(b__)
        ub_ = a__.cross(c__)

        val = ua_.dot(ub_) / (ua_.norm() * ub_.norm())
        angle = self._convert_to_angle(val, snap_to_zero=snap)

        n__ = ua_.normalize()
        return -angle if n__.dot(c__) > 0 else angle

    @staticmethod
    def _convert_to_angle(val: float, snap_to_zero: bool) -> float:
        if snap_to_zero:
            if abs(val - 1) < EPSILON:
                return 0
            elif abs(val + 1) < EPSILON:
                return math.pi
        else:
            if 0 <= val - 1 < EPSILON:
                return 0
            elif -EPSILON < val + 1 <= 0:
                return math.pi
        return math.acos(val)

    def intersections(self, other_arc):
        """Get the two intersections of the greats circles defined by the current arc and *other_arc*."""
        end_lon = self.end.lon
        other_end_lon = other_arc.end.lon

        if self.end.lon - self.start.lon > math.pi:
            end_lon -= 2 * math.pi
        if other_arc.end.lon - other_arc.start.lon > math.pi:
            other_end_lon -= 2 * math.pi
        if self.end.lon - self.start.lon < -math.pi:
            end_lon += 2 * math.pi
        if other_arc.end.lon - other_arc.start.lon < -math.pi:
            other_end_lon += 2 * math.pi

        end_point = Coordinate(math.degrees(modpi(end_lon)), math.degrees(self.end.lat))
        other_end_point = Coordinate(math.degrees(modpi(other_end_lon)), math.degrees(other_arc.end.lat))

        ea_ = self.start.cross2cart(end_point).normalize()
        eb_ = other_arc.start.cross2cart(other_end_point).normalize()

        cross = ea_.cross(eb_)
        lat = math.atan2(cross.z__, math.sqrt(cross.x__ ** 2 + cross.y__ ** 2))
        lon = math.atan2(-cross.y__, cross.x__)

        return (Coordinate(math.degrees(lon), math.degrees(lat)),
                Coordinate(math.degrees(modpi(lon + math.pi)), math.degrees(-lat)))

    def intersects(self, other_arc):
        """Determine if this arc and another arc intersect.

        An arc is defined as the shortest tracks between two points.
        """
        return bool(self.intersection(other_arc))

    def intersection(self, other_arc):
        """Determine the intersection point between this arc and another.

        An arc is defined as the shortest tracks between two points.
        """
        for i in self.intersections(other_arc):
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
            d__ = other_arc.end

            ab_ = a__.distance(b__)
            cd_ = c__.distance(d__)

            if (abs(a__.distance(i) + b__.distance(i) - ab_) < EPSILON and
                    abs(c__.distance(i) + d__.distance(i) - cd_) < EPSILON):
                return i
        return None


def modpi(val):
    """Put *val* between -pi and pi."""
    return (val + math.pi) % (2 * math.pi) - math.pi


def get_polygon_area(corners):
    """Get the area of the convex area defined by *corners*."""
    # We assume the earth is spherical !!!
    # Should be the radius of the earth at the observed position
    R = 1

    c1_ = corners[0]
    area = 0

    for idx in range(1, len(corners) - 1):
        b1_ = Arc(c1_, corners[idx])
        b2_ = Arc(c1_, corners[idx + 1])
        b3_ = Arc(corners[idx], corners[idx + 1])
        e__ = (abs(b1_.angle(b2_, snap=False)) +
               abs(b2_.angle(b3_, snap=False)) +
               abs(b3_.angle(b1_, snap=False)))
        area += e__ - math.pi
    return R ** 2 * area


def get_intersections(b__, boundaries):
    """Get the intersections of *b__* with *boundaries*.

    Returns both the intersection coordinates and the concerned
    boundaries.
    """
    intersections = []
    bounds = []
    for other_b in boundaries:
        inter = b__.intersection(other_b)
        if inter is not None:
            intersections.append(inter)
            bounds.append(other_b)
    return intersections, bounds


def get_first_intersection(b__, boundaries):
    """Get the first intersection on *b__* with *boundaries*."""
    intersections, bounds = get_intersections(b__, boundaries)
    del bounds
    dists = np.array([b__.start.distance(p__) for p__ in intersections])
    indices = dists.argsort()
    if len(intersections) > 0:
        return intersections[indices[0]]
    return None


def get_next_intersection(p__, b__, boundaries):
    """Get the next intersection from the intersection of arcs *p__* and *b__* along segment *b__* with *boundaries*."""
    new_b = Arc(p__, b__.end)
    intersections, bounds = get_intersections(new_b, boundaries)
    dists = np.array([b__.start.distance(p2) for p2 in intersections])
    indices = dists.argsort()
    if len(intersections) > 0 and intersections[indices[0]] != p__:
        return intersections[indices[0]], bounds[indices[0]]
    elif len(intersections) > 1:
        return intersections[indices[1]], bounds[indices[1]]
    return None, None


def point_inside(point, corners):
    """Determine if points are inside 4 corner points.

    This uses great circle arcs as area boundaries.
    """
    arc1 = Arc(corners[0], corners[1])
    arc2 = Arc(corners[1], corners[2])
    arc3 = Arc(corners[2], corners[3])
    arc4 = Arc(corners[3], corners[0])

    arc5 = Arc(corners[1], point)
    arc6 = Arc(corners[3], point)

    angle1 = modpi(arc1.angle(arc2))
    angle1bis = modpi(arc1.angle(arc5))

    angle2 = modpi(arc3.angle(arc4))
    angle2bis = modpi(arc3.angle(arc6))

    return (np.sign(angle1) == np.sign(angle1bis) and
            abs(angle1) > abs(angle1bis) and
            np.sign(angle2) == np.sign(angle2bis) and
            abs(angle2) > abs(angle2bis))


def intersection_polygon(area_corners, segment_corners):
    """Get the intersection polygon between two areas."""
    area_boundaries = [Arc(area_corners[0], area_corners[1]),
                       Arc(area_corners[1], area_corners[2]),
                       Arc(area_corners[2], area_corners[3]),
                       Arc(area_corners[3], area_corners[0])]
    segment_boundaries = [Arc(segment_corners[0], segment_corners[1]),
                          Arc(segment_corners[1], segment_corners[2]),
                          Arc(segment_corners[2], segment_corners[3]),
                          Arc(segment_corners[3], segment_corners[0])]

    angle1 = area_boundaries[0].angle(area_boundaries[1])
    angle2 = segment_boundaries[0].angle(segment_boundaries[1])
    if np.sign(angle1) != np.sign(angle2):
        segment_corners.reverse()
        segment_boundaries = [Arc(segment_corners[0], segment_corners[1]),
                              Arc(segment_corners[1], segment_corners[2]),
                              Arc(segment_corners[2], segment_corners[3]),
                              Arc(segment_corners[3], segment_corners[0])]
    poly = []

    boundaries = area_boundaries
    other_boundaries = segment_boundaries

    b__ = None

    for b__ in boundaries:
        if point_inside(b__.start, segment_corners):
            poly.append(b__.start)
            break
        else:
            inter = get_first_intersection(b__, other_boundaries)
            if inter is not None:
                poly.append(inter)
                break
    if len(poly) == 0:
        return None
    while len(poly) < 2 or poly[0] != poly[-1]:
        inter, b2_ = get_next_intersection(poly[-1], b__, other_boundaries)
        if inter is None:
            poly.append(b__.end)
            idx = (boundaries.index(b__) + 1) % len(boundaries)
            b__ = boundaries[idx]
        else:
            poly.append(inter)
            b__ = b2_
            boundaries, other_boundaries = other_boundaries, boundaries
    return poly[:-1]
