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
"""Define a single great-circle arc on the sphere through the SArc class."""
import numpy as np

from pyresample.future.spherical import EPSILON
from pyresample.spherical import SCoordinate, _unwrap_radians


class Arc(object):
    """An arc of the great circle between two points."""

    def __init__(self, start, end):
        self.start, self.end = start, end

    def __eq__(self, other):
        """Check equality."""
        if self.start == other.start and self.end == other.end:
            return 1
        return 0

    def __ne__(self, other):
        """Check not equal comparison."""
        return not self.__eq__(other)

    def __str__(self):
        """Get simplified representation."""
        return str(self.start) + " -> " + str(self.end)

    def __repr__(self):
        """Get simplified representation."""
        return str(self.start) + " -> " + str(self.end)

    def angle(self, other_arc):
        """Oriented angle between two arcs.

        Returns:
            Angle in radians. A straight line will be 0. A clockwise path
            will be a negative angle and counter-clockwise will be positive.

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

        ua_ = a__.cross2cart(b__)
        ub_ = a__.cross2cart(c__)

        val = ua_.dot(ub_) / (ua_.norm() * ub_.norm())

        if abs(val - 1) < EPSILON:
            angle = 0
        elif abs(val + 1) < EPSILON:
            angle = np.pi
        else:
            angle = np.arccos(val)

        n__ = ua_.normalize()
        if n__.dot(c__.to_cart()) > 0:
            return -angle
        else:
            return angle

    def intersections(self, other_arc):
        """Give the two intersections of the greats circles defined by the current arc and *other_arc*.

        From http://williams.best.vwh.net/intersect.htm
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

        end_point = SCoordinate(end_lon, self.end.lat)
        other_end_point = SCoordinate(other_end_lon, other_arc.end.lat)

        ea_ = self.start.cross2cart(end_point).normalize()
        eb_ = other_arc.start.cross2cart(other_end_point).normalize()

        cross = ea_.cross(eb_)
        lat = np.arctan2(cross.cart[2],
                         np.sqrt(cross.cart[0] ** 2 + cross.cart[1] ** 2))
        lon = np.arctan2(cross.cart[1], cross.cart[0])

        return (SCoordinate(lon, lat),
                SCoordinate(_unwrap_radians(lon + np.pi), -lat))

    def intersects(self, other_arc):
        """Check if the current arc and the *other_arc* intersect.

        An arc is defined as the shortest tracks between two points.
        """
        return bool(self.intersection(other_arc))

    def intersection(self, other_arc):
        """Return where, if the current arc and the *other_arc* intersect.

        None is returned if there is not intersection. An arc is defined
        as the shortest tracks between two points.
        """
        if self == other_arc:
            return None

        for i in self.intersections(other_arc):
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
            d__ = other_arc.end

            ab_ = a__.hdistance(b__)
            cd_ = c__.hdistance(d__)

            if (((i in (a__, b__)) or
                (abs(a__.hdistance(i) + b__.hdistance(i) - ab_) < EPSILON)) and
                ((i in (c__, d__)) or
                 (abs(c__.hdistance(i) + d__.hdistance(i) - cd_) < EPSILON))):
                return i
        return None

    def get_next_intersection(self, arcs, known_inter=None):
        """Get the next intersection between the current arc and *arcs*."""
        res = []
        for arc in arcs:
            inter = self.intersection(arc)
            if (inter is not None and
                    inter != arc.end and
                    inter != self.end):
                res.append((inter, arc))

        def dist(args):
            """Get distance key."""
            return self.start.distance(args[0])

        take_next = False
        for inter, arc in sorted(res, key=dist):
            if known_inter is not None:
                if known_inter == inter:
                    take_next = True
                elif take_next:
                    return inter, arc
            else:
                return inter, arc

        return None, None
