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
"""Define single and multiple points on the sphere through SPoint and SMultiPoint classes."""
import numpy as np

from pyresample.spherical import SCoordinate


class SPoint(SCoordinate):
    """Object representing a single point on a sphere.

    The ``lon`` and ``lat`` coordinates must be provided in radians.
    """

    def __init__(self, lon, lat):
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        if lon.size > 1 or lat.size > 1:
            raise ValueError("Use SMultiPoint to define multiple points.")
        super().__init__(lon, lat)

    @classmethod
    def from_degrees(cls, lon, lat):
        """Create SPoint from lon/lat coordinates in degrees."""
        return cls(np.deg2rad(lon), np.deg2rad(lat))

    def __str__(self):
        """Get simplified representation of lon/lat arrays in radians."""
        return str((float(self.lon), float(self.lat)))

    def __repr__(self):
        """Get simplified representation of lon/lat arrays in radians."""
        return str((float(self.lon), float(self.lat)))

    def to_shapely(self):
        """Convert the SPoint to a shapely Point (in lon/lat degrees)."""
        from shapely.geometry import Point
        point = Point(*self.vertices_in_degrees[0])
        return point


class SMultiPoint(SCoordinate):
    """Object representing multiple points on a sphere."""

    def __init__(self, lon, lat):
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        if lon.ndim == 0 or lat.ndim == 0:
            raise ValueError("Use SPoint to define single points.")
        super().__init__(lon, lat)

    @classmethod
    def from_degrees(cls, lon, lat):
        """Create SMultiPoint from lon/lat coordinates in degrees."""
        return cls(np.deg2rad(lon), np.deg2rad(lat))

    def __eq__(self, other):
        """Check equality."""
        return np.allclose(self.lon, other.lon) and np.allclose(self.lat, other.lat)

    def __str__(self):
        """Get simplified representation of lon/lat arrays in radians."""
        return str(self.vertices)

    def __repr__(self):
        """Get simplified representation of lon/lat arrays in radians."""
        return str(self.vertices)

    def to_shapely(self):
        """Convert the SMultiPoint to a shapely MultiPoint (in lon/lat degrees)."""
        from shapely.geometry import MultiPoint
        point = MultiPoint(self.vertices_in_degrees)
        return point
