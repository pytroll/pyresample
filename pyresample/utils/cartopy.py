#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Pyresample developers
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
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Classes for geometry operations.

See the related Cartopy pull request that implements this functionality
directly in Cartopy: https://github.com/SciTools/cartopy/pull/1888

"""
from __future__ import annotations

import numpy as np
import pyproj

try:
    from xarray import DataArray
except ImportError:
    DataArray = np.ndarray

import cartopy.crs as ccrs
import shapely.geometry as sgeom


class Projection(ccrs.Projection):
    """Flexible generic Projection with optionally specified bounds."""

    def __init__(self,
                 crs: pyproj.CRS,
                 bounds: tuple[float, float, float, float] | None = None,
                 transform_bounds: bool = False):
        """Initialize CRS instance and compute bounds if possible."""
        # NOTE: Skip the base cartopy Projection class __init__
        super(ccrs.Projection, self).__init__(crs)
        if bounds is None and self.area_of_use is not None:
            bounds = (
                self.area_of_use.west,
                self.area_of_use.east,
                self.area_of_use.south,
                self.area_of_use.north,
            )
            transform_bounds = True

        self.bounds = bounds
        if bounds is not None and transform_bounds:
            # Convert lat/lon bounds to projected bounds.
            # Geographic area of the entire dataset referenced to WGS 84
            # NB. We can't use a polygon transform at this stage because
            # that relies on the existence of the map boundary... the very
            # thing we're trying to work out! ;-)
            x0, x1, y0, y1 = bounds
            lons = np.array([x0, x0, x1, x1])
            lats = np.array([y0, y1, y1, y0])
            points = self.transform_points(self.as_geodetic(), lons, lats)
            x = points[:, 0]
            y = points[:, 1]
            self.bounds = (x.min(), x.max(), y.min(), y.max())
        if self.bounds is not None:
            x0, x1, y0, y1 = self.bounds
            self.threshold = min(abs(x1 - x0), abs(y1 - y0)) / 100.

        # For geostationary full disc the boundary needs to be the actuall boundary of the earth disc otherwise
        # when ocean or land features are added to the cartopy plot these spill over.
        if "Geostationary Satellite" not in crs.to_wkt():
            self._boundary = super().boundary
        else:
            self._boundary = self._geos_boundary()

    @staticmethod
    def _geos_boundary():
        """Calculate full disk boundary.

        This code is copied over from the 'Geostationary' class in  'cartopy/lib/cartopy/crs.py'.
        """
        satellite_height = 35785831
        false_easting = 0
        false_northing = 0
        a = float(ccrs.WGS84_SEMIMAJOR_AXIS)
        b = float(ccrs.WGS84_SEMIMINOR_AXIS)
        h = float(satellite_height)

        # To find the bound we trace around where the line from the satellite
        # is tangent to the surface. This involves trigonometry on a sphere
        # centered at the satellite. The two scanning angles form two legs of
        # triangle on this sphere--the hypotenuse "c" (angle arc) is controlled
        # by distance from center to the edge of the ellipse being seen.

        # This is one of the angles in the spherical triangle and used to
        # rotate around and "scan" the boundary
        angleA = np.linspace(0, -2 * np.pi, 91)  # Clockwise boundary.

        # Convert the angle around center to the proper value to use in the
        # parametric form of an ellipse
        th = np.arctan(a / b * np.tan(angleA))

        # Given the position on the ellipse, what is the distance from center
        # to the ellipse--and thus the tangent point
        r = np.hypot(a * np.cos(th), b * np.sin(th))
        sat_dist = a + h

        # Using this distance, solve for sin and tan of c in the triangle that
        # includes the satellite, Earth center, and tangent point--we need to
        # figure out the location of this tangent point on the elliptical
        # cross-section through the Earth towards the satellite, where the
        # major axis is a and the minor is r. With the ellipse centered on the
        # Earth and the satellite on the y-axis (at y = a + h = sat_dist), the
        # equation for an ellipse and some calculus gives us the tangent point
        # (x0, y0) as:
        # y0 = a**2 / sat_dist
        # x0 = r * np.sqrt(1 - a**2 / sat_dist**2)
        # which gives:
        # sin_c = x0 / np.hypot(x0, sat_dist - y0)
        # tan_c = x0 / (sat_dist - y0)
        # A bit of algebra combines these to give directly:
        sin_c = r / np.sqrt(sat_dist ** 2 - a ** 2 + r ** 2)
        tan_c = r / np.sqrt(sat_dist ** 2 - a ** 2)

        # Using Napier's rules for right spherical triangles R2 and R6,
        # (See https://en.wikipedia.org/wiki/Spherical_trigonometry), we can
        # solve for arc angles b and a, which are our x and y scanning angles,
        # respectively.
        coords = np.vstack([np.arctan(np.cos(angleA) * tan_c),  # R6
                            np.arcsin(np.sin(angleA) * sin_c)])  # R2

        # Need to multiply scanning angles by satellite height to get to the
        # actual native coordinates for the projection.
        coords *= h
        coords += np.array([[false_easting], [false_northing]])
        return sgeom.LinearRing(coords.T)

    @property
    def boundary(self):
        """Return boundary."""
        return self._boundary
