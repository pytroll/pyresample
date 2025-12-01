#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2022 Pyresample developers
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
"""Test the boundary objects."""
import unittest

import numpy as np
import pytest

from pyresample.boundary import AreaBoundary


class TestAreaBoundary(unittest.TestCase):
    """Test 'AreaBoundary' class."""

    def test_creation_from_lonlat_sides(self):
        """Test AreaBoundary creation from sides."""
        lon_sides = [np.array([1.0, 1.5, 2.0]),
                     np.array([2.0, 3.0]),
                     np.array([3.0, 3.5, 4.0]),
                     np.array([4.0, 1.0])]
        lat_sides = [np.array([6.0, 6.5, 7.0]),
                     np.array([7.0, 8.0]),
                     np.array([8.0, 8.5, 9.0]),
                     np.array([9.0, 6.0])]
        # Define AreaBoundary
        boundary = AreaBoundary.from_lonlat_sides(lon_sides, lat_sides)

        # Assert sides coincides
        for b_lon, src_lon in zip(boundary.sides_lons, lon_sides, strict=True):
            assert np.allclose(b_lon, src_lon)

        for b_lat, src_lat in zip(boundary.sides_lats, lat_sides, strict=True):
            assert np.allclose(b_lat, src_lat)

    def test_creation(self):
        """Test AreaBoundary creation."""
        list_sides = [(np.array([1., 1.5, 2.]), np.array([6., 6.5, 7.])),
                      (np.array([2., 3.]), np.array([7., 8.])),
                      (np.array([3., 3.5, 4.]), np.array([8., 8.5, 9.])),
                      (np.array([4., 1.]), np.array([9., 6.]))]
        lon_sides = [side[0]for side in list_sides]
        lat_sides = [side[1]for side in list_sides]

        # Define AreaBoundary
        boundary = AreaBoundary(*list_sides)

        # Assert sides coincides
        for b_lon, src_lon in zip(boundary.sides_lons, lon_sides, strict=True):
            assert np.allclose(b_lon, src_lon)

        for b_lat, src_lat in zip(boundary.sides_lats, lat_sides, strict=True):
            assert np.allclose(b_lat, src_lat)

    def test_number_sides_required(self):
        """Test AreaBoundary requires 4 sides ."""
        list_sides = [(np.array([1., 1.5, 2.]), np.array([6., 6.5, 7.])),
                      (np.array([2., 3.]), np.array([7., 8.])),
                      (np.array([3., 3.5, 4.]), np.array([8., 8.5, 9.])),
                      (np.array([4., 1.]), np.array([9., 6.]))]
        with pytest.raises(ValueError):
            AreaBoundary(*list_sides[0:3])

    def test_vertices_property(self):
        """Test AreaBoundary vertices property."""
        lon_sides = [np.array([1.0, 1.5, 2.0]),
                     np.array([2.0, 3.0]),
                     np.array([3.0, 3.5, 4.0]),
                     np.array([4.0, 1.0])]
        lat_sides = [np.array([6.0, 6.5, 7.0]),
                     np.array([7.0, 8.0]),
                     np.array([8.0, 8.5, 9.0]),
                     np.array([9.0, 6.0])]
        # Define AreaBoundary
        boundary = AreaBoundary.from_lonlat_sides(lon_sides, lat_sides)

        # Assert vertices
        expected_vertices = np.array([[1., 6.],
                                      [1.5, 6.5],
                                      [2., 7.],
                                      [3., 8.],
                                      [3.5, 8.5],
                                      [4., 9.]])
        assert np.allclose(boundary.vertices, expected_vertices)

    def test_contour(self):
        """Test that AreaBoundary.contour returns the correct (lon,lat) tuple."""
        list_sides = [(np.array([1., 1.5, 2.]), np.array([6., 6.5, 7.])),
                      (np.array([2., 3.]), np.array([7., 8.])),
                      (np.array([3., 3.5, 4.]), np.array([8., 8.5, 9.])),
                      (np.array([4., 1.]), np.array([9., 6.]))]
        boundary = AreaBoundary(*list_sides)
        lons, lats = boundary.contour()
        assert np.allclose(lons, np.array([1., 1.5, 2., 3., 3.5, 4.]))
        assert np.allclose(lats, np.array([6., 6.5, 7., 8., 8.5, 9.]))

    def test_countour_poly_north_pole(self):
        """Test that a polygon can be made with 32-bit north pole coordinates.

        In a real world case, 32-bit geolocation including a point at exactly the
        north pole lead to a validity check failure because the radians value for
        90N was just barely above pi / 2.

        """
        list_sides = [
            (np.array([1., 1.5, 2.], dtype=np.float32), np.array([87.0, 87.5, 88.0], dtype=np.float32)),
            (np.array([2., 3.], dtype=np.float32), np.array([88.0, 89.0], dtype=np.float32)),
            (np.array([3., 3.5, 4.], dtype=np.float32), np.array([89., 89.5, 90.0], dtype=np.float32)),
            (np.array([4., 1.], dtype=np.float32), np.array([90.0, 87.0], dtype=np.float32)),
        ]
        boundary = AreaBoundary(*list_sides)
        poly = boundary.contour_poly
        # validity checks happen here:
        aedges = list(poly.aedges())
        # ensure 90 degree latitude coordinate is retained
        np.testing.assert_allclose(aedges[4].end.lat, np.radians(90.0))
        np.testing.assert_allclose(aedges[5].start.lat, np.radians(90.0))
