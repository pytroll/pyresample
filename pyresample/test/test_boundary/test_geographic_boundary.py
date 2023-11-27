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
"""Test the GeographicBoundary objects."""

import numpy as np
import pytest

from pyresample import SwathDefinition
from pyresample.boundary import GeographicBoundary


class TestSwathGeographicBoundary():
    """Test 'GeographicBoundary' class for SwathDefinition."""

    def setup_method(self):
        sides_lons = [np.array([1.0, 1.5, 2.0]),
                      np.array([2.0, 3.0]),
                      np.array([3.0, 3.5, 4.0]),
                      np.array([4.0, 1.0])]
        sides_lats = [np.array([6.0, 6.5, 7.0]),
                      np.array([7.0, 8.0]),
                      np.array([8.0, 8.5, 9.0]),
                      np.array([9.0, 6.0])]
        lons = np.array([[1.0, 1.5, 2.0],
                         [4.0, 3.5, 3.0]])
        lats = np.array([[6.0, 6.5, 7.0],
                         [9.0, 8.5, 8.0]])

        self.lons = lons
        self.lats = lats
        self.area = SwathDefinition(lons, lats)
        self.sides_lons = sides_lons
        self.sides_lats = sides_lats
        self.point_inside = (2.5, 7.5)

    def test_creation(self):
        """Test GeographicBoundary creation."""
        # Define GeographicBoundary
        boundary = GeographicBoundary(self.area)

        # Assert sides coincides
        for b_lon, src_lon in zip(boundary.sides_lons, self.sides_lons):
            assert np.allclose(b_lon, src_lon)

        for b_lat, src_lat in zip(boundary.sides_lats, self.sides_lats):
            assert np.allclose(b_lat, src_lat)

    def test_minimum_swath_size(self):
        """Test swath has minimum 2 dimensions."""
        area = SwathDefinition(self.lons[0], self.lats[1])
        with pytest.raises(ValueError):
            GeographicBoundary(area)

    def test_vertices_property(self):
        """Test GeographicBoundary vertices property."""
        # Define GeographicBoundary
        boundary = GeographicBoundary(self.area)
        # Assert vertices
        expected_vertices = np.array([[1., 6.],
                                      [1.5, 6.5],
                                      [2., 7.],
                                      [3., 8.],
                                      [3.5, 8.5],
                                      [4., 9.]])
        assert np.allclose(boundary.vertices, expected_vertices)

    def test_contour(self):
        """Test that GeographicBoundary.contour(closed=False) returns the correct (lon,lat) tuple."""
        # Define GeographicBoundary
        boundary = GeographicBoundary(self.area)
        # Assert contour
        lons, lats = boundary.contour()
        assert np.allclose(lons, np.array([1., 1.5, 2., 3., 3.5, 4.]))
        assert np.allclose(lats, np.array([6., 6.5, 7., 8., 8.5, 9.]))

    def test_contour_closed(self):
        """Test that GeographicBoundary.contour(closed=True) returns the correct (lon,lat) tuple."""
        # Define GeographicBoundary
        boundary = GeographicBoundary(self.area)
        lons, lats = boundary.contour(closed=True)
        assert np.allclose(lons, np.array([1., 1.5, 2., 3., 3.5, 4., 1.]))
        assert np.allclose(lats, np.array([6., 6.5, 7., 8., 8.5, 9., 6.]))
