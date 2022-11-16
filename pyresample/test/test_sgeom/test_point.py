#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2013-2022 Pyresample Developers
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
"""Test cases for SPoint and SMultiPoint."""
import unittest

import numpy as np
import pytest

from pyresample.future.spherical import SMultiPoint, SPoint


class TestSPoint(unittest.TestCase):
    """Test SPoint."""

    def test_latitude_validity(self):
        """Check SPoint raises error if providing bad latitude."""
        # Test latitude outside range
        lon = 0
        lat = np.pi
        with pytest.raises(ValueError):
            SPoint(lon, lat)
        # Test inf
        lon = 0
        lat = np.inf
        with pytest.raises(ValueError):
            SPoint(lon, lat)

    def test_longitude_validity(self):
        """Check SPoint raises error if providing bad longitude."""
        # Test inf
        lon = np.inf
        lat = 0
        with pytest.raises(ValueError):
            SPoint(lon, lat)

    def test_creation_from_degrees(self):
        """Check SPoint creation from lat/lon in degrees."""
        lon = 0
        lat = 20
        p1 = SPoint.from_degrees(lon, lat)
        p2 = SPoint(np.deg2rad(lon), np.deg2rad(lat))
        assert p1 == p2

    def test_vertices(self):
        """Test vertices property."""
        lons = 0
        lats = np.pi / 2
        p = SPoint(lons, lats)
        res = np.array([[0., 1.57079633]])
        assert np.allclose(p.vertices, res)

    def test_vertices_in_degrees(self):
        """Test vertices_in_degrees property."""
        lons = 0
        lats = np.pi / 2
        p = SPoint(lons, lats)
        res = np.array([[0., 90.]])
        assert np.allclose(p.vertices_in_degrees, res)

    def test_raise_error_if_multi_point(self):
        """Check SPoint raises error providing multiple points."""
        lons = np.array([0, np.pi])
        lats = np.array([-np.pi / 2, np.pi / 2])
        with pytest.raises(ValueError):
            SPoint(lons, lats)

    def test_str(self):
        """Check the string representation."""
        d = SPoint(1.0, 0.5)
        self.assertEqual(str(d), "(1.0, 0.5)")

    def test_repr(self):
        """Check the representation."""
        d = SPoint(1.0, 0.5)
        self.assertEqual(repr(d), "(1.0, 0.5)")

    def test_to_shapely(self):
        """Test conversion to shapely."""
        from shapely.geometry import Point
        lon = 0.0
        lat = np.pi / 2
        spherical_point = SPoint(lon, lat)
        shapely_point = Point(0.0, 90.0)
        assert shapely_point.equals_exact(spherical_point.to_shapely(), tolerance=1e-10)


class TestSMultiPoint(unittest.TestCase):
    """Test SMultiPoint."""

    def test_single_point(self):
        """Test behaviour when providing single lon,lat values."""
        # Single values must raise error
        with pytest.raises(ValueError):
            SMultiPoint(2, 1)
        # Array values must not raise error
        p = SMultiPoint([2], [1])
        assert p.lon.shape == (1,)
        assert p.lat.shape == (1,)
        assert p.vertices.shape == (1, 2)

    def test_creation_from_degrees(self):
        """Check SMultiPoint creation from lat/lon in degrees."""
        lon = np.array([0, 10])
        lat = np.array([20, 30])
        p1 = SMultiPoint.from_degrees(lon, lat)
        p2 = SMultiPoint(np.deg2rad(lon), np.deg2rad(lat))
        assert p1 == p2

    def test_vertices(self):
        """Test vertices property."""
        lons = np.array([0, np.pi])
        lats = np.array([-np.pi / 2, np.pi / 2])
        p = SMultiPoint(lons, lats)
        res = np.array([[0., -1.57079633],
                        [-3.14159265, 1.57079633]])
        assert np.allclose(p.vertices, res)

    def test_vertices_in_degrees(self):
        """Test vertices_in_degrees property."""
        lons = np.array([0, np.pi])
        lats = np.array([-np.pi / 2, np.pi / 2])
        p = SMultiPoint(lons, lats)
        res = np.array([[0., -90.],
                        [-180., 90.]])
        assert np.allclose(p.vertices_in_degrees, res)

    def test_distance(self):
        """Test Vincenty formula."""
        lons = np.array([0, np.pi])
        lats = np.array([-np.pi / 2, np.pi / 2])
        p1 = SMultiPoint(lons, lats)
        lons = np.array([0, np.pi / 2, np.pi])
        lats = np.array([-np.pi / 2, 0, np.pi / 2])
        p2 = SMultiPoint(lons, lats)
        d12 = p1.distance(p2)
        d21 = p2.distance(p1)
        self.assertEqual(d12.shape, (2, 3))
        self.assertEqual(d21.shape, (3, 2))
        res = np.array([[0., 1.57079633, 3.14159265],
                        [3.14159265, 1.57079633, 0.]])
        assert np.allclose(d12, res)
        # Special case with 1 point
        p1 = SMultiPoint(lons[[0]], lats[[0]])
        p2 = SMultiPoint(lons[[0]], lats[[0]])
        d12 = p1.distance(p2)
        assert isinstance(d12, float)

    def test_hdistance(self):
        """Test Haversine formula."""
        lons = np.array([0, np.pi])
        lats = np.array([-np.pi / 2, np.pi / 2])
        p1 = SMultiPoint(lons, lats)
        lons = np.array([0, np.pi / 2, np.pi])
        lats = np.array([-np.pi / 2, 0, np.pi / 2])
        p2 = SMultiPoint(lons, lats)
        d12 = p1.hdistance(p2)
        d21 = p2.hdistance(p1)
        self.assertEqual(d12.shape, (2, 3))
        self.assertEqual(d21.shape, (3, 2))
        res = np.array([[0., 1.57079633, 3.14159265],
                        [3.14159265, 1.57079633, 0.]])
        assert np.allclose(d12, res)

    def test_eq(self):
        """Check the equality."""
        lons = [0, np.pi]
        lats = [-np.pi / 2, np.pi / 2]
        p = SMultiPoint(lons, lats)
        p1 = SMultiPoint(lons, lats)
        assert p == p1

    def test_eq_antimeridian(self):
        """Check the equality with longitudes at -180/180 degrees."""
        lons = [np.pi, np.pi]
        lons1 = [-np.pi, -np.pi]
        lats = [-np.pi / 2, np.pi / 2]
        p = SMultiPoint(lons, lats)
        p1 = SMultiPoint(lons1, lats)
        assert p == p1

    def test_neq(self):
        """Check the equality."""
        lons = np.array([0, np.pi])
        lats = [-np.pi / 2, np.pi / 2]
        p = SMultiPoint(lons, lats)
        p1 = SMultiPoint(lons + 0.1, lats)
        assert p != p1

    def test_str(self):
        """Check the string representation."""
        lons = [0, np.pi]
        lats = [-np.pi / 2, np.pi / 2]
        p = SMultiPoint(lons, lats)
        expected_str = '[[ 0.         -1.57079633]\n [-3.14159265  1.57079633]]'
        self.assertEqual(str(p), expected_str)

    def test_repr(self):
        """Check the representation."""
        lons = [0, np.pi]
        lats = [-np.pi / 2, np.pi / 2]
        p = SMultiPoint(lons, lats)
        expected_repr = '[[ 0.         -1.57079633]\n [-3.14159265  1.57079633]]'
        self.assertEqual(repr(p), expected_repr)

    def test_to_shapely(self):
        """Test conversion to shapely."""
        from shapely.geometry import MultiPoint
        lons = np.array([0.0, np.pi])
        lats = np.array([-np.pi / 2, np.pi / 2])
        spherical_multipoint = SMultiPoint(lons, lats)
        shapely_multipoint = MultiPoint([(0.0, -90.0), (-180.0, 90.0)])
        assert shapely_multipoint.equals_exact(spherical_multipoint.to_shapely(), tolerance=1e-10)
