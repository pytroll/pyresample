#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2014-2019 PyTroll developers
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
"""Testing the _spatial_mp module."""

import unittest

import numpy as np

import pyresample._spatial_mp as sp


class SpatialMPTest(unittest.TestCase):
    """Test of spatial_mp."""

    def setUp(self):
        """Set up some variables."""
        self.exp_coords = np.array([[6370997., 0, 0],
                                    [6178887.9339746, 1089504.6535337, 1106312.0189715],
                                    [5233097.4664751, 2440233.4244888, 2692499.6776952]])
        self.lon = np.array([0, 10, 25])
        self.lat = np.array([0, 10, 25])
        self.lon32 = self.lon.astype(np.float32)
        self.lat32 = self.lon.astype(np.float32)
        self.lon64 = self.lon.astype(np.float64)
        self.lat64 = self.lon.astype(np.float64)

    def test_cartesian(self):
        """Test the transform_lonlats of class Cartesian."""
        my_cartesian = sp.Cartesian()
        coords_int = my_cartesian.transform_lonlats(self.lon, self.lat)
        coords_float = my_cartesian.transform_lonlats(self.lon64, self.lat64)
        coords_float32 = my_cartesian.transform_lonlats(self.lon32, self.lat32)
        np.testing.assert_almost_equal(coords_float, self.exp_coords, decimal=3)
        np.testing.assert_almost_equal(coords_int, coords_float, decimal=0)
        self.assertIs(type(coords_float32[0, 0]), np.float32)
        self.assertIs(type(coords_float[0, 0]), np.float64)
        self.assertTrue(np.issubdtype(coords_int.dtype, np.floating))

    def test_cartesian_without_ne(self):
        """Test the transform_lonlats of class Cartesian without numexpr."""
        sp.ne = False
        my_cartesian = sp.Cartesian()
        coords_int = my_cartesian.transform_lonlats(self.lon, self.lat)
        coords_float = my_cartesian.transform_lonlats(self.lon64, self.lat64)
        coords_float32 = my_cartesian.transform_lonlats(self.lon32, self.lat32)
        np.testing.assert_almost_equal(coords_float, self.exp_coords, decimal=3)
        np.testing.assert_almost_equal(coords_int, coords_float, decimal=0)
        self.assertIs(type(coords_float32[0, 0]), np.float32)
        self.assertIs(type(coords_float[0, 0]), np.float64)
        self.assertTrue(np.issubdtype(coords_int.dtype, np.floating))
