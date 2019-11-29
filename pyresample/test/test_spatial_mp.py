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


# try:
#     from unittest import mock
# except ImportError:
#     # separate mock package py<3.3
#     import mock
import unittest
import numpy as np

from pyresample._spatial_mp import Cartesian


# class SpatialMPTest(unittest.TestCase):
#     @mock.patch('pyresample._spatial_mp.pyproj.Proj.__init__', return_value=None)
#     def test_base_proj_epsg(self, proj_init):
#         """Test Proj creation with EPSG codes"""
#         if pyproj.__version__ < '2':
#             return self.skipTest(reason='pyproj 2+ only')
#
#         args = [
#             [None, {'init': 'EPSG:6932'}],
#             [{'init': 'EPSG:6932'}, {}],
#             [None, {'EPSG': '6932'}],
#             [{'EPSG': '6932'}, {}]
#         ]
#         for projparams, kwargs in args:
#             BaseProj(projparams, **kwargs)
#             proj_init.assert_called_with(projparams='EPSG:6932', preserve_units=mock.ANY)
#             proj_init.reset_mock()

class SpatialMPTest(unittest.TestCase):
    """Test of spatial_mp."""

    def test_cartesian(self):
        """Test the transform_lonlats of class Cartesian."""
        exp_coords = np.array([[6370997., 0, 0],
                               [6178887.9339746, 1089504.6535337, 1106312.0189715],
                               [5233097.4664751, 2440233.4244888, 2692499.6776952]])
        lon = np.array([0, 10, 25])
        lat = np.array([0, 10, 25])
        my_cartesian = Cartesian()
        coords_int = my_cartesian.transform_lonlats(lon, lat)
        coords_float = my_cartesian.transform_lonlats(lon.astype(np.float64), lat.astype(np.float64))

        np.testing.assert_almost_equal(coords_float, exp_coords, decimal=3)
        np.testing.assert_almost_equal(coords_int, coords_float, decimal=0)


def suite():
    """Test suite for _spatial_mp."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(SpatialMPTest))
    return mysuite


if __name__ == '__main__':
    unittest.main()
