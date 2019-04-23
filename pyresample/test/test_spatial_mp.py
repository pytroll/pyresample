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
"""Testing the _spatial_mp module"""


try:
    from unittest import mock
except ImportError:
    # separate mock package py<3.3
    import mock
import pyproj
import unittest

from pyresample._spatial_mp import BaseProj


class SpatialMPTest(unittest.TestCase):
    @mock.patch('pyresample._spatial_mp.pyproj.Proj.__init__', return_value=None)
    def test_base_proj_epsg(self, proj_init):
        """Test Proj creation with EPSG codes"""
        if pyproj.__version__ < '2':
            return self.skipTest(reason='pyproj 2+ only')

        args = [
            [None, {'init': 'EPSG:6932'}],
            [{'init': 'EPSG:6932'}, {}],
            [None, {'EPSG': '6932'}],
            [{'EPSG': '6932'}, {}]
        ]
        for projparams, kwargs in args:
            BaseProj(projparams, **kwargs)
            proj_init.assert_called_with(projparams='EPSG:6932', preserve_units=mock.ANY)
            proj_init.reset_mock()


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(SpatialMPTest))
    return mysuite


if __name__ == '__main__':
    unittest.main()
