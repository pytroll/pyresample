#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test the EWA ll2cr code.
"""
import sys
import logging
import numpy as np
from pyresample.test.utils import create_test_longitude, create_test_latitude
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

LOG = logging.getLogger(__name__)


dynamic_wgs84 = {
    "grid_name": "test_wgs84_fit",
    "origin_x": None,
    "origin_y": None,
    "width": None,
    "height": None,
    "cell_width": 0.0057,
    "cell_height": -0.0057,
    "proj4_definition": "+proj=latlong +datum=WGS84 +ellps=WGS84 +no_defs",
}

static_lcc = {
    "grid_name": "test_lcc",
    "origin_x": -1950510.636800,
    "origin_y": 4368587.226913,
    "width": 5120,
    "height": 5120,
    "cell_width": 1015.9,
    "cell_height": -1015.9,
    "proj4_definition": "+proj=lcc +a=6371200 +b=6371200 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs",
}


class TestLL2CRStatic(unittest.TestCase):
    def test_lcc_basic1(self):
        from pyresample.ewa import _ll2cr
        lon_arr = create_test_longitude(-95.0, -75.0, (50, 100), dtype=np.float64)
        lat_arr = create_test_latitude(18.0, 40.0, (50, 100), dtype=np.float64)
        grid_info = static_lcc.copy()
        fill_in = np.nan
        proj_str = grid_info["proj4_definition"]
        cw = grid_info["cell_width"]
        ch = grid_info["cell_height"]
        ox = grid_info["origin_x"]
        oy = grid_info["origin_y"]
        w = grid_info["width"]
        h = grid_info["height"]
        points_in_grid = _ll2cr.ll2cr_static(lon_arr, lat_arr, fill_in, proj_str,
                                                               cw, ch, w, h, ox, oy)
        self.assertEqual(points_in_grid, lon_arr.size, "all these test points should fall in this grid")

    def test_lcc_fail1(self):
        from pyresample.ewa import _ll2cr
        lon_arr = create_test_longitude(-15.0, 15.0, (50, 100), dtype=np.float64)
        lat_arr = create_test_latitude(18.0, 40.0, (50, 100), dtype=np.float64)
        grid_info = static_lcc.copy()
        fill_in = np.nan
        proj_str = grid_info["proj4_definition"]
        cw = grid_info["cell_width"]
        ch = grid_info["cell_height"]
        ox = grid_info["origin_x"]
        oy = grid_info["origin_y"]
        w = grid_info["width"]
        h = grid_info["height"]
        points_in_grid = _ll2cr.ll2cr_static(lon_arr, lat_arr, fill_in, proj_str,
                                             cw, ch, w, h, ox, oy)
        self.assertEqual(points_in_grid, 0, "none of these test points should fall in this grid")


class TestLL2CRDynamic(unittest.TestCase):
    def test_latlong_basic1(self):
        from pyresample.ewa import _ll2cr
        lon_arr = create_test_longitude(-95.0, -75.0, (50, 100), dtype=np.float64)
        lat_arr = create_test_latitude(15.0, 30.0, (50, 100), dtype=np.float64)
        grid_info = dynamic_wgs84.copy()
        fill_in = np.nan
        proj_str = grid_info["proj4_definition"]
        cw = grid_info["cell_width"]
        ch = grid_info["cell_height"]
        ox = grid_info["origin_x"]
        oy = grid_info["origin_y"]
        w = grid_info["width"]
        h = grid_info["height"]
        points_in_grid, lon_res, lat_res, ox, oy, w, h = _ll2cr.ll2cr_dynamic(lon_arr, lat_arr, fill_in, proj_str,
                                                                              cw, ch, w, h, ox, oy)
        self.assertEqual(points_in_grid, lon_arr.size, "all points should be contained in a dynamic grid")
        self.assertIs(lon_arr, lon_res)
        self.assertIs(lat_arr, lat_res)
        self.assertEqual(lon_arr[0, 0], 0, "ll2cr returned the wrong result for a dynamic latlong grid")
        self.assertEqual(lat_arr[-1, 0], 0, "ll2cr returned the wrong result for a dynamic latlong grid")

    def test_latlong_basic2(self):
        from pyresample.ewa import _ll2cr
        lon_arr = create_test_longitude(-95.0, -75.0, (50, 100), twist_factor=0.6, dtype=np.float64)
        lat_arr = create_test_latitude(15.0, 30.0, (50, 100), twist_factor=-0.1, dtype=np.float64)
        grid_info = dynamic_wgs84.copy()
        fill_in = np.nan
        proj_str = grid_info["proj4_definition"]
        cw = grid_info["cell_width"]
        ch = grid_info["cell_height"]
        ox = grid_info["origin_x"]
        oy = grid_info["origin_y"]
        w = grid_info["width"]
        h = grid_info["height"]
        points_in_grid, lon_res, lat_res, ox, oy, w, h = _ll2cr.ll2cr_dynamic(lon_arr, lat_arr, fill_in, proj_str,
                                                                              cw, ch, w, h, ox, oy)
        self.assertEqual(points_in_grid, lon_arr.size, "all points should be contained in a dynamic grid")
        self.assertIs(lon_arr, lon_res)
        self.assertIs(lat_arr, lat_res)
        self.assertEqual(lon_arr[0, 0], 0, "ll2cr returned the wrong result for a dynamic latlong grid")
        self.assertEqual(lat_arr[-1, 0], 0, "ll2cr returned the wrong result for a dynamic latlong grid")

    def test_latlong_dateline1(self):
        from pyresample.ewa import _ll2cr
        lon_arr = create_test_longitude(165.0, -165.0, (50, 100), twist_factor=0.6, dtype=np.float64)
        lat_arr = create_test_latitude(15.0, 30.0, (50, 100), twist_factor=-0.1, dtype=np.float64)
        grid_info = dynamic_wgs84.copy()
        fill_in = np.nan
        proj_str = grid_info["proj4_definition"]
        cw = grid_info["cell_width"]
        ch = grid_info["cell_height"]
        ox = grid_info["origin_x"]
        oy = grid_info["origin_y"]
        w = grid_info["width"]
        h = grid_info["height"]
        points_in_grid, lon_res, lat_res, ox, oy, w, h = _ll2cr.ll2cr_dynamic(lon_arr, lat_arr, fill_in, proj_str,
                                                                              cw, ch, w, h, ox, oy)
        self.assertEqual(points_in_grid, lon_arr.size, "all points should be contained in a dynamic grid")
        self.assertIs(lon_arr, lon_res)
        self.assertIs(lat_arr, lat_res)
        self.assertEqual(lon_arr[0, 0], 0, "ll2cr returned the wrong result for a dynamic latlong grid")
        self.assertEqual(lat_arr[-1, 0], 0, "ll2cr returned the wrong result for a dynamic latlong grid")
        self.assertTrue(np.all(np.diff(lon_arr[0]) >= 0), "ll2cr didn't return monotonic columns over the dateline")


class TestLL2CRWrapper(unittest.TestCase):
    def test_basic1(self):
        from pyresample.ewa import ll2cr
        from pyresample.geometry import SwathDefinition, AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        lon_arr = create_test_longitude(-95.0, -75.0, (50, 100), dtype=np.float64)
        lat_arr = create_test_latitude(18.0, 40.0, (50, 100), dtype=np.float64)
        swath_def = SwathDefinition(lon_arr, lat_arr)
        grid_info = static_lcc.copy()
        cw = grid_info["cell_width"]
        ch = grid_info["cell_height"]
        ox = grid_info["origin_x"]
        oy = grid_info["origin_y"]
        w = grid_info["width"]
        h = grid_info["height"]
        half_w = abs(cw / 2.)
        half_h = abs(ch / 2.)
        extents = [
            ox - half_w, oy - h * abs(ch) - half_h,
            ox + w * abs(cw) + half_w, oy + half_h
        ]
        area = AreaDefinition('test_area', 'test_area', 'test_area',
                              proj4_str_to_dict(grid_info['proj4_definition']),
                              w, h, extents)
        points_in_grid, lon_res, lat_res, = ll2cr(swath_def, area,
                                                  fill=np.nan, copy=False)
        self.assertEqual(points_in_grid, lon_arr.size, "all points should be contained in a dynamic grid")
        self.assertIs(lon_arr, lon_res)
        self.assertIs(lat_arr, lat_res)
        self.assertEqual(points_in_grid, lon_arr.size, "all these test points should fall in this grid")


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestLL2CRStatic))
    mysuite.addTest(loader.loadTestsFromTestCase(TestLL2CRDynamic))
    mysuite.addTest(loader.loadTestsFromTestCase(TestLL2CRWrapper))

    return mysuite


if __name__ == '__main__':
    unittest.main()
