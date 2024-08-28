#!/usr/bin/env python
# encoding: utf8
#
# Copyright (C) 2014-2020 PyTroll developers
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
"""Test the quicklook plotting functions."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

from pyresample.test.utils import TEST_FILES_PATH

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass  # Postpone fail to individual tests

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = None


MERIDIANS1 = np.array([-180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80,
                       -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30,
                       40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                       150, 160, 170, 180], dtype=np.int64)

PARALLELS1 = np.array([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30,
                       40, 50, 60, 70, 80, 90], dtype=np.int64)


class Test(unittest.TestCase):
    """Test the plot utilities."""

    filename = os.path.abspath(os.path.join(TEST_FILES_PATH, 'ssmis_swath.npz'))
    data = np.load(filename)['data']
    lons = data[:, 0].astype(np.float64)
    lats = data[:, 1].astype(np.float64)
    tb37v = data[:, 2].astype(np.float64)

    # screen out the fill values
    fvalue = -10000000000.0
    valid_fov = (lons != fvalue) * (lats != fvalue) * (tb37v != fvalue)
    lons = lons[valid_fov]
    lats = lats[valid_fov]
    tb37v = tb37v[valid_fov]

    def setUp(self):
        """Set up the services for the test functions."""
        pass

    def test_ellps2axis(self):
        """Test the ellps2axis function."""
        from pyresample import plot
        a, b = plot.ellps2axis('WGS84')
        self.assertAlmostEqual(a, 6378137.0,
                               msg='Failed to get semi-major axis of ellipsis')
        self.assertAlmostEqual(b, 6356752.3142451793,
                               msg='Failed to get semi-minor axis of ellipsis')

    @unittest.skipIf(Basemap is None, "basemap is not available")
    def test_area_def2basemap(self):
        """Test the area to Basemap object conversion function."""
        from pyresample import parse_area_file, plot
        area_def = parse_area_file(os.path.join(TEST_FILES_PATH, 'areas.yaml'), 'ease_sh')[0]
        bmap = plot.area_def2basemap(area_def)
        self.assertTrue(bmap.rmajor == bmap.rminor and bmap.rmajor == 6371228.0,
                        'Failed to create Basemap object')

    @mock.patch('matplotlib.ticker.FixedLocator')
    @mock.patch('matplotlib.pyplot.axes')
    def test_add_gridlines(self, axes, fx_locator):
        """Test the adding of gridlines to matplotlib plotting."""
        from pyresample.plot import _add_gridlines
        fx_locator.return_value = mock.MagicMock()
        axes.return_value = mock.MagicMock()

        retv = _add_gridlines(axes, None, None)
        fx_locator.assert_not_called()
        self.assertEqual(retv.xlines, False)
        self.assertEqual(retv.ylines, False)

        retv = _add_gridlines(axes, 10, None)
        fx_locator.assert_called_once()

        meridians = fx_locator.call_args_list[0][0][0]
        np.testing.assert_array_equal(meridians, MERIDIANS1)

        retv = _add_gridlines(axes, 10, 10)
        parallels = fx_locator.call_args_list[-1][0][0]
        np.testing.assert_array_equal(parallels, PARALLELS1)
        ncalls = fx_locator.call_count
        self.assertEqual(ncalls, 3)

        retv = _add_gridlines(axes, None, 10)
        ncalls = fx_locator.call_count
        self.assertEqual(ncalls, 4)

        retv = _add_gridlines(axes, 0, 0)
        ncalls = fx_locator.call_count
        self.assertEqual(ncalls, 4)

    def test_translate_coast_res(self):
        """Test the translation of coast resolution arguments from old basemap notation to cartopy."""
        from pyresample.plot import BASEMAP_NOT_CARTOPY, _translate_coast_resolution_to_cartopy

        with self.assertRaises(KeyError) as raises:
            if sys.version_info > (3,):
                self.assertEqual(raises.msg, None)
            retv, _ = _translate_coast_resolution_to_cartopy('200m')

        if BASEMAP_NOT_CARTOPY:
            retv, _ = _translate_coast_resolution_to_cartopy('c')
            self.assertEqual(retv, 'c')
            retv, _ = _translate_coast_resolution_to_cartopy('110m')
            self.assertEqual(retv, 'l')
            retv, _ = _translate_coast_resolution_to_cartopy('10m')
            self.assertEqual(retv, 'f')
            retv, _ = _translate_coast_resolution_to_cartopy('50m')
            self.assertEqual(retv, 'i')

        if not BASEMAP_NOT_CARTOPY:
            retv, _ = _translate_coast_resolution_to_cartopy('c')
            self.assertEqual(retv, '110m')
            retv, _ = _translate_coast_resolution_to_cartopy('l')
            self.assertEqual(retv, '110m')
            retv, _ = _translate_coast_resolution_to_cartopy('i')
            self.assertEqual(retv, '50m')
            retv, _ = _translate_coast_resolution_to_cartopy('h')
            self.assertEqual(retv, '10m')
            retv, _ = _translate_coast_resolution_to_cartopy('f')
            self.assertEqual(retv, '10m')
            retv, _ = _translate_coast_resolution_to_cartopy('110m')
            self.assertEqual(retv, '110m')
            retv, _ = _translate_coast_resolution_to_cartopy('10m')
            self.assertEqual(retv, '10m')

    def test_plate_carreeplot(self):
        """Test the Plate Caree plotting functionality."""
        from pyresample import geometry, kd_tree, parse_area_file, plot
        area_def = parse_area_file(os.path.join(TEST_FILES_PATH, 'areas.yaml'), 'pc_world')[0]
        swath_def = geometry.SwathDefinition(self.lons, self.lats)
        result = kd_tree.resample_nearest(swath_def, self.tb37v, area_def,
                                          radius_of_influence=20000,
                                          fill_value=None)

        plot._get_quicklook(area_def, result, num_meridians=0, num_parallels=0)
        plot._get_quicklook(area_def, result, num_meridians=10, num_parallels=10)
        plot._get_quicklook(area_def, result, num_meridians=None, num_parallels=None)

    def test_easeplot(self):
        """Test the plotting on the ease grid area."""
        from pyresample import geometry, kd_tree, parse_area_file, plot
        area_def = parse_area_file(os.path.join(TEST_FILES_PATH, 'areas.yaml'), 'ease_sh')[0]
        swath_def = geometry.SwathDefinition(self.lons, self.lats)
        result = kd_tree.resample_nearest(swath_def, self.tb37v, area_def,
                                          radius_of_influence=20000,
                                          fill_value=None)
        plot._get_quicklook(area_def, result)

    def test_orthoplot(self):
        """Test the ortho plotting."""
        from pyresample import geometry, kd_tree, parse_area_file, plot
        area_def = parse_area_file(os.path.join(TEST_FILES_PATH, 'areas.cfg'), 'ortho')[0]
        swath_def = geometry.SwathDefinition(self.lons, self.lats)
        result = kd_tree.resample_nearest(swath_def, self.tb37v, area_def,
                                          radius_of_influence=20000,
                                          fill_value=None)
        plot._get_quicklook(area_def, result)
