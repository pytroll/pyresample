#!/usr/bin/env python
# encoding: utf8
#
# Copyright (C) 2014-2018 PyTroll developers
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

import unittest
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass  # Postpone fail to individual tests

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = None


class Test(unittest.TestCase):

    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'test_files', 'ssmis_swath.npz'))
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

    def test_ellps2axis(self):
        from pyresample import plot
        a, b = plot.ellps2axis('WGS84')
        self.assertAlmostEqual(a, 6378137.0,
                               msg='Failed to get semi-major axis of ellipsis')
        self.assertAlmostEqual(b, 6356752.3142451793,
                               msg='Failed to get semi-minor axis of ellipsis')

    @unittest.skipIf(Basemap is None, "basemap is not available")
    def test_area_def2basemap(self):
        from pyresample import plot
        from pyresample import parse_area_file
        area_def = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'ease_sh')[0]
        bmap = plot.area_def2basemap(area_def)
        self.assertTrue(bmap.rmajor == bmap.rminor and bmap.rmajor == 6371228.0,
                        'Failed to create Basemap object')

    def test_plate_carreeplot(self):
        from pyresample import plot, kd_tree, geometry
        from pyresample import parse_area_file
        area_def = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'pc_world')[0]
        swath_def = geometry.SwathDefinition(self.lons, self.lats)
        result = kd_tree.resample_nearest(swath_def, self.tb37v, area_def,
                                          radius_of_influence=20000,
                                          fill_value=None)
        plot._get_quicklook(area_def, result, num_meridians=0, num_parallels=0)

    def test_easeplot(self):
        from pyresample import plot, kd_tree, geometry
        from pyresample import parse_area_file
        area_def = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'ease_sh')[0]
        swath_def = geometry.SwathDefinition(self.lons, self.lats)
        result = kd_tree.resample_nearest(swath_def, self.tb37v, area_def,
                                          radius_of_influence=20000,
                                          fill_value=None)
        plot._get_quicklook(area_def, result)

    def test_orthoplot(self):
        from pyresample import plot, kd_tree, geometry
        from pyresample import parse_area_file
        area_def = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'ortho')[0]
        swath_def = geometry.SwathDefinition(self.lons, self.lats)
        result = kd_tree.resample_nearest(swath_def, self.tb37v, area_def,
                                          radius_of_influence=20000,
                                          fill_value=None)
        plot._get_quicklook(area_def, result)


def suite():
    """The test suite."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
