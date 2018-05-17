# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2018  Pytroll Developers
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
"""Testing the data_reduce module."""

import unittest
import numpy as np
from pyresample import geometry
from pyresample.data_reduce import (get_valid_index_from_cartesian_grid,
                                    swath_from_lonlat_grid,
                                    swath_from_lonlat_boundaries,
                                    swath_from_cartesian_grid,
                                    get_valid_index_from_lonlat_grid)


class Test(unittest.TestCase):

    """Unit testing the data_reduce module."""

    @classmethod
    def setUpClass(cls):
        """Get ready for testing."""
        cls.area_def = geometry.AreaDefinition('areaD',
                                               'Europe (3km, HRV, VTC)',
                                               'areaD',
                                               {'a': '6378144.0',
                                                'b': '6356759.0',
                                                'lat_0': '50.00',
                                                'lat_ts': '50.00',
                                                'lon_0': '8.00',
                                                'proj': 'stere'},
                                               800,
                                               800,
                                               [-1370912.72,
                                                   -909968.64000000001,
                                                   1029087.28,
                                                   1490031.3600000001])

    def test_reduce(self):
        data = np.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = np.fromfunction(
            lambda y, x: -180 + (360.0 / 1000) * x, (1000, 1000))
        lats = np.fromfunction(
            lambda y, x: -90 + (180.0 / 1000) * y, (1000, 1000))
        grid_lons, grid_lats = self.area_def.get_lonlats()
        lons, lats, data = swath_from_lonlat_grid(grid_lons, grid_lats,
                                                  lons, lats, data,
                                                  7000)
        cross_sum = data.sum()
        expected = 20685125.0
        self.assertAlmostEqual(cross_sum, expected)

    def test_reduce_boundary(self):
        data = np.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = np.fromfunction(
            lambda y, x: -180 + (360.0 / 1000) * x, (1000, 1000))
        lats = np.fromfunction(
            lambda y, x: -90 + (180.0 / 1000) * y, (1000, 1000))
        boundary_lonlats = self.area_def.get_boundary_lonlats()
        lons, lats, data = swath_from_lonlat_boundaries(boundary_lonlats[0],
                                                        boundary_lonlats[1],
                                                        lons, lats, data, 7000)
        cross_sum = data.sum()
        expected = 20685125.0
        self.assertAlmostEqual(cross_sum, expected)

    def test_cartesian_reduce(self):
        data = np.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = np.fromfunction(
            lambda y, x: -180 + (360.0 / 1000) * x, (1000, 1000))
        lats = np.fromfunction(
            lambda y, x: -90 + (180.0 / 1000) * y, (1000, 1000))
        grid = self.area_def.get_cartesian_coords()
        lons, lats, data = swath_from_cartesian_grid(grid, lons, lats, data,
                                                     7000)
        cross_sum = data.sum()
        expected = 20685125.0
        self.assertAlmostEqual(cross_sum, expected)

    def test_area_con_reduce(self):
        data = np.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = np.fromfunction(
            lambda y, x: -180 + (360.0 / 1000) * x, (1000, 1000))
        lats = np.fromfunction(
            lambda y, x: -90 + (180.0 / 1000) * y, (1000, 1000))
        grid_lons, grid_lats = self.area_def.get_lonlats()
        valid_index = get_valid_index_from_lonlat_grid(grid_lons, grid_lats,
                                                       lons, lats, 7000)
        data = data[valid_index]
        cross_sum = data.sum()
        expected = 20685125.0
        self.assertAlmostEqual(cross_sum, expected)

    def test_area_con_cartesian_reduce(self):
        data = np.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = np.fromfunction(
            lambda y, x: -180 + (360.0 / 1000) * x, (1000, 1000))
        lats = np.fromfunction(
            lambda y, x: -90 + (180.0 / 1000) * y, (1000, 1000))
        cart_grid = self.area_def.get_cartesian_coords()
        valid_index = get_valid_index_from_cartesian_grid(cart_grid,
                                                          lons, lats, 7000)
        data = data[valid_index]
        cross_sum = data.sum()
        expected = 20685125.0
        self.assertAlmostEqual(cross_sum, expected)

    def test_reduce_north_pole(self):
        """Test reducing around the poles."""

        from pyresample import utils
        area_id = 'ease_sh'
        description = 'Antarctic EASE grid'
        proj_id = 'ease_sh'
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        x_size = 425
        y_size = 425
        area_extent = (-5326849.0625, -5326849.0625,
                       5326849.0625, 5326849.0625)
        area_def = utils.get_area_def(area_id, description, proj_id,
                                      projection, x_size, y_size, area_extent)

        grid_lons, grid_lats = area_def.get_lonlats()

        area_id = 'ease_sh'
        description = 'Antarctic EASE grid'
        proj_id = 'ease_sh'
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        x_size = 1000
        y_size = 1000
        area_extent = (-532684.0625, -532684.0625, 532684.0625, 532684.0625)
        smaller_area_def = utils.get_area_def(area_id, description, proj_id,
                                              projection, x_size, y_size,
                                              area_extent)

        data = np.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons, lats = smaller_area_def.get_lonlats()

        lons, lats, data = swath_from_lonlat_grid(grid_lons, grid_lats,
                                                  lons, lats, data, 7000)

        cross_sum = data.sum()
        expected = 999000000.0
        self.assertAlmostEqual(cross_sum, expected)


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite


if __name__ == '__main__':
    unittest.main()
