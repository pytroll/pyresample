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
"""Test the geometry objects."""
import random
import sys
import unittest
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Proj

from pyresample import geo_filter, geometry
from pyresample.geometry import (
    IncompatibleAreas,
    SwathDefinition,
    combine_area_extents_vertical,
    concatenate_area_defs,
)
from pyresample.test.utils import catch_warnings


class Test(unittest.TestCase):
    """Unit testing the geometry and geo_filter modules."""

    def test_base_type(self):
        """Test the base type."""
        lons1 = np.arange(-135., +135, 50.)
        lats = np.ones_like(lons1) * 70.

        # Test dtype is preserved without longitude wrapping
        basedef = geometry.BaseDefinition(lons1, lats)
        lons, _ = basedef.get_lonlats()
        self.assertEqual(lons.dtype, lons1.dtype,
                         "BaseDefinition did not maintain dtype of longitudes (in:%s out:%s)" %
                         (lons1.dtype, lons.dtype,))

        lons1_ints = lons1.astype('int')
        basedef = geometry.BaseDefinition(lons1_ints, lats)
        lons, _ = basedef.get_lonlats()
        self.assertEqual(lons.dtype, lons1_ints.dtype,
                         "BaseDefinition did not maintain dtype of longitudes (in:%s out:%s)" %
                         (lons1_ints.dtype, lons.dtype,))

        # Test dtype is preserved with automatic longitude wrapping
        lons2 = np.where(lons1 < 0, lons1 + 360, lons1)
        with catch_warnings():
            basedef = geometry.BaseDefinition(lons2, lats)

        lons, _ = basedef.get_lonlats()
        self.assertEqual(lons.dtype, lons2.dtype,
                         "BaseDefinition did not maintain dtype of longitudes (in:%s out:%s)" %
                         (lons2.dtype, lons.dtype,))

        lons2_ints = lons2.astype('int')
        with catch_warnings():
            basedef = geometry.BaseDefinition(lons2_ints, lats)

        lons, _ = basedef.get_lonlats()
        self.assertEqual(lons.dtype, lons2_ints.dtype,
                         "BaseDefinition did not maintain dtype of longitudes (in:%s out:%s)" %
                         (lons2_ints.dtype, lons.dtype,))

    def test_area_hash(self):
        """Test the area hash."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

        self.assertIsInstance(hash(area_def), int)

        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           {'a': '6378144.0',
                                            'b': '6356759.0',
                                            'lat_ts': '50.00',
                                            'lon_0': '8.00',
                                            'lat_0': '50.00',
                                            'proj': 'stere'},
                                           800,
                                           800,
                                           [-1370912.72,
                                               -909968.64000000001,
                                               1029087.28,
                                               1490031.3600000001])

        self.assertIsInstance(hash(area_def), int)

        area_def = geometry.AreaDefinition('New area', 'Europe', 'areaD',
                                           {'a': '6378144.0',
                                            'b': '6356759.0',
                                            'lat_ts': '50.00',
                                            'lon_0': '8.00',
                                            'lat_0': '50.00',
                                            'proj': 'stere'},
                                           800,
                                           800,
                                           [-1370912.72,
                                               -909968.64000000001,
                                               1029087.28,
                                               1490031.3600000001])

        self.assertIsInstance(hash(area_def), int)

    def test_get_array_hashable(self):
        """Test making the array hashable."""
        arr = np.array([1.2, 1.3, 1.4, 1.5])
        if sys.byteorder == 'little':
            # arr.view(np.uint8)
            reference = np.array([51, 51, 51, 51, 51, 51, 243,
                                  63, 205, 204, 204, 204, 204,
                                  204, 244, 63, 102, 102, 102, 102,
                                  102, 102, 246, 63, 0, 0,
                                  0, 0, 0, 0, 248, 63],
                                 dtype=np.uint8)
        else:
            # on le machines use arr.byteswap().view(np.uint8)
            reference = np.array([63, 243, 51, 51, 51, 51, 51,
                                  51, 63, 244, 204, 204, 204,
                                  204, 204, 205, 63, 246, 102, 102,
                                  102, 102, 102, 102, 63, 248,
                                  0, 0, 0, 0, 0, 0],
                                 dtype=np.uint8)

        np.testing.assert_allclose(reference,
                                   geometry.get_array_hashable(arr))

        try:
            import xarray as xr
        except ImportError:
            pass
        else:
            xrarr = xr.DataArray(arr)
            np.testing.assert_allclose(reference,
                                       geometry.get_array_hashable(arr))

            xrarr.attrs['hash'] = 42
            self.assertEqual(geometry.get_array_hashable(xrarr),
                             xrarr.attrs['hash'])

    def test_swath_hash(self):
        """Test swath hash."""
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)

        self.assertIsInstance(hash(swath_def), int)

    def test_swath_hash_dask(self):
        """Test hashing SwathDefinitions with dask arrays underneath."""
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        dalons = da.from_array(lons, chunks=1000)
        dalats = da.from_array(lats, chunks=1000)
        swath_def = geometry.SwathDefinition(dalons, dalats)
        self.assertIsInstance(hash(swath_def), int)

    def test_swath_hash_xarray(self):
        """Test hashing SwathDefinitions with DataArrays underneath."""
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        xrlons = xr.DataArray(lons)
        xrlats = xr.DataArray(lats)
        swath_def = geometry.SwathDefinition(xrlons, xrlats)
        self.assertIsInstance(hash(swath_def), int)

    def test_swath_hash_xarray_with_dask(self):
        """Test hashing SwathDefinitions with DataArrays:dask underneath."""
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        dalons = da.from_array(lons, chunks=1000)
        dalats = da.from_array(lats, chunks=1000)
        xrlons = xr.DataArray(dalons)
        xrlats = xr.DataArray(dalats)
        swath_def = geometry.SwathDefinition(xrlons, xrlats)
        self.assertIsInstance(hash(swath_def), int)

    def test_non_contiguous_swath_hash(self):
        """Test swath hash."""
        lons = np.array([[1.2, 1.3, 1.4, 1.5],
                         [1.2, 1.3, 1.4, 1.5]])
        lats = np.array([[65.9, 65.86, 65.82, 65.78],
                         [65.9, 65.86, 65.82, 65.78]])
        swath_def = geometry.SwathDefinition(lons, lats)
        swath_def_subset = swath_def[:, slice(0, 2)]
        self.assertIsInstance(hash(swath_def_subset), int)

    def test_area_equal(self):
        """Test areas equality."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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
        area_def2 = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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
        self.assertFalse(
            area_def != area_def2, 'area_defs are not equal as expected')

    def test_not_area_equal(self):
        """Test areas inequality."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

        msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
                                           'msg_full',
                                           {'a': '6378169.0',
                                            'b': '6356584.0',
                                            'h': '35785831.0',
                                            'lon_0': '0',
                                            'proj': 'geos'},
                                           3712,
                                           3712,
                                           [-5568742.4000000004,
                                               -5568742.4000000004,
                                               5568742.4000000004,
                                               5568742.4000000004]
                                           )
        self.assertFalse(
            area_def == msg_area, 'area_defs are not expected to be equal')
        self.assertFalse(
            area_def == "area", 'area_defs are not expected to be equal')

    def test_swath_equal_area(self):
        """Test equality swath area."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

        swath_def = geometry.SwathDefinition(*area_def.get_lonlats())

        self.assertFalse(
            swath_def != area_def, "swath_def and area_def should be equal")

        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

        self.assertFalse(
            area_def != swath_def, "swath_def and area_def should be equal")

    def test_swath_not_equal_area(self):
        """Test inequality swath area."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)

        self.assertFalse(
            swath_def == area_def, "swath_def and area_def should be different")

        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

        self.assertFalse(
            area_def == swath_def, "swath_def and area_def should be different")

    def test_grid_filter_valid(self):
        """Test valid grid filtering."""
        lons = np.array([-170, -30, 30, 170])
        lats = np.array([20, -40, 50, -80])
        swath_def = geometry.SwathDefinition(lons, lats)
        filter_area = geometry.AreaDefinition('test', 'test', 'test',
                                              {'proj': 'eqc', 'lon_0': 0.0,
                                                  'lat_0': 0.0},
                                              8, 8,
                                              (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           ])
        grid_filter = geo_filter.GridFilter(filter_area, filter)
        valid_index = grid_filter.get_valid_index(swath_def)
        expected = np.array([1, 0, 0, 1])
        self.assertTrue(
            np.array_equal(valid_index, expected), 'Failed to find grid filter')

    def test_grid_filter(self):
        """Test filtering a grid."""
        lons = np.array([-170, -30, 30, 170])
        lats = np.array([20, -40, 50, -80])
        swath_def = geometry.SwathDefinition(lons, lats)
        data = np.array([1, 2, 3, 4])
        filter_area = geometry.AreaDefinition('test', 'test', 'test',
                                              {'proj': 'eqc', 'lon_0': 0.0,
                                                  'lat_0': 0.0},
                                              8, 8,
                                              (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           ])
        grid_filter = geo_filter.GridFilter(filter_area, filter)
        swath_def_f, data_f = grid_filter.filter(swath_def, data)
        expected = np.array([1, 4])
        self.assertTrue(
            np.array_equal(data_f, expected), 'Failed grid filtering data')
        expected_lons = np.array([-170, 170])
        expected_lats = np.array([20, -80])
        self.assertTrue(np.array_equal(swath_def_f.lons[:], expected_lons) and
                        np.array_equal(swath_def_f.lats[:], expected_lats),
                        'Failed finding grid filtering lon lats')

    def test_grid_filter2D(self):
        """Test filtering a 2D grid."""
        lons = np.array([[-170, -30, 30, 170],
                         [-170, -30, 30, 170]])
        lats = np.array([[20, -40, 50, -80],
                         [25, -35, 55, -75]])
        swath_def = geometry.SwathDefinition(lons, lats)
        data1 = np.ones((2, 4))
        data2 = np.ones((2, 4)) * 2
        data3 = np.ones((2, 4)) * 3
        data = np.dstack((data1, data2, data3))
        filter_area = geometry.AreaDefinition('test', 'test', 'test',
                                              {'proj': 'eqc', 'lon_0': 0.0,
                                                  'lat_0': 0.0},
                                              8, 8,
                                              (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           ])
        grid_filter = geo_filter.GridFilter(filter_area, filter, nprocs=2)
        swath_def_f, data_f = grid_filter.filter(swath_def, data)
        expected = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.assertTrue(
            np.array_equal(data_f, expected), 'Failed 2D grid filtering data')
        expected_lons = np.array([-170, 170, -170, 170])
        expected_lats = np.array([20, -80, 25, -75])
        self.assertTrue(np.array_equal(swath_def_f.lons[:], expected_lons) and
                        np.array_equal(swath_def_f.lats[:], expected_lats),
                        'Failed finding 2D grid filtering lon lats')

    def test_boundary(self):
        """Test getting the boundary."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           {'a': '6378144.0',
                                            'b': '6356759.0',
                                            'lat_0': '50.00',
                                            'lat_ts': '50.00',
                                            'lon_0': '8.00',
                                            'proj': 'stere'},
                                           10,
                                           10,
                                           [-1370912.72,
                                               -909968.64000000001,
                                               1029087.28,
                                               1490031.3600000001])
        proj_x_boundary, proj_y_boundary = area_def.projection_x_coords, area_def.projection_y_coords
        expected_x = np.array([-1250912.72, -1010912.72, -770912.72,
                               -530912.72, -290912.72, -50912.72, 189087.28,
                               429087.28, 669087.28, 909087.28])
        expected_y = np.array([1370031.36, 1130031.36, 890031.36, 650031.36,
                               410031.36, 170031.36, -69968.64, -309968.64,
                               -549968.64, -789968.64])
        self.assertTrue(np.allclose(proj_x_boundary, expected_x),
                        'Failed to find projection x coords')
        self.assertTrue(np.allclose(proj_y_boundary, expected_y),
                        'Failed to find projection y coords')

    def test_area_extent_ll(self):
        """Test getting the lower left area extent."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           {'a': '6378144.0',
                                            'b': '6356759.0',
                                            'lat_0': '50.00',
                                            'lat_ts': '50.00',
                                            'lon_0': '8.00',
                                            'proj': 'stere'},
                                           10,
                                           10,
                                           [-1370912.72,
                                               -909968.64000000001,
                                               1029087.28,
                                               1490031.3600000001])
        self.assertAlmostEqual(sum(area_def.area_extent_ll),
                               122.06448093539757, 5,
                               'Failed to get lon and lats of area extent')

    def test_latlong_area(self):
        """Test getting lons and lats from an area."""
        area_def = geometry.AreaDefinition('', '', '',
                                           {'proj': 'latlong'},
                                           360, 180,
                                           [-180, -90, 180, 90])
        lons, lats = area_def.get_lonlats()
        self.assertEqual(lons[0, 0], -179.5)
        self.assertEqual(lats[0, 0], 89.5)

    def test_get_array_indices_from_lonlat_mask_actual_values(self):
        """Test that the masked values of get_array_indices_from_lonlat can be valid."""
        from pyresample import get_area_def

        # The area of our source data
        area_id = 'orig'
        area_name = 'Test area'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'lat_1': 25.,
                     'lat_2': 25., 'lon_0': 0.0, 'proj': 'lcc', 'units': 'm'}
        area_def = get_area_def(area_id,
                                area_name,
                                proj_id,
                                proj_dict,
                                x_size, y_size,
                                area_extent)

        # Choose a point just outside the area
        x, y = area_def.get_array_indices_from_lonlat([33.5], [-40.5])
        assert x.item() == 3723
        assert y.item() == 3746

    def test_lonlat2colrow(self):
        """Test lonlat2colrow."""
        from pyresample import utils
        area_id = 'meteosat_0deg'
        area_name = 'Meteosat 0 degree Service'
        proj_id = 'geos0'
        x_size = 3712
        y_size = 3712
        area_extent = [-5570248.477339261, -5567248.074173444,
                       5567248.074173444, 5570248.477339261]
        proj_dict = {'a': '6378169.00',
                     'b': '6356583.80',
                     'h': '35785831.0',
                     'lon_0': '0.0',
                     'proj': 'geos'}
        area = utils.get_area_def(area_id,
                                  area_name,
                                  proj_id,
                                  proj_dict,
                                  x_size, y_size,
                                  area_extent)

        # Imatra, Wiesbaden
        longitudes = np.array([28.75242, 8.24932])
        latitudes = np.array([61.17185, 50.08258])
        cols__, rows__ = area.get_array_indices_from_lonlat(longitudes, latitudes)

        # test arrays
        cols_expects = np.array([2304, 2040])
        rows_expects = np.array([186, 341])
        np.testing.assert_array_equal(cols__, cols_expects)
        np.testing.assert_array_equal(rows__, rows_expects)

        # test scalars
        lon, lat = (-8.125547604568746, -14.345524111874646)
        self.assertEqual(area.get_array_indices_from_lonlat(lon, lat), (1567, 2375))

    def test_colrow2lonlat(self):
        """Test colrow2lonlat."""
        from pyresample import utils

        # test square, symmetric areadef
        area_id = 'meteosat_0deg'
        area_name = 'Meteosat 0 degree Service'
        proj_id = 'geos0'
        x_size = 3712
        y_size = 3712
        area_extent = [-5570248.477339261, -5567248.074173444,
                       5567248.074173444, 5570248.477339261]
        proj_dict = {'a': '6378169.00',
                     'b': '6356583.80',
                     'h': '35785831.0',
                     'lon_0': '0.0',
                     'proj': 'geos'}
        area = utils.get_area_def(area_id,
                                  area_name,
                                  proj_id,
                                  proj_dict,
                                  x_size, y_size,
                                  area_extent)

        # Imatra, Wiesbaden
        cols = np.array([2304, 2040])
        rows = np.array([186, 341])
        lons__, lats__ = area.colrow2lonlat(cols, rows)

        # test arrays
        lon_expects = np.array([28.77763033, 8.23765962])
        lat_expects = np.array([61.20120556, 50.05836402])
        self.assertTrue(np.allclose(lons__, lon_expects, rtol=0, atol=1e-7))
        self.assertTrue(np.allclose(lats__, lat_expects, rtol=0, atol=1e-7))

        # test scalars
        lon__, lat__ = area.colrow2lonlat(1567, 2375)
        lon_expect = -8.125547604568746
        lat_expect = -14.345524111874646
        self.assertTrue(np.allclose(lon__, lon_expect, rtol=0, atol=1e-7))
        self.assertTrue(np.allclose(lat__, lat_expect, rtol=0, atol=1e-7))

        # test rectangular areadef
        area_id = 'eurol'
        area_name = 'Euro 3.0km area - Europe'
        proj_id = 'eurol'
        x_size = 2560
        y_size = 2048
        area_extent = [-3780000.0, -7644000.0, 3900000.0, -1500000.0]
        proj_dict = {
            'lat_0': 90.0,
            'lon_0': 0.0,
            'lat_ts': 60.0,
            'ellps': 'WGS84',
            'proj': 'stere'}
        area = utils.get_area_def(area_id,
                                  area_name,
                                  proj_id,
                                  proj_dict,
                                  x_size, y_size,
                                  area_extent)

        # Darmstadt, Gibraltar
        cols = np.array([1477, 1069])
        rows = np.array([938, 1513])
        lons__, lats__ = area.colrow2lonlat(cols, rows)

        # test arrays
        lon_expects = np.array([8.597949006575268, -5.404744177829209])
        lat_expects = np.array([49.79024658538765, 36.00540657185169])
        self.assertTrue(np.allclose(lons__, lon_expects, rtol=0, atol=1e-7))
        self.assertTrue(np.allclose(lats__, lat_expects, rtol=0, atol=1e-7))

        # test scalars
        # Selva di Val Gardena
        lon__, lat__ = area.colrow2lonlat(1582, 1049)
        lon_expect = 11.75721385976652
        lat_expect = 46.56384754346095
        self.assertTrue(np.allclose(lon__, lon_expect, rtol=0, atol=1e-7))
        self.assertTrue(np.allclose(lat__, lat_expect, rtol=0, atol=1e-7))

    def test_get_proj_coords_basic(self):
        """Test basic get_proj_coords usage."""
        from pyresample import utils
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 10
        y_size = 10
        area_extent = [1000000, 0, 1050000, 50000]
        proj_dict = {"proj": 'laea', 'lat_0': '60', 'lon_0': '0', 'a': '6371228.0', 'units': 'm'}
        area_def = utils.get_area_def(area_id, area_name, proj_id, proj_dict, x_size, y_size, area_extent)

        xcoord, ycoord = area_def.get_proj_coords()
        self.assertTrue(np.allclose(xcoord[0, :],
                                    np.array([1002500., 1007500., 1012500.,
                                              1017500., 1022500., 1027500.,
                                              1032500., 1037500., 1042500.,
                                              1047500.])))
        self.assertTrue(np.allclose(ycoord[:, 0],
                                    np.array([47500., 42500., 37500., 32500.,
                                              27500., 22500., 17500., 12500.,
                                              7500., 2500.])))

        xcoord, ycoord = area_def.get_proj_coords(data_slice=(slice(None, None, 2),
                                                              slice(None, None, 2)))

        self.assertTrue(np.allclose(xcoord[0, :],
                                    np.array([1002500., 1012500., 1022500.,
                                              1032500., 1042500.])))
        self.assertTrue(np.allclose(ycoord[:, 0],
                                    np.array([47500., 37500., 27500., 17500.,
                                              7500.])))

    def test_get_proj_coords_rotation(self):
        """Test basic get_proj_coords usage with rotation specified."""
        from pyresample.geometry import AreaDefinition
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 10
        y_size = 10
        area_extent = [1000000, 0, 1050000, 50000]
        proj_dict = {"proj": 'laea', 'lat_0': '60', 'lon_0': '0', 'a': '6371228.0', 'units': 'm'}
        area_def = AreaDefinition(area_id, area_name, proj_id, proj_dict, x_size, y_size, area_extent, rotation=45)

        xcoord, ycoord = area_def.get_proj_coords()
        np.testing.assert_allclose(xcoord[0, :],
                                   np.array([742462.120246, 745997.654152, 749533.188058, 753068.721964,
                                             756604.25587, 760139.789776, 763675.323681, 767210.857587,
                                             770746.391493, 774281.925399]))
        np.testing.assert_allclose(ycoord[:, 0],
                                   np.array([-675286.976033, -678822.509939, -682358.043845, -685893.577751,
                                             -689429.111657, -692964.645563, -696500.179469, -700035.713375,
                                             -703571.247281, -707106.781187]))

        xcoord, ycoord = area_def.get_proj_coords(data_slice=(slice(None, None, 2), slice(None, None, 2)))
        np.testing.assert_allclose(xcoord[0, :],
                                   np.array([742462.120246, 749533.188058, 756604.25587, 763675.323681,
                                             770746.391493]))
        np.testing.assert_allclose(ycoord[:, 0],
                                   np.array([-675286.976033, -682358.043845, -689429.111657, -696500.179469,
                                             -703571.247281]))

    def test_get_proj_coords_dask(self):
        """Test get_proj_coords usage with dask arrays."""
        from pyresample import get_area_def
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 10
        y_size = 10
        area_extent = [1000000, 0, 1050000, 50000]
        proj_dict = {"proj": 'laea', 'lat_0': '60', 'lon_0': '0', 'a': '6371228.0', 'units': 'm'}
        area_def = get_area_def(area_id, area_name, proj_id, proj_dict, x_size, y_size, area_extent)

        xcoord, ycoord = area_def.get_proj_coords(chunks=4096)
        # make sure different chunk size provides a different dask name
        xcoord2, ycoord2 = area_def.get_proj_coords(chunks=2048)
        assert xcoord2.name != xcoord.name
        assert ycoord2.name != ycoord.name

        xcoord = xcoord.compute()
        ycoord = ycoord.compute()
        self.assertTrue(np.allclose(xcoord[0, :],
                                    np.array([1002500., 1007500., 1012500.,
                                              1017500., 1022500., 1027500.,
                                              1032500., 1037500., 1042500.,
                                              1047500.])))
        self.assertTrue(np.allclose(ycoord[:, 0],
                                    np.array([47500., 42500., 37500., 32500.,
                                              27500., 22500., 17500., 12500.,
                                              7500., 2500.])))

        # use the shared method and provide chunks and slices
        xcoord, ycoord = area_def.get_proj_coords(data_slice=(slice(None, None, 2),
                                                              slice(None, None, 2)),
                                                  chunks=4096)
        xcoord = xcoord.compute()
        ycoord = ycoord.compute()
        self.assertTrue(np.allclose(xcoord[0, :],
                                    np.array([1002500., 1012500., 1022500.,
                                              1032500., 1042500.])))
        self.assertTrue(np.allclose(ycoord[:, 0],
                                    np.array([47500., 37500., 27500., 17500.,
                                              7500.])))

    def test_roundtrip_lonlat_array_coordinates(self):
        """Test roundtrip."""
        from pyresample import get_area_def
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 100
        y_size = 100
        area_extent = [0, -500000, 1000000, 500000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '50',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = get_area_def(area_id,
                                area_name,
                                proj_id,
                                proj_dict,
                                x_size, y_size,
                                area_extent)
        lat, lon = 48.832222, 2.355556  # Paris, 13th arrondissement, France
        x__, y__ = area_def.get_array_coordinates_from_lonlat(lon, lat)
        res_lon, res_lat = area_def.get_lonlat_from_array_coordinates(x__, y__)
        np.testing.assert_allclose([res_lon, res_lat], [lon, lat])

    def test_roundtrip_lonlat_array_coordinates_for_dask_array(self):
        """Test roundrip for dask arrays."""
        import dask.array as da

        from pyresample import get_area_def
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 100
        y_size = 100
        area_extent = [0, -500000, 1000000, 500000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '50',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = get_area_def(area_id,
                                area_name,
                                proj_id,
                                proj_dict,
                                x_size, y_size,
                                area_extent)
        lat1, lon1 = 48.832222, 2.355556  # Paris, 13th arrondissement, France
        lat2, lon2 = 58.6, 16.2  # Norrköping, Sweden
        lon = da.from_array([lon1, lon2])
        lat = da.from_array([lat1, lat2])
        x__, y__ = area_def.get_array_coordinates_from_lonlat(lon, lat)
        res_lon, res_lat = area_def.get_lonlat_from_array_coordinates(x__, y__)
        assert isinstance(res_lon, da.Array)
        assert isinstance(res_lat, da.Array)
        np.testing.assert_allclose(res_lon, lon)
        np.testing.assert_allclose(res_lat, lat)

    def test_get_lonlats_vs_get_lonlat(self):
        """Test that both function yield similar results."""
        from pyresample import get_area_def
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 100
        y_size = 100
        area_extent = [0, -500000, 1000000, 500000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '50',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = get_area_def(area_id,
                                area_name,
                                proj_id,
                                proj_dict,
                                x_size, y_size,
                                area_extent)
        lons, lats = area_def.get_lonlats()
        x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))
        lon, lat = area_def.get_lonlat_from_array_coordinates(x, y)
        np.testing.assert_allclose(lons, lon)
        np.testing.assert_allclose(lats, lat)

    def test_area_corners_around_south_pole(self):
        """Test corner values for the ease-sh area."""
        import numpy as np

        from pyresample.geometry import AreaDefinition
        area_id = 'ease_sh'
        description = 'Antarctic EASE grid'
        proj_id = 'ease_sh'
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        width = 425
        height = 425
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
        area_def = AreaDefinition(area_id, description, proj_id, projection,
                                  width, height, area_extent)

        expected = [(-45.0, -17.713517415148853),
                    (45.000000000000014, -17.71351741514884),
                    (135.0, -17.713517415148825),
                    (-135.00000000000003, -17.71351741514884)]
        actual = [(np.rad2deg(coord.lon), np.rad2deg(coord.lat)) for coord in area_def.corners]
        np.testing.assert_allclose(actual, expected)

    def test_get_xy_from_lonlat(self):
        """Test the function get_xy_from_lonlat."""
        from pyresample import utils
        area_id = 'test'
        area_name = 'Test area with 2x2 pixels'
        proj_id = 'test'
        x_size = 2
        y_size = 2
        area_extent = [1000000, 0, 1050000, 50000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '60',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = utils.get_area_def(area_id,
                                      area_name,
                                      proj_id,
                                      proj_dict,
                                      x_size, y_size,
                                      area_extent)
        p__ = Proj(proj_dict)
        lon_ul, lat_ul = p__(1000000, 50000, inverse=True)
        lon_ur, lat_ur = p__(1050000, 50000, inverse=True)
        lon_ll, lat_ll = p__(1000000, 0, inverse=True)
        lon_lr, lat_lr = p__(1050000, 0, inverse=True)

        eps_lonlat = 0.01
        eps_meters = 100
        x__, y__ = area_def.get_xy_from_lonlat(lon_ul + eps_lonlat,
                                               lat_ul - eps_lonlat)
        x_expect, y_expect = 0, 0
        self.assertEqual(x__, x_expect)
        self.assertEqual(y__, y_expect)
        x__, y__ = area_def.get_xy_from_lonlat(lon_ur - eps_lonlat,
                                               lat_ur - eps_lonlat)
        self.assertEqual(x__, 1)
        self.assertEqual(y__, 0)
        x__, y__ = area_def.get_xy_from_lonlat(lon_ll + eps_lonlat,
                                               lat_ll + eps_lonlat)
        self.assertEqual(x__, 0)
        self.assertEqual(y__, 1)
        x__, y__ = area_def.get_xy_from_lonlat(lon_lr - eps_lonlat,
                                               lat_lr + eps_lonlat)
        self.assertEqual(x__, 1)
        self.assertEqual(y__, 1)

        lon, lat = p__(1025000 - eps_meters, 25000 - eps_meters, inverse=True)
        x__, y__ = area_def.get_xy_from_lonlat(lon, lat)
        self.assertEqual(x__, 0)
        self.assertEqual(y__, 1)

        lon, lat = p__(1025000 + eps_meters, 25000 - eps_meters, inverse=True)
        x__, y__ = area_def.get_xy_from_lonlat(lon, lat)
        self.assertEqual(x__, 1)
        self.assertEqual(y__, 1)

        lon, lat = p__(1025000 - eps_meters, 25000 + eps_meters, inverse=True)
        x__, y__ = area_def.get_xy_from_lonlat(lon, lat)
        self.assertEqual(x__, 0)
        self.assertEqual(y__, 0)

        lon, lat = p__(1025000 + eps_meters, 25000 + eps_meters, inverse=True)
        x__, y__ = area_def.get_xy_from_lonlat(lon, lat)
        self.assertEqual(x__, 1)
        self.assertEqual(y__, 0)

        lon, lat = p__(999000, -10, inverse=True)
        self.assertRaises(ValueError, area_def.get_xy_from_lonlat, lon, lat)
        self.assertRaises(ValueError, area_def.get_xy_from_lonlat, 0., 0.)

        # Test getting arrays back:
        lons = [lon_ll + eps_lonlat, lon_ur - eps_lonlat]
        lats = [lat_ll + eps_lonlat, lat_ur - eps_lonlat]
        x__, y__ = area_def.get_xy_from_lonlat(lons, lats)

        x_expects = np.array([0, 1])
        y_expects = np.array([1, 0])
        self.assertTrue((x__.data == x_expects).all())
        self.assertTrue((y__.data == y_expects).all())

    def test_get_slice_starts_stops(self):
        """Check area slice end-points."""
        from pyresample import utils
        area_id = 'orig'
        area_name = 'Test area'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        target_area = utils.get_area_def(area_id,
                                         area_name,
                                         proj_id,
                                         proj_dict,
                                         x_size, y_size,
                                         area_extent)

        # Expected result is the same for all cases
        expected = (3, 3709, 3, 3709)

        # Source and target have the same orientation
        area_extent = (-5580248.477339745, -5571247.267842293, 5577248.074173927, 5580248.477339745)
        source_area = utils.get_area_def(area_id,
                                         area_name,
                                         proj_id,
                                         proj_dict,
                                         x_size, y_size,
                                         area_extent)
        res = source_area._get_slice_starts_stops(target_area)
        assert res == expected

        # Source is flipped in X direction
        area_extent = (5577248.074173927, -5571247.267842293, -5580248.477339745, 5580248.477339745)
        source_area = utils.get_area_def(area_id,
                                         area_name,
                                         proj_id,
                                         proj_dict,
                                         x_size, y_size,
                                         area_extent)
        res = source_area._get_slice_starts_stops(target_area)
        assert res == expected

        # Source is flipped in Y direction
        area_extent = (-5580248.477339745, 5580248.477339745, 5577248.074173927, -5571247.267842293)
        source_area = utils.get_area_def(area_id,
                                         area_name,
                                         proj_id,
                                         proj_dict,
                                         x_size, y_size,
                                         area_extent)
        res = source_area._get_slice_starts_stops(target_area)
        assert res == expected

        # Source is flipped in both X and Y directions
        area_extent = (5577248.074173927, 5580248.477339745, -5580248.477339745, -5571247.267842293)
        source_area = utils.get_area_def(area_id,
                                         area_name,
                                         proj_id,
                                         proj_dict,
                                         x_size, y_size,
                                         area_extent)
        res = source_area._get_slice_starts_stops(target_area)
        assert res == expected

    def test_proj_str(self):
        """Test the 'proj_str' property of AreaDefinition."""
        from collections import OrderedDict

        from pyresample.test.utils import friendly_crs_equal

        # pyproj 2.0+ adds a +type=crs parameter
        proj_dict = OrderedDict()
        proj_dict['proj'] = 'stere'
        proj_dict['a'] = 6378144.0
        proj_dict['b'] = 6356759.0
        proj_dict['lat_0'] = 90.00
        proj_dict['lat_ts'] = 50.00
        proj_dict['lon_0'] = 8.00
        area = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                       proj_dict, 10, 10,
                                       [-1370912.72, -909968.64, 1029087.28,
                                        1490031.36])
        friendly_crs_equal(
            '+a=6378144.0 +b=6356759.0 +lat_0=90.0 +lat_ts=50.0 '
            '+lon_0=8.0 +proj=stere',
            area
        )
        # try a omerc projection and no_rot parameters
        proj_dict['proj'] = 'omerc'
        proj_dict['lat_0'] = 50.0
        proj_dict['alpha'] = proj_dict.pop('lat_ts')
        proj_dict['no_rot'] = True
        area = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                       proj_dict, 10, 10,
                                       [-1370912.72, -909968.64, 1029087.28,
                                        1490031.36])
        friendly_crs_equal(
            '+proj=omerc +a=6378144.0 +b=6356759.0 +lat_0=50.0 '
            '+lon_0=8.0 +alpha=50.0 +no_rot',
            area
        )

        # EPSG
        # With pyproj 2.0+ we expand EPSG to full parameter list
        full_proj = ('+datum=WGS84 +lat_0=-90 +lon_0=0 +no_defs '
                     '+proj=laea +type=crs +units=m +x_0=0 +y_0=0')
        projections = [
            ('+init=EPSG:6932', full_proj),
            ('EPSG:6932', full_proj)
        ]
        for projection, expected_proj in projections:
            area = geometry.AreaDefinition(
                area_id='ease-sh-2.0',
                description='25km EASE Grid 2.0 (Southern Hemisphere)',
                proj_id='ease-sh-2.0',
                projection=projection,
                width=123, height=123,
                area_extent=[-40000., -40000., 40000., 40000.])
            self.assertEqual(area.proj_str, expected_proj)

        # CRS with towgs84 in it
        # we remove towgs84 if they are all 0s
        projection = {'proj': 'laea', 'lat_0': 52, 'lon_0': 10, 'x_0': 4321000, 'y_0': 3210000,
                      'ellps': 'GRS80', 'towgs84': '0,0,0,0,0,0,0', 'units': 'm', 'no_defs': True}
        area = geometry.AreaDefinition(
            area_id='test_towgs84',
            description='',
            proj_id='',
            projection=projection,
            width=123, height=123,
            area_extent=[-40000., -40000., 40000., 40000.])
        self.assertEqual(area.proj_str,
                         '+ellps=GRS80 +lat_0=52 +lon_0=10 +no_defs +proj=laea '
                         # '+towgs84=0.0,0.0,0.0,0.0,0.0,0.0,0.0 '
                         '+type=crs +units=m '
                         '+x_0=4321000 +y_0=3210000')
        projection = {'proj': 'laea', 'lat_0': 52, 'lon_0': 10, 'x_0': 4321000, 'y_0': 3210000,
                      'ellps': 'GRS80', 'towgs84': '0,5,0,0,0,0,0', 'units': 'm', 'no_defs': True}
        area = geometry.AreaDefinition(
            area_id='test_towgs84',
            description='',
            proj_id='',
            projection=projection,
            width=123, height=123,
            area_extent=[-40000., -40000., 40000., 40000.])
        self.assertEqual(area.proj_str,
                         '+ellps=GRS80 +lat_0=52 +lon_0=10 +no_defs +proj=laea '
                         '+towgs84=0.0,5.0,0.0,0.0,0.0,0.0,0.0 '
                         '+type=crs +units=m '
                         '+x_0=4321000 +y_0=3210000')

    def test_striding(self):
        """Test striding AreaDefinitions."""
        from pyresample import utils

        area_id = 'orig'
        area_name = 'Test area'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_def = utils.get_area_def(area_id,
                                      area_name,
                                      proj_id,
                                      proj_dict,
                                      x_size, y_size,
                                      area_extent)

        reduced_area = area_def[::4, ::4]
        np.testing.assert_allclose(reduced_area.area_extent, (area_extent[0],
                                                              area_extent[1] + 3 * area_def.pixel_size_y,
                                                              area_extent[2] - 3 * area_def.pixel_size_x,
                                                              area_extent[3]))
        self.assertEqual(reduced_area.shape, (928, 928))

    def test_get_lonlats_options(self):
        """Test that lotlat options are respected."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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
        (lon, _) = area_def.get_lonlats(dtype="f4")
        self.assertEqual(lon.dtype, np.dtype("f4"))

        (lon, _) = area_def.get_lonlats(dtype="f8")
        self.assertEqual(lon.dtype, np.dtype("f8"))

        from dask.array.core import Array as dask_array
        (lon, _) = area_def.get_lonlats(dtype="f4", chunks=4)
        self.assertEqual(lon.dtype, np.dtype("f4"))
        self.assertIsInstance(lon, dask_array)

        (lon, _) = area_def.get_lonlats(dtype="f8", chunks=4)
        self.assertEqual(lon.dtype, np.dtype("f8"))
        self.assertIsInstance(lon, dask_array)

    def test_area_def_geocentric_resolution(self):
        """Test the AreaDefinition.geocentric_resolution method."""
        from pyresample import get_area_def
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        # metered projection
        area_def = get_area_def('orig', 'Test area', 'test',
                                proj_dict,
                                3712, 3712,
                                area_extent)
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(10646.562531, geo_res, rtol=1e-05)

        # non-square area non-space area
        area_extent = (-4570248.477339745, -3561247.267842293, 0, 3570248.477339745)
        area_def = get_area_def('orig', 'Test area', 'test',
                                proj_dict,
                                2000, 5000,
                                area_extent)
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(2397.687307, geo_res, rtol=1e-05)

        # lon/lat
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'proj': 'latlong'}
        area_def = get_area_def('orig', 'Test area', 'test',
                                proj_dict,
                                3712, 3712,
                                [-130, 30, -120, 40])
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(298.647232, geo_res, rtol=1e-04)

    def test_area_def_geocentric_resolution_latlong(self):
        """Test the AreaDefinition.geocentric_resolution method on a latlong projection."""
        from pyresample import get_area_def
        area_extent = (-110.0, 45.0, -95.0, 55.0)
        # metered projection
        area_def = get_area_def('orig', 'Test area', 'test',
                                {"EPSG": "4326"},
                                3712, 3712,
                                area_extent)
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(299.411133, geo_res)

    def test_from_epsg(self):
        """Test the from_epsg class method."""
        from pyresample.geometry import AreaDefinition
        sweref = AreaDefinition.from_epsg('3006', 2000)
        assert sweref.name == 'SWEREF99 TM'
        assert sweref.proj_dict == {'ellps': 'GRS80', 'no_defs': None,
                                    'proj': 'utm', 'type': 'crs', 'units': 'm',
                                    'zone': 33}
        assert sweref.width == 453
        assert sweref.height == 794
        import numpy as np
        np.testing.assert_allclose(sweref.area_extent,
                                   (181896.3291, 6101648.0705,
                                    1086312.942376, 7689478.3056))

    def test_from_cf(self):
        """Test the from_cf class method."""
        # prepare a netCDF/CF lookalike with xarray
        import xarray as xr

        from pyresample.geometry import AreaDefinition
        nlat = 19
        nlon = 37
        ds = xr.Dataset({'temp': (('lat', 'lon'), np.ma.masked_all((nlat, nlon)))},
                        coords={'lat': np.linspace(-90., +90., num=nlat),
                                'lon': np.linspace(-180., +180., num=nlon)},)
        ds['lat'].attrs['units'] = 'degreeN'
        ds['lat'].attrs['standard_name'] = 'latitude'
        ds['lon'].attrs['units'] = 'degreeE'
        ds['lon'].attrs['standard_name'] = 'longitude'

        # call from_cf() and check the results
        adef = AreaDefinition.from_cf(ds, )

        self.assertEqual(adef.shape, (19, 37))
        xc = adef.projection_x_coords
        yc = adef.projection_y_coords
        self.assertEqual(xc[0], -180., msg="Wrong x axis (index 0)")
        self.assertEqual(xc[1], -180. + 10.0, msg="Wrong x axis (index 1)")
        self.assertEqual(yc[0], -90., msg="Wrong y axis (index 0)")
        self.assertEqual(yc[1], -90. + 10.0, msg="Wrong y axis (index 1)")

    @unittest.skipIf(CRS is None, "pyproj 2.0+ required")
    def test_area_def_init_projection(self):
        """Test AreaDefinition with different projection definitions."""
        proj_dict = {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '90.00',
            'lat_ts': '50.00',
            'lon_0': '8.00',
            'proj': 'stere'
        }
        crs = CRS(CRS.from_dict(proj_dict).to_wkt())
        # pass CRS object directly
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           crs,
                                           800, 800,
                                           [-1370912.72, -909968.64000000001,
                                            1029087.28, 1490031.3600000001])
        self.assertEqual(crs, area_def.crs)
        # PROJ dictionary
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           crs.to_dict(),
                                           800, 800,
                                           [-1370912.72, -909968.64000000001,
                                            1029087.28, 1490031.3600000001])
        self.assertEqual(crs, area_def.crs)
        # PROJ string
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           crs.to_string(),
                                           800, 800,
                                           [-1370912.72, -909968.64000000001,
                                            1029087.28, 1490031.3600000001])
        self.assertEqual(crs, area_def.crs)
        # WKT2
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           crs.to_wkt(),
                                           800, 800,
                                           [-1370912.72, -909968.64000000001,
                                            1029087.28, 1490031.3600000001])
        self.assertEqual(crs, area_def.crs)
        # WKT1_ESRI
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           crs.to_wkt(version='WKT1_ESRI'),
                                           800, 800,
                                           [-1370912.72, -909968.64000000001,
                                            1029087.28, 1490031.3600000001])
        # WKT1 to WKT2 has some different naming of things so this fails
        # self.assertEqual(crs, area_def.crs)

    def test_areadef_immutable(self):
        """Test that some properties of an area definition are immutable."""
        area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                           {'a': '6378144.0',
                                            'b': '6356759.0',
                                            'lat_0': '50.00',
                                            'lat_ts': '50.00',
                                            'lon_0': '8.00',
                                            'proj': 'stere'},
                                           10,
                                           10,
                                           [-1370912.72,
                                               -909968.64000000001,
                                               1029087.28,
                                               1490031.3600000001])
        with pytest.raises(AttributeError):
            area_def.shape = (10, 10)
        with pytest.raises(AttributeError):
            area_def.proj_str = "seaweed"
        with pytest.raises(AttributeError):
            area_def.area_extent = (-1000000, -900000, 1000000, 1500000)


class TestMakeSliceDivisible(unittest.TestCase):
    """Test the _make_slice_divisible."""

    def test_make_slice_divisible(self):
        """Test that making area shape divisible by a given factor works."""
        from pyresample.geometry import _make_slice_divisible

        # Divisible by 2
        sli = slice(10, 21)
        factor = 2
        self.assertNotEqual((sli.stop - sli.start) % factor, 0)
        res = _make_slice_divisible(sli, 1000, factor=factor)
        self.assertEqual((res.stop - res.start) % factor, 0)

        # Divisible by 3
        sli = slice(10, 23)
        factor = 3
        self.assertNotEqual((sli.stop - sli.start) % factor, 0)
        res = _make_slice_divisible(sli, 1000, factor=factor)
        self.assertEqual((res.stop - res.start) % factor, 0)

        # Divisible by 5
        sli = slice(10, 23)
        factor = 5
        self.assertNotEqual((sli.stop - sli.start) % factor, 0)
        res = _make_slice_divisible(sli, 1000, factor=factor)
        self.assertEqual((res.stop - res.start) % factor, 0)


def assert_np_dict_allclose(dict1, dict2):
    """Check allclose on dicts."""
    assert set(dict1.keys()) == set(dict2.keys())
    for key, val in dict1.items():
        try:
            np.testing.assert_allclose(val, dict2[key])
        except TypeError:
            assert val == dict2[key]


class TestSwathDefinition(unittest.TestCase):
    """Test the SwathDefinition."""

    def test_swath(self):
        """Test swath."""
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        swath_def = geometry.SwathDefinition(lons1, lats1)

        lons2, lats2 = swath_def.get_lonlats()

        self.assertFalse(id(lons1) != id(lons2) or id(lats1) != id(lats2),
                         msg='Caching of swath coordinates failed')

    def test_slice(self):
        """Test that SwathDefinitions can be sliced."""
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        swath_def = geometry.SwathDefinition(lons1, lats1)
        new_swath_def = swath_def[1000:4000, 20:40]
        self.assertTupleEqual(new_swath_def.lons.shape, (3000, 20))
        self.assertTupleEqual(new_swath_def.lats.shape, (3000, 20))

    def test_concat_1d(self):
        """Test concatenating in 1d."""
        lons1 = np.array([1, 2, 3])
        lats1 = np.array([1, 2, 3])
        lons2 = np.array([4, 5, 6])
        lats2 = np.array([4, 5, 6])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def_concat = swath_def1.concatenate(swath_def2)
        expected = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(swath_def_concat.lons, expected) and
                        np.array_equal(swath_def_concat.lons, expected),
                        'Failed to concatenate 1D swaths')

    def test_concat_2d(self):
        """Test concatenating in 2d."""
        lons1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lats1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lons2 = np.array([[4, 5, 6], [6, 7, 8]])
        lats2 = np.array([[4, 5, 6], [6, 7, 8]])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def_concat = swath_def1.concatenate(swath_def2)
        expected = np.array(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 5, 6], [6, 7, 8]])
        self.assertTrue(np.array_equal(swath_def_concat.lons, expected) and
                        np.array_equal(swath_def_concat.lons, expected),
                        'Failed to concatenate 2D swaths')

    def test_append_1d(self):
        """Test appending in 1d."""
        lons1 = np.array([1, 2, 3])
        lats1 = np.array([1, 2, 3])
        lons2 = np.array([4, 5, 6])
        lats2 = np.array([4, 5, 6])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def1.append(swath_def2)
        expected = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(swath_def1.lons, expected) and
                        np.array_equal(swath_def1.lons, expected),
                        'Failed to append 1D swaths')

    def test_append_2d(self):
        """Test appending in 2d."""
        lons1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lats1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lons2 = np.array([[4, 5, 6], [6, 7, 8]])
        lats2 = np.array([[4, 5, 6], [6, 7, 8]])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def1.append(swath_def2)
        expected = np.array(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 5, 6], [6, 7, 8]])
        self.assertTrue(np.array_equal(swath_def1.lons, expected) and
                        np.array_equal(swath_def1.lons, expected),
                        'Failed to append 2D swaths')

    def test_swath_equal(self):
        """Test swath equality."""
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)
        swath_def2 = geometry.SwathDefinition(lons, lats)
        # Identical lons and lats
        self.assertFalse(
            swath_def != swath_def2, 'swath_defs are not equal as expected')
        # Identical objects
        self.assertFalse(
            swath_def != swath_def, 'swath_defs are not equal as expected')

        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        lons2 = np.array([1.2, 1.3, 1.4, 1.5])
        lats2 = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        # different arrays, same values
        self.assertFalse(
            swath_def != swath_def2, 'swath_defs are not equal as expected')

        lons = np.array([1.2, 1.3, 1.4, np.nan])
        lats = np.array([65.9, 65.86, 65.82, np.nan])
        lons2 = np.array([1.2, 1.3, 1.4, np.nan])
        lats2 = np.array([65.9, 65.86, 65.82, np.nan])
        swath_def = geometry.SwathDefinition(lons, lats)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        # different arrays, same values, with nans
        self.assertFalse(
            swath_def != swath_def2, 'swath_defs are not equal as expected')

        try:
            import dask.array as da
            lons = da.from_array(np.array([1.2, 1.3, 1.4, np.nan]), chunks=2)
            lats = da.from_array(np.array([65.9, 65.86, 65.82, np.nan]), chunks=2)
            lons2 = da.from_array(np.array([1.2, 1.3, 1.4, np.nan]), chunks=2)
            lats2 = da.from_array(np.array([65.9, 65.86, 65.82, np.nan]), chunks=2)
            swath_def = geometry.SwathDefinition(lons, lats)
            swath_def2 = geometry.SwathDefinition(lons2, lats2)
            # different arrays, same values, with nans
            self.assertFalse(
                swath_def != swath_def2, 'swath_defs are not equal as expected')
        except ImportError:
            pass

        try:
            import xarray as xr
            lons = xr.DataArray(np.array([1.2, 1.3, 1.4, np.nan]))
            lats = xr.DataArray(np.array([65.9, 65.86, 65.82, np.nan]))
            lons2 = xr.DataArray(np.array([1.2, 1.3, 1.4, np.nan]))
            lats2 = xr.DataArray(np.array([65.9, 65.86, 65.82, np.nan]))
            swath_def = geometry.SwathDefinition(lons, lats)
            swath_def2 = geometry.SwathDefinition(lons2, lats2)
            # different arrays, same values, with nans
            self.assertFalse(
                swath_def != swath_def2, 'swath_defs are not equal as expected')

        except ImportError:
            pass

    def test_swath_not_equal(self):
        """Test swath inequality."""
        lats1 = np.array([65.9, 65.86, 65.82, 65.78])
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats2 = np.array([65.91, 65.85, 65.80, 65.75])
        swath_def = geometry.SwathDefinition(lons, lats1)
        swath_def2 = geometry.SwathDefinition(lons, lats2)
        self.assertFalse(
            swath_def == swath_def2, 'swath_defs are not expected to be equal')

    def test_compute_omerc_params(self):
        """Test omerc parameters computation."""
        lats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                         [80.84000396728516, 60.74200439453125, 34.08500289916992],
                         [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T

        lons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                         [79.11000061035156, 7.284000396728516, -5.107000350952148],
                         [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T

        area = geometry.SwathDefinition(lons, lats)
        proj_dict = {'lonc': -11.391744043133668, 'ellps': 'WGS84',
                     'proj': 'omerc', 'alpha': 9.185764390923012,
                     'gamma': 0, 'lat_0': -0.2821013754097188}
        assert_np_dict_allclose(area._compute_omerc_parameters('WGS84'),
                                proj_dict)
        import xarray as xr
        lats = xr.DataArray(np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469, np.nan],
                                      [80.84000396728516, 60.74200439453125, 34.08500289916992, np.nan],
                                      [67.07600402832031, 54.147003173828125, 30.547000885009766, np.nan]]).T,
                            dims=['y', 'x'])

        lons = xr.DataArray(np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906, np.nan],
                                      [79.11000061035156, 7.284000396728516, -5.107000350952148, np.nan],
                                      [81.26400756835938, 29.672000885009766, 10.260000228881836, np.nan]]).T)

        area = geometry.SwathDefinition(lons, lats)
        proj_dict = {'lonc': -11.391744043133668, 'ellps': 'WGS84',
                     'proj': 'omerc', 'alpha': 9.185764390923012,
                     'gamma': 0, 'lat_0': -0.2821013754097188}
        assert_np_dict_allclose(area._compute_omerc_parameters('WGS84'),
                                proj_dict)

    def test_get_edge_lonlats(self):
        """Test the `get_edge_lonlats` functionality."""
        lats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                         [80.84000396728516, 60.74200439453125, 34.08500289916992],
                         [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T

        lons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                         [79.11000061035156, 7.284000396728516, -5.107000350952148],
                         [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T

        area = geometry.SwathDefinition(lons, lats)
        lons, lats = area.get_edge_lonlats()

        np.testing.assert_allclose(lons, [-90.67900085, 79.11000061, 81.26400757,
                                          81.26400757, 29.67200089, 10.26000023,
                                          10.26000023, -5.10700035, -21.52500153,
                                          -21.52500153, -21.56500053, -90.67900085])
        np.testing.assert_allclose(lats, [85.23900604, 80.84000397, 67.07600403,
                                          67.07600403, 54.14700317, 30.54700089,
                                          30.54700089, 34.0850029, 35.58000183,
                                          35.58000183, 62.25600433, 85.23900604])

        lats = np.array([[80., 80., 80.],
                         [80., 90., 80],
                         [80., 80., 80.]]).T

        lons = np.array([[-45., 0., 45.],
                         [-90, 0., 90.],
                         [-135., -180., 135.]]).T

        area = geometry.SwathDefinition(lons, lats)
        lons, lats = area.get_edge_lonlats()

        np.testing.assert_allclose(lons, [-45., -90., -135., -135., -180., 135.,
                                          135., 90., 45., 45., 0., -45.])
        np.testing.assert_allclose(lats, [80., 80., 80., 80., 80., 80., 80.,
                                          80., 80., 80., 80., 80.])

    def test_compute_optimal_bb(self):
        """Test computing the bb area."""
        import xarray as xr
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        area = geometry.SwathDefinition(lons, lats)

        res = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'})

        np.testing.assert_allclose(res.area_extent, [-2348379.728104, 3228086.496211,
                                                     2432121.058435, 10775774.254169])
        proj_dict = {'gamma': 0.0, 'lonc': -11.391744043133668,
                     'ellps': 'WGS84', 'proj': 'omerc',
                     'alpha': 9.185764390923012, 'lat_0': -0.2821013754097188}
        # pyproj2 adds some extra defaults
        proj_dict.update({'x_0': 0, 'y_0': 0, 'units': 'm',
                          'k': 1, 'gamma': 0,
                          'no_defs': None, 'type': 'crs'})
        assert_np_dict_allclose(res.proj_dict, proj_dict)
        self.assertEqual(res.shape, (6, 3))

        area = geometry.SwathDefinition(nplons, nplats)

        res = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'})

        np.testing.assert_allclose(res.area_extent, [-2348379.728104, 3228086.496211,
                                                     2432121.058435, 10775774.254169])
        proj_dict = {'gamma': 0.0, 'lonc': -11.391744043133668,
                     'ellps': 'WGS84', 'proj': 'omerc',
                     'alpha': 9.185764390923012, 'lat_0': -0.2821013754097188}
        # pyproj2 adds some extra defaults
        proj_dict.update({'x_0': 0, 'y_0': 0, 'units': 'm',
                          'k': 1, 'gamma': 0,
                          'no_defs': None, 'type': 'crs'})
        assert_np_dict_allclose(res.proj_dict, proj_dict)
        self.assertEqual(res.shape, (6, 3))

    def test_compute_optimal_bb_with_resolution(self):
        """Test computing the bb area while passing in the resolution."""
        import xarray as xr
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        area = geometry.SwathDefinition(lons, lats)

        res1000 = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'}, resolution=1000)
        res10000 = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'}, resolution=10000)
        assert res1000.shape[0] // 10 == res10000.shape[0]
        assert res1000.shape[1] // 10 == res10000.shape[1]

    def test_aggregation(self):
        """Test aggregation on SwathDefinitions."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        window_size = 2
        resolution = 3
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats, chunks=2), dims=['y', 'x'],
                             attrs={'resolution': resolution})
        xlons = xr.DataArray(da.from_array(lons, chunks=2), dims=['y', 'x'],
                             attrs={'resolution': resolution})
        from pyresample.geometry import SwathDefinition
        sd = SwathDefinition(xlons, xlats)
        res = sd.aggregate(y=window_size, x=window_size)
        np.testing.assert_allclose(res.lons, [[179, -179]])
        np.testing.assert_allclose(res.lats, [[0.5, 0.5]], atol=2e-5)
        self.assertAlmostEqual(res.lons.resolution, window_size * resolution)
        self.assertAlmostEqual(res.lats.resolution, window_size * resolution)

    def test_striding(self):
        """Test striding."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats, chunks=2), dims=['y', 'x'])
        xlons = xr.DataArray(da.from_array(lons, chunks=2), dims=['y', 'x'])
        from pyresample.geometry import SwathDefinition
        sd = SwathDefinition(xlons, xlats)
        res = sd[::2, ::2]
        np.testing.assert_allclose(res.lons, [[178.5, -179.5]])
        np.testing.assert_allclose(res.lats, [[0, 0]], atol=2e-5)

    def test_swath_def_geocentric_resolution(self):
        """Test the SwathDefinition.geocentric_resolution method."""
        import dask.array as da
        import numpy as np
        import xarray as xr

        from pyresample.geometry import SwathDefinition
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])
        xlats = xr.DataArray(da.from_array(lats, chunks=2), dims=['y', 'x'])
        xlons = xr.DataArray(da.from_array(lons, chunks=2), dims=['y', 'x'])
        sd = SwathDefinition(xlons, xlats)
        geo_res = sd.geocentric_resolution()
        # google says 1 degrees of longitude is about ~111.321km
        # so this seems good
        np.testing.assert_allclose(111301.237078, geo_res)

        # with a resolution attribute that is None
        xlons.attrs['resolution'] = None
        xlats.attrs['resolution'] = None
        sd = SwathDefinition(xlons, xlats)
        geo_res = sd.geocentric_resolution()
        np.testing.assert_allclose(111301.237078, geo_res)

        # with a resolution attribute that is a number
        xlons.attrs['resolution'] = 111301.237078 / 2
        xlats.attrs['resolution'] = 111301.237078 / 2
        sd = SwathDefinition(xlons, xlats)
        geo_res = sd.geocentric_resolution()
        np.testing.assert_allclose(111301.237078, geo_res)

        # 1D
        xlats = xr.DataArray(da.from_array(lats.ravel(), chunks=2), dims=['y'])
        xlons = xr.DataArray(da.from_array(lons.ravel(), chunks=2), dims=['y'])
        sd = SwathDefinition(xlons, xlats)
        self.assertRaises(RuntimeError, sd.geocentric_resolution)

    def test_crs_is_stored(self):
        """Check that the CRS attribute is stored when passed."""
        from pyresample.geometry import SwathDefinition
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])

        from pyproj import CRS

        expected_crs = CRS(proj="longlat", ellps="bessel")

        sd = SwathDefinition(lons, lats, crs=expected_crs)
        assert sd.crs == expected_crs

    def test_crs_is_created_by_default(self):
        """Check that the CRS attribute is set to a default."""
        from pyresample.geometry import SwathDefinition
        lats = np.array([[0, 0, 0, 0], [1, 1, 1, 1.0]])
        lons = np.array([[178.5, 179.5, -179.5, -178.5], [178.5, 179.5, -179.5, -178.5]])

        from pyproj import CRS

        expected_crs = CRS(proj="longlat", ellps="WGS84")

        sd = SwathDefinition(lons, lats)
        assert sd.crs == expected_crs


class TestStackedAreaDefinition:
    """Test the StackedAreaDefition."""

    def test_append(self):
        """Appending new definitions."""
        area1 = geometry.AreaDefinition("area1", 'area1', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3715498.9194295374,
                                            -1830246.0673044831, 3251436.5796920112)
                                        )

        area2 = geometry.AreaDefinition("area2", 'area2', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 4179561.259167064,
                                            -1830246.0673044831, 3715498.9194295374)
                                        )

        adef = geometry.StackedAreaDefinition(area1, area2)
        assert len(adef.defs) == 1
        assert adef.defs[0].area_extent == (3738502.0095458371,
                                            4179561.259167064,
                                            -1830246.0673044831,
                                            3251436.5796920112)

        # same
        area3 = geometry.AreaDefinition("area3", 'area3', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3251436.5796920112,
                                         -1830246.0673044831, 2787374.2399544837))
        adef.append(area3)
        assert len(adef.defs) == 1
        assert adef.defs[0].area_extent == (3738502.0095458371,
                                            4179561.259167064,
                                            -1830246.0673044831,
                                            2787374.2399544837)
        assert isinstance(adef.squeeze(), geometry.AreaDefinition)

        # transition
        area4 = geometry.AreaDefinition("area4", 'area4', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (5567747.7409681147, 2787374.2399544837,
                                         -1000.3358822065015, 2323311.9002169576))

        adef.append(area4)
        assert len(adef.defs) == 2
        assert adef.defs[-1].area_extent == (5567747.7409681147,
                                             2787374.2399544837,
                                             -1000.3358822065015,
                                             2323311.9002169576)

        assert adef.height == 4 * 464
        assert isinstance(adef.squeeze(), geometry.StackedAreaDefinition)

        adef2 = geometry.StackedAreaDefinition()
        assert len(adef2.defs) == 0

        adef2.append(adef)
        assert len(adef2.defs) == 2
        assert adef2.defs[-1].area_extent == (5567747.7409681147,
                                              2787374.2399544837,
                                              -1000.3358822065015,
                                              2323311.9002169576)

        assert adef2.height == 4 * 464

    def test_get_lonlats(self):
        """Test get_lonlats on StackedAreaDefinition."""
        area3 = geometry.AreaDefinition("area3", 'area3', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3251436.5796920112,
                                         -1830246.0673044831, 2787374.2399544837))

        # transition
        area4 = geometry.AreaDefinition("area4", 'area4', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (5567747.7409681147, 2787374.2399544837,
                                         -1000.3358822065015, 2323311.9002169576))

        final_area = geometry.StackedAreaDefinition(area3, area4)
        assert len(final_area.defs) == 2
        lons, lats = final_area.get_lonlats()
        lons0, lats0 = final_area.defs[0].get_lonlats()
        lons1, lats1 = final_area.defs[1].get_lonlats()
        np.testing.assert_allclose(lons[:464, :], lons0)
        np.testing.assert_allclose(lons[464:, :], lons1)
        np.testing.assert_allclose(lats[:464, :], lats0)
        np.testing.assert_allclose(lats[464:, :], lats1)

        # check that get_lonlats with chunks definition doesn't cause errors and output arrays are equal
        with pytest.raises(ValueError):
            # too many chunks
            _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((200, 264, 464), (5570,)))
        # right amount of chunks, different shape
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((464, 470), (5568,)))
        # only one chunk value
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=464)

        # only one set of chunks in a tuple
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=(464, 5568))
        # too few chunks
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((464,), (5568,)))
        # right amount of chunks, same shape
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((464, 464), (5568,)))

    def test_combine_area_extents(self):
        """Test combination of area extents."""
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 6, 3, 2)
        res = combine_area_extents_vertical(area1, area2)
        assert res == [1, 6, 3, 4]

        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        res = combine_area_extents_vertical(area1, area2)
        assert res == [1, 2, 3, 6]

        # Non contiguous area extends shouldn't be combinable
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 5, 3, 7)
        pytest.raises(IncompatibleAreas, combine_area_extents_vertical,
                      area1, area2)

    def test_append_area_defs_fail(self):
        """Fail appending areas."""
        area1 = MagicMock()
        area1.proj_dict = {"proj": 'A'}
        area1.width = 4
        area1.height = 5
        area2 = MagicMock()
        area2.proj_dict = {'proj': 'B'}
        area2.width = 4
        area2.height = 6
        # res = combine_area_extents_vertical(area1, area2)
        pytest.raises(IncompatibleAreas, concatenate_area_defs, area1, area2)

    @patch('pyresample.geometry.AreaDefinition')
    def test_append_area_defs(self, adef):
        """Test appending area definitions."""
        x_size = random.randrange(6425)
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area1.crs = 'some_crs'
        area1.height = random.randrange(6425)
        area1.width = x_size

        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        area2.crs = 'some_crs'
        area2.height = random.randrange(6425)
        area2.width = x_size

        concatenate_area_defs(area1, area2)
        area_extent = [1, 2, 3, 6]
        y_size = area1.height + area2.height
        adef.assert_called_once_with(area1.area_id, area1.description, area1.proj_id,
                                     area1.crs, area1.width, y_size, area_extent)


class TestCreateAreaDef:
    """Test the 'create_area_def' utility function."""

    @staticmethod
    def _compare_area_defs(actual, expected, use_proj4=False):
        if use_proj4:
            # some EPSG codes have a lot of extra metadata that makes the CRS
            # unequal. Skip real area equality and use this as an approximation
            actual_str = actual.crs.to_proj4()
            expected_str = expected.crs.to_proj4()
            assert actual_str == expected_str
            assert actual.shape == expected.shape
            np.allclose(actual.area_extent, expected.area_extent)
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        'projection',
        [
            {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m',
            '+init=EPSG:3409',
            'EPSG:3409',
        ])
    @pytest.mark.parametrize(
        'center',
        [
            [0, 0],
            'a',
            (1, 2, 3),
        ])
    @pytest.mark.parametrize('units', ['meters', 'degrees'])
    def test_create_area_def_base_combinations(self, projection, center, units):
        """Test create_area_def and the four sub-methods that call it in AreaDefinition."""
        from pyresample.area_config import create_area_def as cad
        from pyresample.geometry import AreaDefinition

        area_id = 'ease_sh'
        description = 'Antarctic EASE grid'
        proj_id = 'ease_sh'
        shape = (425, 850)
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
        base_def = AreaDefinition(
            area_id, description, '',
            {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            shape[1], shape[0], area_extent)

        # Tests that incorrect lists do not create an area definition, that both projection strings and
        # dicts are accepted, and that degrees and meters both create the same area definition.
        # area_list used to check that areas are all correct at the end.
        # essentials = center, radius, upper_left_extent, resolution, shape.
        if 'm' in units:
            # Meters.
            essentials = [[0, 0], [5326849.0625, 5326849.0625], (-5326849.0625, 5326849.0625),
                          (12533.7625, 25067.525), (425, 850)]
        else:
            # Degrees.
            essentials = [(0.0, -90.0), 49.4217406986, (-45.0, -17.516001139327766),
                          (0.11271481862984278, 0.22542974631297721), (425, 850)]
        # If center is valid, use it.
        if len(center) == 2:
            center = essentials[0]

        args = (area_id, projection)
        kwargs = dict(
            proj_id=proj_id,
            upper_left_extent=essentials[2],
            center=center,
            shape=essentials[4],
            resolution=essentials[3],
            radius=essentials[1],
            description=description,
            units=units,
            rotation=45,
        )

        should_fail = isinstance(center, str) or len(center) != 2
        if should_fail:
            pytest.raises(ValueError, cad, *args, **kwargs)
        else:
            area_def = cad(*args, **kwargs)
            self._compare_area_defs(area_def, base_def, use_proj4="EPSG" in projection)

    def test_create_area_def_extra_combinations(self):
        """Test extra combinations of create_area_def parameters."""
        from xarray import DataArray

        from pyresample import create_area_def as cad
        from pyresample.geometry import AreaDefinition

        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        area_id = 'ease_sh'
        description = 'Antarctic EASE grid'
        shape = (425, 850)
        upper_left_extent = (-5326849.0625, 5326849.0625)
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
        resolution = (12533.7625, 25067.525)
        radius = [5326849.0625, 5326849.0625]
        base_def = AreaDefinition(
            area_id, description, '',
            {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            shape[1], shape[0], area_extent)

        # Tests that specifying units through xarrays works.
        area_def = cad(area_id, projection, shape=shape,
                       area_extent=DataArray(
                           (-135.0, -17.516001139327766, 45.0, -17.516001139327766),
                           attrs={'units': 'degrees'}))
        self._compare_area_defs(area_def, base_def)

        # Tests area functions 1-A and 2-A.
        area_def = cad(area_id, projection, resolution=resolution, area_extent=area_extent)
        self._compare_area_defs(area_def, base_def)

        # Tests area function 1-B. Also test that DynamicAreaDefinition arguments don't crash AreaDefinition.
        area_def = cad(area_id, projection, shape=shape, center=[0, 0],
                       upper_left_extent=upper_left_extent, optimize_projection=None)
        self._compare_area_defs(area_def, base_def)

        # Tests area function 1-C.
        area_def = cad(area_id, projection, shape=shape, center=[0, 0],
                       radius=radius)
        self._compare_area_defs(area_def, base_def)

        # Tests area function 1-D.
        area_def = cad(area_id, projection, shape=shape,
                       radius=radius, upper_left_extent=upper_left_extent)
        self._compare_area_defs(area_def, base_def)

        # Tests all 4 user cases.
        area_def = AreaDefinition.from_extent(area_id, projection, shape, area_extent)
        self._compare_area_defs(area_def, base_def)

        area_def = AreaDefinition.from_circle(area_id, projection, [0, 0], radius,
                                              resolution=resolution)
        self._compare_area_defs(area_def, base_def)
        area_def = AreaDefinition.from_area_of_interest(area_id, projection,
                                                        shape, [0, 0],
                                                        resolution)
        self._compare_area_defs(area_def, base_def)
        area_def = AreaDefinition.from_ul_corner(area_id, projection, shape,
                                                 upper_left_extent,
                                                 resolution)
        self._compare_area_defs(area_def, base_def)

    def test_create_area_def_nonpole_center(self):
        """Test that a non-pole center can be used."""
        from pyresample import create_area_def as cad
        from pyresample.geometry import AreaDefinition
        area_def = cad('ease_sh', '+a=6371228.0 +units=m +lon_0=0 +proj=merc +lat_0=0',
                       center=(0, 0), radius=45,
                       resolution=(1, 0.9999291722135637),
                       units='degrees')
        assert isinstance(area_def, AreaDefinition)
        np.testing.assert_allclose(area_def.area_extent, (-5003950.7698, -5615432.0761, 5003950.7698, 5615432.0761))
        assert area_def.shape == (101, 90)

    def test_aggregate(self):
        """Test aggregation of AreaDefinitions."""
        area = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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
        res = area.aggregate(x=4, y=2)
        assert res.proj_dict == area.proj_dict
        np.testing.assert_allclose(res.area_extent, area.area_extent)
        assert res.shape[0] == area.shape[0] / 2
        assert res.shape[1] == area.shape[1] / 4


def _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks):
    """Compute the lons and lats with chunk definition and check that they are as expected."""
    lons_c, lats_c = final_area.get_lonlats(chunks=chunks)
    np.testing.assert_array_equal(lons, lons_c)
    np.testing.assert_array_equal(lats, lats_c)


class TestDynamicAreaDefinition:
    """Test the DynamicAreaDefinition class."""

    def test_freeze(self):
        """Test freezing the area."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'laea'})
        lons = [10, 10, 22, 22]
        lats = [50, 66, 66, 50]
        result = area.freeze((lons, lats),
                             resolution=3000,
                             proj_info={'lon_0': 16, 'lat_0': 58})

        np.testing.assert_allclose(result.area_extent, (-432079.38952,
                                                        -872594.690447,
                                                        432079.38952,
                                                        904633.303964))
        assert result.proj_dict['lon_0'] == 16
        assert result.proj_dict['lat_0'] == 58
        assert result.width == 288
        assert result.height == 592

        # make sure that setting `proj_info` once doesn't
        # set it in the dynamic area
        result = area.freeze((lons, lats),
                             resolution=3000,
                             proj_info={'lon_0': 0})
        np.testing.assert_allclose(result.area_extent, (538546.7274949469,
                                                        5380808.879250369,
                                                        1724415.6519203288,
                                                        6998895.701001488))
        assert result.proj_dict['lon_0'] == 0
        # lat_0 could be provided or not depending on version of pyproj
        assert result.proj_dict.get('lat_0', 0) == 0
        assert result.width == 395
        assert result.height == 539

    def test_freeze_when_area_is_optimized_and_has_a_resolution(self):
        """Test freezing an optimized area with a resolution."""
        import xarray as xr
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        swath = geometry.SwathDefinition(lons, lats)

        area10km = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                                  {'ellps': 'WGS84', 'proj': 'omerc'},
                                                  resolution=10000,
                                                  optimize_projection=True)

        result10km = area10km.freeze(swath)
        assert result10km.shape == (679, 330)

    def test_freeze_when_area_is_optimized_and_a_resolution_is_provided(self):
        """Test freezing an optimized area when provided a resolution."""
        import xarray as xr
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        swath = geometry.SwathDefinition(lons, lats)

        area10km = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                                  {'ellps': 'WGS84', 'proj': 'omerc'},
                                                  optimize_projection=True)

        result10km = area10km.freeze(swath, 10000)
        assert result10km.shape == (679, 330)

    @pytest.mark.parametrize(
        ('lats',),
        [
            (np.linspace(-25.0, -10.0, 10),),
            (np.linspace(10.0, 25.0, 10),),
            (np.linspace(75, 90.0, 10),),
            (np.linspace(-75, -90.0, 10),),
        ],
    )
    @pytest.mark.parametrize('use_dask', [False, True])
    def test_freeze_longlat_antimeridian(self, lats, use_dask):
        """Test geographic areas over the antimeridian."""
        import dask

        from pyresample.test.utils import CustomScheduler
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              'EPSG:4326')
        lons = np.linspace(175, 185, 10)
        lons[lons > 180] -= 360
        is_pole = (np.abs(lats) > 88).any()
        if use_dask:
            # if we aren't at a pole then we adjust the coordinates
            # that takes a total of 2 computations
            num_computes = 1 if is_pole else 2
            lons = da.from_array(lons, chunks=2)
            lats = da.from_array(lats, chunks=2)
            with dask.config.set(scheduler=CustomScheduler(num_computes)):
                result = area.freeze((lons, lats),
                                     resolution=0.0056)
        else:
            result = area.freeze((lons, lats),
                                 resolution=0.0056)

        extent = result.area_extent
        if is_pole:
            assert extent[0] < -178
            assert extent[2] > 178
            assert result.width == 64088
        else:
            assert extent[0] > 0
            assert extent[2] > 0
            assert result.width == 1787
        assert result.height == 2680

    def test_freeze_with_bb(self):
        """Test freezing the area with bounding box computation."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area', {'proj': 'omerc'},
                                              optimize_projection=True)
        lons = [[10, 12.1, 14.2, 16.3],
                [10, 12, 14, 16],
                [10, 11.9, 13.8, 15.7]]
        lats = [[66, 67, 68, 69.],
                [58, 59, 60, 61],
                [50, 51, 52, 53]]
        import xarray as xr
        sdef = geometry.SwathDefinition(xr.DataArray(lons), xr.DataArray(lats))
        result = area.freeze(sdef)
        np.testing.assert_allclose(result.area_extent,
                                   [-335439.956533, 5502125.451125,
                                    191991.313351, 7737532.343683])

        assert result.width == 4
        assert result.height == 18
        # Test for properties and shape usage in freeze.
        area = geometry.DynamicAreaDefinition('test_area', 'A test area', {'proj': 'merc'},
                                              width=4, height=18)
        assert (18, 4) == area.shape
        result = area.freeze(sdef)
        np.testing.assert_allclose(result.area_extent,
                                   (996309.4426, 6287132.757981, 1931393.165263, 10837238.860543))
        area = geometry.DynamicAreaDefinition('test_area', 'A test area', {'proj': 'merc'},
                                              resolution=1000)
        assert 1000 == area.pixel_size_x
        assert 1000 == area.pixel_size_y

    def test_compute_domain(self):
        """Test computing size and area extent."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'laea'})
        corners = [1, 1, 9, 9]
        pytest.raises(ValueError, area.compute_domain, corners, 1, 1)

        area_extent, x_size, y_size = area.compute_domain(corners, shape=(5, 5))
        assert area_extent == (0, 0, 10, 10)
        assert x_size == 5
        assert y_size == 5

        area_extent, x_size, y_size = area.compute_domain(corners, resolution=2)
        assert area_extent == (0, 0, 10, 10)
        assert x_size == 5
        assert y_size == 5

    @pytest.mark.parametrize(
        (
            "antimeridian_mode",
            "expected_shape",
            "expected_extents",
            "include_proj_components",
            "exclude_proj_components"
        ),
        [
            (None, (21, 59), (164.75, 24.75, 194.25, 35.25), tuple(), ("+pm=180",)),
            ("modify_extents", (21, 59), (164.75, 24.75, 194.25, 35.25), tuple(), ("+pm=180",)),
            ("modify_crs", (21, 59), (164.75 - 180.0, 24.75, 194.25 - 180.0, 35.25), ("+pm=180",), tuple()),
            ("global_extents", (21, 720), (-180.0, 24.75, 180.0, 35.25), tuple(), ("+pm=180",)),
        ],
    )
    @pytest.mark.parametrize("use_dask", [False, True])
    def test_antimeridian_mode(self,
                               use_dask,
                               antimeridian_mode,
                               expected_shape,
                               expected_extents,
                               include_proj_components,
                               exclude_proj_components):
        """Test that antimeridian_mode affects the result."""
        dyn_area = geometry.DynamicAreaDefinition('test_area', '', {'proj': 'longlat'})
        lons, lats = _get_fake_antimeridian_lonlats(use_dask)
        area = dyn_area.freeze(lonslats=(lons, lats), resolution=0.5, antimeridian_mode=antimeridian_mode)
        proj_str = area.crs.to_proj4()

        assert area.shape == expected_shape
        np.testing.assert_allclose(area.area_extent, expected_extents)
        for include_comp in include_proj_components:
            assert include_comp in proj_str
        for exclude_comp in exclude_proj_components:
            assert exclude_comp not in proj_str

    def test_create_area_def_dynamic_areas(self):
        """Test certain parameter combinations produce a DynamicAreaDefinition."""
        from pyresample import create_area_def as cad
        from pyresample.geometry import DynamicAreaDefinition
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        shape = (425, 850)
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

        assert isinstance(cad('ease_sh', projection, shape=shape), DynamicAreaDefinition)
        assert isinstance(cad('ease_sh', projection, area_extent=area_extent), DynamicAreaDefinition)

    def test_create_area_def_dynamic_omerc(self):
        """Test 'omerc' projections work in 'create_area_def'."""
        from pyresample import create_area_def as cad
        from pyresample.geometry import DynamicAreaDefinition
        area_def = cad('omerc_bb', {'ellps': 'WGS84', 'proj': 'omerc'})
        assert isinstance(area_def, DynamicAreaDefinition)


def _get_fake_antimeridian_lonlats(use_dask: bool) -> tuple:
    lon_min = 165
    lon_max = 195
    lons = np.arange(lon_min, lon_max, dtype=np.float64)
    lons[lons >= 180] -= 360.0
    lats = np.linspace(25.0, 35.0, lons.size, dtype=np.float64)
    if use_dask:
        lons = da.from_array(lons, chunks=lons.size // 3)
        lats = da.from_array(lats, chunks=lons.size // 3)
    return lons, lats


@pytest.fixture
def truncated_geos_area():
    """Create a truncated geostationary area."""
    projection = {'a': '6378169', 'h': '35785831', 'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                  'rf': '295.488065897014', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
    area_extent = (5567248.0742, 5570248.4773, -5570248.4773, 1393687.2705)
    width = 3712
    height = 1392
    geos_area = geometry.AreaDefinition('msg_rss', "msg_rss", "msg_rss", projection, width, height, area_extent)
    return geos_area


@pytest.fixture
def truncated_geos_area_in_space():
    """Create a truncated geostationary area."""
    projection = {'a': '6378169', 'h': '35785831', 'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                  'rf': '295.488065897014', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
    area_extent = (5575000, 5575000, 5570000, 5570000)
    width = 10
    height = 10
    geos_area = geometry.AreaDefinition('msg_rss', "msg_rss", "msg_rss", projection, width, height, area_extent)
    return geos_area


class TestGeostationaryTools:
    """Test the geostationary bbox tools."""

    def test_get_full_geostationary_bbox(self, truncated_geos_area):
        nb_points = 20
        x, y = geometry.get_full_geostationary_bounding_box_in_proj_coords(truncated_geos_area, nb_points)
        assert len(x) == nb_points
        assert len(y) == nb_points

        assert x[0] != x[-1]
        assert y[0] != y[-1]

        expected_x = np.array([-5.43062255e+06, -5.16482897e+06, -4.39346593e+06, -3.19203985e+06,
                               -1.67815466e+06, 3.32529726e-10, 1.67815466e+06, 3.19203985e+06,
                               4.39346593e+06, 5.16482897e+06, 5.43062255e+06, 5.16482897e+06,
                               4.39346593e+06, 3.19203985e+06, 1.67815466e+06, 3.32529726e-10,
                               -1.67815466e+06, -3.19203985e+06, -4.39346593e+06, -5.16482897e+06])

        expected_y = np.array([6.62789871e-10, 1.67242779e+06, 3.18114670e+06, 4.37847280e+06,
                               5.14720348e+06, 5.41209002e+06, 5.14720348e+06, 4.37847280e+06,
                               3.18114670e+06, 1.67242779e+06, -0.00000000e+00, -1.67242779e+06,
                               -3.18114670e+06, -4.37847280e+06, -5.14720348e+06, -5.41209002e+06,
                               -5.14720348e+06, -4.37847280e+06, -3.18114670e+06, -1.67242779e+06])

        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)

    def test_get_geostationary_bbox_works_with_truncated_area(self, truncated_geos_area):
        """Ensure the geostationary bbox works when truncated."""
        lon, lat = geometry.get_geostationary_bounding_box_in_lonlats(truncated_geos_area, 20)

        expected_lon = np.array(
            [-64.24072434653284, -68.69662326361153, -65.92516214783112, -60.726360278290336,
             -47.39851775032484, 9.500000000000018, 66.39851775032487, 79.72636027829033,
             84.92516214783113, 87.69662326361151, 83.24072434653286])
        expected_lat = np.array(
            [14.554922655532085, 17.768795771961937, 35.34328897185421, 52.597860701318254, 69.00533141646078,
             79.1481121862375, 69.00533141646076, 52.597860701318254, 35.34328897185421, 17.768795771961933,
             14.554922655532085])
        np.testing.assert_allclose(lon, expected_lon)
        np.testing.assert_allclose(lat, expected_lat)

    def test_get_geostationary_bbox_works_with_truncated_area_proj_coords(self, truncated_geos_area):
        """Ensure the geostationary bbox works when truncated."""
        x, y = geometry.get_geostationary_bounding_box_in_proj_coords(truncated_geos_area, 20)

        expected_x = np.array(
            [-5209128.302753595, -5164828.965702432, -4393465.934674804, -3192039.8468840676, -1678154.6586309497,
             3.325297262895822e-10, 1678154.6586309501, 3192039.846884068, 4393465.934674805, 5164828.965702432,
             5209128.302753594])
        expected_y = np.array(
            [1393687.2705, 1672427.7900638399, 3181146.6955466354, 4378472.798117005, 5147203.47659387,
             5412090.016106332, 5147203.476593869, 4378472.798117005, 3181146.695546635, 1672427.7900638392,
             1393687.2705])

        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)

    def test_get_geostationary_bbox_does_not_contain_inf(self, truncated_geos_area):
        """Ensure the geostationary bbox does not contain np.inf."""
        lon, lat = geometry.get_geostationary_bounding_box_in_lonlats(truncated_geos_area, 20)
        assert not any(np.isinf(lon))
        assert not any(np.isinf(lat))

    def test_get_geostationary_bbox_returns_empty_lonlats_in_space(self, truncated_geos_area_in_space):
        """Ensure the geostationary bbox is empty when in space."""
        lon, lat = geometry.get_geostationary_bounding_box_in_lonlats(truncated_geos_area_in_space, 20)

        assert len(lon) == 0
        assert len(lat) == 0

    def test_get_geostationary_bbox(self):
        """Get the geostationary bbox."""
        geos_area = MagicMock()
        lon_0 = 0
        proj_dict = {'a': 6378169.00,
                     'b': 6356583.80,
                     'h': 35785831.00,
                     'lon_0': lon_0,
                     'proj': 'geos'}
        geos_area.crs = CRS(proj_dict)
        geos_area.area_extent = [-5500000., -5500000., 5500000., 5500000.]

        lon, lat = geometry.get_geostationary_bounding_box_in_lonlats(geos_area, 20)
        expected_lon = np.array([-78.19662326, -75.42516215, -70.22636028,
                                 -56.89851775, 0., 56.89851775, 70.22636028,
                                 75.42516215, 78.19662326, 79.23372832, 78.19662326,
                                 75.42516215, 70.22636028, 56.89851775, 0.,
                                 -56.89851775, -70.22636028, -75.42516215, -78.19662326, -79.23372832, ])
        expected_lat = np.array([17.76879577, 35.34328897, 52.5978607,
                                 69.00533142, 79.14811219, 69.00533142, 52.5978607,
                                 35.34328897, 17.76879577, -0., -17.76879577,
                                 -35.34328897, -52.5978607, -69.00533142, -79.14811219,
                                 -69.00533142, -52.5978607, -35.34328897, -17.76879577, 0.])

        np.testing.assert_allclose(lon, expected_lon, atol=1e-07)
        np.testing.assert_allclose(lat, expected_lat, atol=1e-07)

        geos_area = MagicMock()
        lon_0 = 10
        proj_dict = {'a': 6378169.00,
                     'b': 6356583.80,
                     'h': 35785831.00,
                     'lon_0': lon_0,
                     'proj': 'geos'}
        geos_area.crs = CRS(proj_dict)
        geos_area.area_extent = [-5500000., -5500000., 5500000., 5500000.]

        lon, lat = geometry.get_geostationary_bounding_box_in_lonlats(geos_area, 20)
        np.testing.assert_allclose(lon, expected_lon + lon_0)

    def test_get_geostationary_angle_extent(self):
        """Get max geostationary angles."""
        geos_area = MagicMock()
        proj_dict = {
            'proj': 'geos',
            'sweep': 'x',
            'lon_0': -89.5,
            'a': 6378169.00,
            'b': 6356583.80,
            'h': 35785831.00,
            'units': 'm'}
        geos_area.crs = CRS(proj_dict)

        expected = (0.15185342867090912, 0.15133555510297725)
        np.testing.assert_allclose(expected,
                                   geometry.get_geostationary_angle_extent(geos_area))

        proj_dict['a'] = 1000.0
        proj_dict['b'] = 1000.0
        proj_dict['h'] = np.sqrt(2) * 1000.0 - 1000.0
        geos_area.crs = CRS(proj_dict)

        expected = (np.deg2rad(45), np.deg2rad(45))
        np.testing.assert_allclose(expected,
                                   geometry.get_geostationary_angle_extent(geos_area))

        proj_dict = {
            'proj': 'geos',
            'sweep': 'x',
            'lon_0': -89.5,
            'ellps': 'GRS80',
            'h': 35785831.00,
            'units': 'm'}
        geos_area.crs = CRS(proj_dict)
        expected = (0.15185277703584374, 0.15133971368991794)
        np.testing.assert_allclose(expected,
                                   geometry.get_geostationary_angle_extent(geos_area))


class TestCrop(unittest.TestCase):
    """Test the area helpers."""

    def test_sub_area(self):
        """Sub area slicing."""
        area = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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
        res = area[slice(20, 720), slice(100, 500)]
        np.testing.assert_allclose((-1070912.72, -669968.6399999999,
                                    129087.28000000003, 1430031.36),
                                   res.area_extent)
        self.assertEqual(res.shape, (700, 400))


def test_enclose_areas():
    """Test enclosing areas."""
    from pyresample.geometry import create_area_def, enclose_areas
    proj_dict = {'proj': 'geos', 'sweep': 'x', 'lon_0': 0, 'h': 35786023,
                 'x_0': 0, 'y_0': 0, 'ellps': 'GRS80', 'units': 'm',
                 'no_defs': None, 'type': 'crs'}
    proj_dict_alt = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0,
                     'units': 'm'}

    ar1 = create_area_def(
        "test-area",
        projection=proj_dict,
        units="m",
        area_extent=[0, 20, 100, 120],
        shape=(10, 10))

    ar2 = create_area_def(
        "test-area",
        projection=proj_dict,
        units="m",
        area_extent=[20, 40, 120, 140],
        shape=(10, 10))

    ar3 = create_area_def(
        "test-area",
        projection=proj_dict,
        units="m",
        area_extent=[20, 0, 120, 100],
        shape=(10, 10))

    ar4 = create_area_def(
        "test-area",
        projection=proj_dict_alt,
        units="m",
        area_extent=[20, 0, 120, 100],
        shape=(10, 10))

    ar5 = create_area_def(
        "test-area",
        projection=proj_dict,
        units="m",
        area_extent=[-50, -50, 50, 50],
        shape=(100, 100))

    ar_joined = enclose_areas(ar1, ar2, ar3)
    np.testing.assert_allclose(ar_joined.area_extent, [0, 0, 120, 140])
    with pytest.raises(ValueError):
        enclose_areas(ar3, ar4)
    with pytest.raises(ValueError):
        enclose_areas(ar3, ar5)
    with pytest.raises(TypeError):
        enclose_areas()


class TestAreaDefGetAreaSlices(unittest.TestCase):
    """Test AreaDefinition's get_area_slices."""

    def test_get_area_slices(self):
        """Check area slicing."""
        from pyresample import get_area_def

        # The area of our source data
        area_id = 'orig'
        area_name = 'Test area'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_def = get_area_def(area_id,
                                area_name,
                                proj_id,
                                proj_dict,
                                x_size, y_size,
                                area_extent)

        # An area that is a subset of the original one
        area_to_cover = get_area_def(
            'cover_subset',
            'Area to cover',
            'test',
            proj_dict,
            1000, 1000,
            area_extent=(area_extent[0] + 10000,
                         area_extent[1] + 10000,
                         area_extent[2] - 10000,
                         area_extent[3] - 10000))
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        self.assertEqual(slice(3, 3709, None), slice_x)
        self.assertEqual(slice(3, 3709, None), slice_y)

        # An area similar to the source data but not the same
        area_id = 'cover'
        area_name = 'Area to cover'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339261, -5567248.074173444, 5567248.074173444, 5570248.477339261)
        proj_dict = {'a': 6378169.5, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}

        area_to_cover = get_area_def(area_id,
                                     area_name,
                                     proj_id,
                                     proj_dict,
                                     x_size, y_size,
                                     area_extent)
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        self.assertEqual(slice(46, 3667, None), slice_x)
        self.assertEqual(slice(56, 3659, None), slice_y)

        area_to_cover = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                                {'a': 6378144.0,
                                                 'b': 6356759.0,
                                                 'lat_0': 50.00,
                                                 'lat_ts': 50.00,
                                                 'lon_0': 8.00,
                                                 'proj': 'stere'},
                                                10,
                                                10,
                                                [-1370912.72,
                                                 -909968.64,
                                                 1029087.28,
                                                 1490031.36])
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        self.assertEqual(slice_x, slice(1610, 2343))
        self.assertEqual(slice_y, slice(158, 515, None))

        # The same as source area, but flipped in X and Y
        area_id = 'cover'
        area_name = 'Area to cover'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (5567248.074173927, 5570248.477339745, -5570248.477339745, -5561247.267842293)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}

        area_to_cover = get_area_def(area_id,
                                     area_name,
                                     proj_id,
                                     proj_dict,
                                     x_size, y_size,
                                     area_extent)
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        self.assertEqual(slice(0, x_size, None), slice_x)
        self.assertEqual(slice(0, y_size, None), slice_y)

        # totally different area
        projections = [{"init": 'EPSG:4326'}, 'EPSG:4326']
        for projection in projections:
            area_to_cover = geometry.AreaDefinition(
                'epsg4326', 'Global equal latitude/longitude grid for global sphere',
                'epsg4326',
                projection,
                8192,
                4096,
                [-180.0, -90.0, 180.0, 90.0])

            slice_x, slice_y = area_def.get_area_slices(area_to_cover)
            assert isinstance(slice_x.start, int)
            assert isinstance(slice_y.start, int)
            self.assertEqual(slice_x, slice(46, 3667, None))
            self.assertEqual(slice_y, slice(56, 3659, None))

    def test_get_area_slices_nongeos(self):
        """Check area slicing for non-geos projections."""
        from pyresample import get_area_def

        # The area of our source data
        area_id = 'orig'
        area_name = 'Test area'
        proj_id = 'test'
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'lat_1': 25.,
                     'lat_2': 25., 'lon_0': 0.0, 'proj': 'lcc', 'units': 'm'}
        area_def = get_area_def(area_id,
                                area_name,
                                proj_id,
                                proj_dict,
                                x_size, y_size,
                                area_extent)

        # An area that is a subset of the original one
        area_to_cover = get_area_def(
            'cover_subset',
            'Area to cover',
            'test',
            proj_dict,
            1000, 1000,
            area_extent=(area_extent[0] + 10000,
                         area_extent[1] + 10000,
                         area_extent[2] - 10000,
                         area_extent[3] - 10000))
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        self.assertEqual(slice(3, 3709, None), slice_x)
        self.assertEqual(slice(3, 3709, None), slice_y)

    def test_on_flipped_geos_area(self):
        """Test get_area_slices on flipped areas."""
        from pyresample.geometry import AreaDefinition
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        expected_slice_lines = slice(60, 91)
        expected_slice_cols = slice(90, 100)
        cropped_area = src_area[expected_slice_lines, expected_slice_cols]
        slice_cols, slice_lines = src_area.get_area_slices(cropped_area)
        assert slice_lines == expected_slice_lines
        assert slice_cols == expected_slice_cols

        expected_slice_cols = slice(30, 61)
        cropped_area = src_area[expected_slice_lines, expected_slice_cols]
        slice_cols, slice_lines = src_area.get_area_slices(cropped_area)
        assert slice_lines == expected_slice_lines
        assert slice_cols == expected_slice_cols


class TestBoundary(unittest.TestCase):
    """Test 'boundary' method for <area_type>Definition classes."""

    def test_polar_south_pole_projection(self):
        """Test boundary for polar projection around the south pole."""
        # Define polar projection
        proj_dict_polar_sh = {
            'proj_id': "polar_sh_projection",
            "area_id": 'polar_sh_projection',
            "description": 'Antarctic EASE grid',
            # projection : 'EPSG:3409',
            "projection": {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            "width": 2,
            "height": 2,
            "area_extent": (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625),
        }
        # Define AreaDefintion and retrieve AreaBoundary
        areadef = geometry.AreaDefinition(**proj_dict_polar_sh)
        boundary = areadef.boundary(force_clockwise=False)

        # Check boundary shape
        height, width = areadef.shape
        n_vertices = (width - 1) * 2 + (height - 1) * 2
        assert boundary.vertices.shape == (n_vertices, 2)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[-45., -55.61313895],
                                      [45., -55.61313895],
                                      [135., -55.61313895],
                                      [-135., -55.61313895]])
        assert np.allclose(expected_vertices, boundary.vertices)

    def test_nort_pole_projection(self):
        """Test boundary for polar projection around the nort pole."""
        # Define polar projection
        proj_dict_polar_nh = {
            'proj_id': "polar_nh_projection",
            "area_id": 'polar_nh_projection',
            "description": 'Artic EASE grid',
            "projection": {'proj': 'laea', 'lat_0': 90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            "width": 2,
            "height": 2,
            "area_extent": (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625),
        }
        # Define AreaDefintion and retrieve AreaBoundary
        areadef = geometry.AreaDefinition(**proj_dict_polar_nh)
        boundary = areadef.boundary(force_clockwise=False)

        # Check boundary shape
        height, width = areadef.shape
        n_vertices = (width - 1) * 2 + (height - 1) * 2
        assert boundary.vertices.shape == (n_vertices, 2)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[-135., 55.61313895],
                                      [135., 55.61313895],
                                      [45., 55.61313895],
                                      [-45., 55.61313895]])
        assert np.allclose(expected_vertices, boundary.vertices)

    def test_geostationary_projection(self):
        """Test boundary for geostationary projection."""
        # Define geostationary projection
        proj_dict_geostationary = {
            'proj_id': "dummy_geo_projection",
            "area_id": 'dummy_geo_projection',
            "description": 'geostationary projection',
            "projection": {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.00, 'lon_0': 0, 'proj': 'geos'},
            "area_extent": (-5500000., -5500000., 5500000., 5500000.),
            "width": 100,
            "height": 100,
        }
        # Define AreaDefintion and retrieve AreaBoundary
        areadef = geometry.AreaDefinition(**proj_dict_geostationary)

        # Check default boundary shape
        default_n_vertices = 50
        boundary = areadef.boundary(frequency=None)
        assert boundary.vertices.shape == (default_n_vertices, 2)

        # Check minimum boundary vertices
        n_vertices = 3
        minimum_n_vertices = 4
        boundary = areadef.boundary(frequency=n_vertices)
        assert boundary.vertices.shape == (minimum_n_vertices, 2)

        # Check odd frequency number
        # - Rounded to the sequent even number (to construct the sides)
        n_odd_vertices = 5
        boundary = areadef.boundary(frequency=n_odd_vertices)
        assert boundary.vertices.shape == (n_odd_vertices + 1, 2)

        # Check boundary vertices
        n_vertices = 10
        boundary = areadef.boundary(frequency=n_vertices, force_clockwise=False)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[-7.54251621e+01, 3.53432890e+01],
                                      [-5.68985178e+01, 6.90053314e+01],
                                      [5.68985178e+01, 6.90053314e+01],
                                      [7.54251621e+01, 3.53432890e+01],
                                      [7.92337283e+01, -0.00000000e+00],
                                      [7.54251621e+01, -3.53432890e+01],
                                      [5.68985178e+01, -6.90053314e+01],
                                      [-5.68985178e+01, -6.90053314e+01],
                                      [-7.54251621e+01, -3.53432890e+01],
                                      [-7.92337283e+01, 6.94302533e-15]])
        np.testing.assert_allclose(expected_vertices, boundary.vertices)

    def test_global_platee_caree_projection(self):
        """Test boundary for global platee caree projection."""
        # Define global projection
        proj_dict_global_wgs84 = {
            'proj_id': "epsg4326",
            'area_id': 'epsg4326',
            'description': 'Global equal latitude/longitude grid for global sphere',
            "projection": 'EPSG:4326',
            "width": 4,
            "height": 4,
            "area_extent": (-180.0, -90.0, 180.0, 90.0),
        }
        # Define AreaDefintion and retrieve AreaBoundary
        areadef = geometry.AreaDefinition(**proj_dict_global_wgs84)
        boundary = areadef.boundary(force_clockwise=False)

        # Check boundary shape
        height, width = areadef.shape
        n_vertices = (width - 1) * 2 + (height - 1) * 2
        assert boundary.vertices.shape == (n_vertices, 2)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[-135., 67.5],
                                      [-45., 67.5],
                                      [45., 67.5],
                                      [135., 67.5],
                                      [135., 22.5],
                                      [135., -22.5],
                                      [135., -67.5],
                                      [45., -67.5],
                                      [-45., -67.5],
                                      [-135., -67.5],
                                      [-135., -22.5],
                                      [-135., 22.5]])
        assert np.allclose(expected_vertices, boundary.vertices)

    def test_minimal_global_platee_caree_projection(self):
        """Test boundary for global platee caree projection."""
        # Define minimal global projection
        proj_dict_global_wgs84 = {
            'proj_id': "epsg4326",
            'area_id': 'epsg4326',
            'description': 'Global equal latitude/longitude grid for global sphere',
            "projection": 'EPSG:4326',
            "width": 2,
            "height": 2,
            "area_extent": (-180.0, -90.0, 180.0, 90.0),
        }

        # Define AreaDefintion and retrieve AreaBoundary
        areadef = geometry.AreaDefinition(**proj_dict_global_wgs84)
        boundary = areadef.boundary(force_clockwise=False)

        # Check boundary shape
        height, width = areadef.shape
        n_vertices = (width - 1) * 2 + (height - 1) * 2
        assert boundary.vertices.shape == (n_vertices, 2)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[-90., 45.],
                                      [90., 45.],
                                      [90., -45.],
                                      [-90., -45.]])
        assert np.allclose(expected_vertices, boundary.vertices)

    def test_local_area_projection(self):
        """Test local area projection in meter."""
        # Define ch1903 projection (eastings, northings)
        proj_dict_ch1903 = {
            'proj_id': "swiss_area",
            'area_id': 'swiss_area',
            'description': 'Swiss CH1903+ / LV95',
            "projection": 'EPSG:2056',
            "width": 2,
            "height": 2,
            "area_extent": (2_600_000.0, 1_050_000, 2_800_000.0, 1_170_000),
        }

        # Define AreaDefintion and retrieve AreaBoundary
        areadef = geometry.AreaDefinition(**proj_dict_ch1903)
        boundary = areadef.boundary(force_clockwise=False)

        # Check boundary shape
        height, width = areadef.shape
        n_vertices = (width - 1) * 2 + (height - 1) * 2
        assert boundary.vertices.shape == (n_vertices, 2)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[8.08993639, 46.41074744],
                                      [9.39028624, 46.39582417],
                                      [9.37106733, 45.85619242],
                                      [8.08352612, 45.87097006]])
        assert np.allclose(expected_vertices, boundary.vertices)

    def test_swath_definition(self):
        """Test boundary for swath definition."""
        lons = np.array([[1.2, 1.3, 1.4, 1.5],
                         [1.2, 1.3, 1.4, 1.5]])
        lats = np.array([[65.9, 65.86, 65.82, 65.78],
                         [65.89, 65.86, 65.82, 65.78]])

        # Define SwathDefinition and retrieve AreaBoundary
        swath_def = SwathDefinition(lons, lats)
        boundary = swath_def.boundary(force_clockwise=False)

        # Check boundary shape
        height, width = swath_def.shape
        n_vertices = (width - 1) * 2 + (height - 1) * 2
        assert boundary.vertices.shape == (n_vertices, 2)

        # Check boundary vertices is in correct order
        expected_vertices = np.array([[1.2, 65.9],
                                      [1.3, 65.86],
                                      [1.4, 65.82],
                                      [1.5, 65.78],
                                      [1.5, 65.78],
                                      [1.4, 65.82],
                                      [1.3, 65.86],
                                      [1.2, 65.89]])
        assert np.allclose(expected_vertices, boundary.vertices)
