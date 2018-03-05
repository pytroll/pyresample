from __future__ import with_statement

import random
import sys

import numpy as np

from pyresample import geo_filter, geometry
from pyresample.geometry import (IncompatibleAreas,
                                 combine_area_extents_vertical,
                                 concatenate_area_defs)
from pyresample.test.utils import catch_warnings

try:
    from unittest.mock import MagicMock, patch
except ImportError:
    # separate mock package py<3.3
    from mock import MagicMock, patch

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class Test(unittest.TestCase):

    """Unit testing the geometry and geo_filter modules"""

    def assert_raises(self, exception, call_able, *args):
        """assertRaises() has changed from py2.6 to 2.7! Here is an attempt to
        cover both"""
        import sys
        if sys.version_info < (2, 7):
            self.assertRaises(exception, call_able, *args)
        else:
            with self.assertRaises(exception):
                call_able(*args)

    def test_lonlat_precomp(self):
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
        lons, lats = area_def.get_lonlats()
        lon, lat = area_def.get_lonlat(400, 400)
        self.assertAlmostEqual(lon, 5.5028467120975835,
                               msg='lon retrieval from precomputated grid failed')
        self.assertAlmostEqual(lat, 52.566998432390619,
                               msg='lat retrieval from precomputated grid failed')

    def test_cartesian(self):
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
        cart_coords = area_def.get_cartesian_coords()
        exp = 5872039989466.8457031
        self.assertTrue((cart_coords.sum() - exp) < 1e-7 * exp,
                        msg='Calculation of cartesian coordinates failed')

    def test_base_type(self):
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
        arr = np.array([1.2, 1.3, 1.4, 1.5])
        if sys.byteorder == 'little':
            # arr.view(np.uint8)
            reference = np.array([51,  51,  51,  51,  51,  51, 243,
                                  63, 205, 204, 204, 204, 204,
                                  204, 244,  63, 102, 102, 102, 102,
                                  102, 102, 246,  63,   0,   0,
                                  0,   0,   0,   0, 248,  63],
                                 dtype=np.uint8)
        else:
            # on le machines use arr.byteswap().view(np.uint8)
            reference = np.array([63, 243,  51,  51,  51,  51,  51,
                                  51,  63, 244, 204, 204, 204,
                                  204, 204, 205,  63, 246, 102, 102,
                                  102, 102, 102, 102,  63, 248,
                                  0,   0,   0,   0,   0,   0],
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
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)

        self.assertIsInstance(hash(swath_def), int)

        try:
            import dask.array as da
        except ImportError:
            print("Not testing with dask arrays")
        else:
            dalons = da.from_array(lons, chunks=1000)
            dalats = da.from_array(lats, chunks=1000)
            swath_def = geometry.SwathDefinition(dalons, dalats)

            self.assertIsInstance(hash(swath_def), int)

        try:
            import xarray as xr
        except ImportError:
            print("Not testing with xarray")
        else:
            xrlons = xr.DataArray(lons)
            xrlats = xr.DataArray(lats)
            swath_def = geometry.SwathDefinition(xrlons, xrlats)

            self.assertIsInstance(hash(swath_def), int)

        try:
            import xarray as xr
            import dask.array as da
        except ImportError:
            print("Not testing with xarrays and dask arrays")
        else:
            xrlons = xr.DataArray(da.from_array(lons, chunks=1000))
            xrlats = xr.DataArray(da.from_array(lats, chunks=1000))
            swath_def = geometry.SwathDefinition(xrlons, xrlats)

            self.assertIsInstance(hash(swath_def), int)

        lons = np.ma.array([1.2, 1.3, 1.4, 1.5])
        lats = np.ma.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)

        self.assertIsInstance(hash(swath_def), int)

    def test_area_equal(self):
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

    def test_swath_equal_area(self):
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
        self.assertTrue(np.array_equal(swath_def_f.lons[:], expected_lons)
                        and np.array_equal(swath_def_f.lats[:], expected_lats),
                        'Failed finding grid filtering lon lats')

    def test_grid_filter2D(self):
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
        self.assertTrue(np.array_equal(swath_def_f.lons[:], expected_lons)
                        and np.array_equal(swath_def_f.lats[:], expected_lats),
                        'Failed finding 2D grid filtering lon lats')

    def test_boundary(self):
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
        area_def = geometry.AreaDefinition('', '', '',
                                           {'proj': 'latlong'},
                                           360, 180,
                                           [-180, -90, 180, 90])
        lons, lats = area_def.get_lonlats()
        self.assertEqual(lons[0, 0], -179.5)
        self.assertEqual(lats[0, 0], 89.5)

    def test_lonlat2colrow(self):
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
        cols__, rows__ = area.lonlat2colrow(longitudes, latitudes)

        # test arrays
        cols_expects = np.array([2304, 2040])
        rows_expects = np.array([186, 341])
        self.assertTrue((cols__ == cols_expects).all())
        self.assertTrue((rows__ == rows_expects).all())

        # test scalars
        lon, lat = (-8.125547604568746, -14.345524111874646)
        self.assertTrue(area.lonlat2colrow(lon, lat) == (1567, 2375))

    def test_colrow2lonlat(self):

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

    def test_get_xy_from_lonlat(self):
        """Test the function get_xy_from_lonlat"""
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
        import pyproj
        p__ = pyproj.Proj(proj_dict)
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
        self.assert_raises(ValueError, area_def.get_xy_from_lonlat, lon, lat)
        self.assert_raises(ValueError, area_def.get_xy_from_lonlat, 0., 0.)

        # Test getting arrays back:
        lons = [lon_ll + eps_lonlat, lon_ur - eps_lonlat]
        lats = [lat_ll + eps_lonlat, lat_ur - eps_lonlat]
        x__, y__ = area_def.get_xy_from_lonlat(lons, lats)

        x_expects = np.array([0, 1])
        y_expects = np.array([1, 0])
        self.assertTrue((x__.data == x_expects).all())
        self.assertTrue((y__.data == y_expects).all())


def assert_np_dict_allclose(dict1, dict2):

    assert set(dict1.keys()) == set(dict2.keys())
    for key, val in dict1.items():
        try:
            np.testing.assert_allclose(val, dict2[key])
        except TypeError:
            assert(val == dict2[key])


class TestSwathDefinition(unittest.TestCase):
    """Test the SwathDefinition."""

    def test_swath(self):
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        swath_def = geometry.SwathDefinition(lons1, lats1)

        lons2, lats2 = swath_def.get_lonlats()

        self.assertFalse(id(lons1) != id(lons2) or id(lats1) != id(lats2),
                         msg='Caching of swath coordinates failed')

    def test_concat_1d(self):
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
        proj_dict = {'no_rot': True, 'lonc': 5.340645620216994,
                     'ellps': 'WGS84', 'proj': 'omerc',
                     'alpha': 19.022450179020247, 'lat_0': 60.7420043944989}
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

        np.testing.assert_allclose(lons, [-90.67900085, 79.11000061,  81.26400757,
                                          81.26400757, 29.67200089, 10.26000023,
                                          10.26000023, -5.10700035, -21.52500153,
                                          -21.52500153, -21.56500053, -90.67900085])
        np.testing.assert_allclose(lats, [85.23900604, 80.84000397, 67.07600403,
                                          67.07600403, 54.14700317, 30.54700089,
                                          30.54700089, 34.0850029, 35.58000183,
                                          35.58000183, 62.25600433,  85.23900604])

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
        lats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                         [80.84000396728516, 60.74200439453125, 34.08500289916992],
                         [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T

        lons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                         [79.11000061035156, 7.284000396728516, -5.107000350952148],
                         [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T

        area = geometry.SwathDefinition(lons, lats)

        res = area.compute_optimal_bb_area({'proj': 'omerc', 'ellps': 'WGS84'})

        np.testing.assert_allclose(res.area_extent, (2286629.731529,
                                                     -2359693.817959,
                                                     11729881.856072,
                                                     2437001.523925))
        proj_dict = {'no_rot': True, 'lonc': 5.340645620216994,
                     'ellps': 'WGS84', 'proj': 'omerc',
                     'alpha': 19.022450179020247, 'lat_0': 60.7420043944989}
        assert_np_dict_allclose(res.proj_dict, proj_dict)
        self.assertEqual(res.shape, (3, 3))


class TestStackedAreaDefinition(unittest.TestCase):
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
        self.assertEqual(len(adef.defs), 1)
        self.assertTupleEqual(adef.defs[0].area_extent,
                              (3738502.0095458371, 4179561.259167064,
                               -1830246.0673044831, 3251436.5796920112))

        # same

        area3 = geometry.AreaDefinition("area3", 'area3', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3251436.5796920112,
                                         -1830246.0673044831, 2787374.2399544837))
        adef.append(area3)
        self.assertEqual(len(adef.defs), 1)
        self.assertTupleEqual(adef.defs[0].area_extent,
                              (3738502.0095458371, 4179561.259167064,
                               -1830246.0673044831, 2787374.2399544837))

        self.assertIsInstance(adef.squeeze(), geometry.AreaDefinition)

        # transition
        area4 = geometry.AreaDefinition("area4", 'area4', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (5567747.7409681147, 2787374.2399544837,
                                         -1000.3358822065015, 2323311.9002169576))

        adef.append(area4)
        self.assertEqual(len(adef.defs), 2)
        self.assertTupleEqual(adef.defs[-1].area_extent,
                              (5567747.7409681147, 2787374.2399544837,
                               -1000.3358822065015, 2323311.9002169576))

        self.assertEqual(adef.y_size, 4 * 464)
        self.assertIsInstance(adef.squeeze(), geometry.StackedAreaDefinition)

        adef2 = geometry.StackedAreaDefinition()
        self.assertEqual(len(adef2.defs), 0)

        adef2.append(adef)
        self.assertEqual(len(adef2.defs), 2)
        self.assertTupleEqual(adef2.defs[-1].area_extent,
                              (5567747.7409681147, 2787374.2399544837,
                               -1000.3358822065015, 2323311.9002169576))

        self.assertEqual(adef2.y_size, 4 * 464)

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
        self.assertEqual(len(final_area.defs), 2)
        lons, lats = final_area.get_lonlats()
        lons0, lats0 = final_area.defs[0].get_lonlats()
        lons1, lats1 = final_area.defs[1].get_lonlats()
        np.testing.assert_allclose(lons[:464, :], lons0)
        np.testing.assert_allclose(lons[464:, :], lons1)
        np.testing.assert_allclose(lats[:464, :], lats0)
        np.testing.assert_allclose(lats[464:, :], lats1)

    def test_combine_area_extents(self):
        """Test combination of area extents."""
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 6, 3, 2)
        res = combine_area_extents_vertical(area1, area2)
        self.assertListEqual(res, [1, 6, 3, 4])

        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        res = combine_area_extents_vertical(area1, area2)
        self.assertListEqual(res, [1, 2, 3, 6])

    def test_append_area_defs_fail(self):
        """Fail appending areas."""
        area1 = MagicMock()
        area1.proj_dict = {"proj": 'A'}
        area1.x_size = 4
        area1.y_size = 5
        area2 = MagicMock()
        area2.proj_dict = {'proj': 'B'}
        area2.x_size = 4
        area2.y_size = 6
        # res = combine_area_extents_vertical(area1, area2)
        self.assertRaises(IncompatibleAreas,
                          concatenate_area_defs, area1, area2)

    @patch('pyresample.geometry.AreaDefinition')
    def test_append_area_defs(self, adef):
        """Test appending area definitions."""
        x_size = random.randrange(6425)
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area1.proj_dict = {"proj": 'A'}
        area1.y_size = random.randrange(6425)
        area1.x_size = x_size

        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        area2.proj_dict = {"proj": 'A'}
        area2.y_size = random.randrange(6425)
        area2.x_size = x_size

        concatenate_area_defs(area1, area2)
        area_extent = [1, 2, 3, 6]
        y_size = area1.y_size + area2.y_size
        adef.assert_called_once_with(area1.area_id, area1.name, area1.proj_id,
                                     area1.proj_dict, area1.x_size, y_size,
                                     area_extent)


class TestDynamicAreaDefinition(unittest.TestCase):
    """Test the DynamicAreaDefinition class."""

    def test_freeze(self):
        """Test freezing the area."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'laea'})
        lons = [10, 10, 22, 22]
        lats = [50, 66, 66, 50]
        result = area.freeze((lons, lats),
                             resolution=3000,
                             proj_info={'lon0': 16, 'lat0': 58})

        np.testing.assert_allclose(result.area_extent, (538546.7274949469,
                                                        5380808.879250369,
                                                        1724415.6519203288,
                                                        6998895.701001488))
        self.assertEqual(result.proj_dict['lon0'], 16)
        self.assertEqual(result.proj_dict['lat0'], 58)
        self.assertEqual(result.x_size, 395)
        self.assertEqual(result.y_size, 539)

    def test_freeze_with_bb(self):
        """Test freezing the area with bounding box computation."""
        # area = geometry.DynamicAreaDefinition('test_area', 'A test area',
        #                                       {'proj': 'omerc'},
        #                                       optimize_projection=False)
        # lons = [[10, 12.1, 14.2, 16.3],
        #         [10, 12, 14, 16],
        #         [10, 11.9, 13.8, 15.7]]
        # lats = [[66, 67, 68, 69.],
        #         [58, 59, 60, 61],
        #         [50, 51, 52, 53]]
        # sdef = geometry.SwathDefinition(lons, lats)
        # result = area.freeze(sdef,
        #                      resolution=1000)
        # self.assertTupleEqual(result.area_extent, (5578795.1654752363,
        #                                            -270848.61872542271,
        #                                            7694893.3964453982,
        #                                            126974.877141819))
        # self.assertEqual(result.x_size, 2116)
        # self.assertEqual(result.y_size, 398)

        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'omerc'},
                                              optimize_projection=True)
        lons = [[10, 12.1, 14.2, 16.3],
                [10, 12, 14, 16],
                [10, 11.9, 13.8, 15.7]]
        lats = [[66, 67, 68, 69.],
                [58, 59, 60, 61],
                [50, 51, 52, 53]]
        sdef = geometry.SwathDefinition(lons, lats)
        result = area.freeze(sdef,
                             resolution=1000)
        np.testing.assert_allclose(result.area_extent, (5050520.6077326955,
                                                        -336485.86803662963,
                                                        8223167.9541879389,
                                                        192612.12645302597))
        self.assertEqual(result.x_size, 3)
        self.assertEqual(result.y_size, 4)

    def test_compute_domain(self):
        """Test computing size and area extent."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'laea'})
        corners = [1, 1, 9, 9]
        self.assertRaises(ValueError, area.compute_domain, corners, 1, 1)

        area_extent, x_size, y_size = area.compute_domain(corners, size=(5, 5))
        self.assertTupleEqual(area_extent, (0, 0, 10, 10))
        self.assertEqual(x_size, 5)
        self.assertEqual(y_size, 5)

        area_extent, x_size, y_size = area.compute_domain(corners, resolution=2)
        self.assertTupleEqual(area_extent, (0, 0, 10, 10))
        self.assertEqual(x_size, 5)
        self.assertEqual(y_size, 5)


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))
    mysuite.addTest(loader.loadTestsFromTestCase(TestStackedAreaDefinition))
    mysuite.addTest(loader.loadTestsFromTestCase(TestDynamicAreaDefinition))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSwathDefinition))

    return mysuite


if __name__ == '__main__':
    unittest.main()
