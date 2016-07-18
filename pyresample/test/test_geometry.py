from __future__ import with_statement

import sys
import numpy as np
from pyresample.test.utils import catch_warnings
from pyresample import geometry, geo_filter

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
                                                1490031.3600000001],
                                            lons=lons, lats=lats)
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

    def test_base_lat_invalid(self):

        lons = np.arange(-135., +135, 20.)
        lats = np.ones_like(lons) * 70.
        lats[0] = -95
        lats[1] = +95
        self.assertRaises(
            ValueError, geometry.BaseDefinition, lons=lons, lats=lats)

    def test_base_lon_wrapping(self):

        lons1 = np.arange(-135., +135, 50.)
        lats = np.ones_like(lons1) * 70.

        with catch_warnings() as w1:
            base_def1 = geometry.BaseDefinition(lons1, lats)
            self.assertFalse(
                len(w1) != 0, 'Got warning <%s>, but was not expecting one' % w1)

        lons2 = np.where(lons1 < 0, lons1 + 360, lons1)
        with catch_warnings() as w2:
            base_def2 = geometry.BaseDefinition(lons2, lats)
            self.assertFalse(
                len(w2) != 1, 'Failed to trigger a warning on longitude wrapping')
            self.assertFalse(('-180:+180' not in str(w2[0].message)),
                             'Failed to trigger correct warning about longitude wrapping')

        self.assertFalse(
            base_def1 != base_def2, 'longitude wrapping to [-180:+180] did not work')

        with catch_warnings() as w3:
            base_def3 = geometry.BaseDefinition(None, None)
            self.assertFalse(
                len(w3) != 0, 'Got warning <%s>, but was not expecting one' % w3)

        self.assert_raises(ValueError, base_def3.get_lonlats)

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
        with catch_warnings() as w:
            basedef = geometry.BaseDefinition(lons2, lats)

        lons, _ = basedef.get_lonlats()
        self.assertEqual(lons.dtype, lons2.dtype,
                         "BaseDefinition did not maintain dtype of longitudes (in:%s out:%s)" %
                         (lons2.dtype, lons.dtype,))

        lons2_ints = lons2.astype('int')
        with catch_warnings() as w:
            basedef = geometry.BaseDefinition(lons2_ints, lats)

        lons, _ = basedef.get_lonlats()
        self.assertEqual(lons.dtype, lons2_ints.dtype,
                         "BaseDefinition did not maintain dtype of longitudes (in:%s out:%s)" %
                         (lons2_ints.dtype, lons.dtype,))

    def test_swath(self):
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        swath_def = geometry.SwathDefinition(lons1, lats1)

        lons2, lats2 = swath_def.get_lonlats()

        self.assertFalse(id(lons1) != id(lons2) or id(lats1) != id(lats2),
                         msg='Caching of swath coordinates failed')

    def test_swath_wrap(self):
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats1 = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))

        lons1 += 180.
        with catch_warnings() as w1:
            swath_def = geometry.BaseDefinition(lons1, lats1)
            self.assertFalse(
                len(w1) != 1, 'Failed to trigger a warning on longitude wrapping')
            self.assertFalse(('-180:+180' not in str(w1[0].message)),
                             'Failed to trigger correct warning about longitude wrapping')

        lons2, lats2 = swath_def.get_lonlats()

        self.assertTrue(id(lons1) != id(lons2),
                        msg='Caching of swath coordinates failed with longitude wrapping')

        self.assertTrue(lons2.min() > -180 and lons2.max() < 180,
                        'Wrapping of longitudes failed for SwathDefinition')

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

    def test_swath_equal(self):
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)
        swath_def2 = geometry.SwathDefinition(lons, lats)
        self.assertFalse(
            swath_def != swath_def2, 'swath_defs are not equal as expected')

    def test_swath_not_equal(self):
        lats1 = np.array([65.9, 65.86, 65.82, 65.78])
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats2 = np.array([65.91, 65.85, 65.80, 65.75])
        swath_def = geometry.SwathDefinition(lons, lats1)
        swath_def2 = geometry.SwathDefinition(lons, lats2)
        self.assertFalse(
            swath_def == swath_def2, 'swath_defs are not expected to be equal')

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
        proj_x_boundary, proj_y_boundary = area_def.proj_x_coords, area_def.proj_y_coords
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
        area_extent = [-5570248.477339261, -5567248.074173444, 5567248.074173444, 5570248.477339261]
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
        self.assertTrue(area.lonlat2colrow(lon,lat) == (1567, 2375))


    def test_colrow2lonlat(self):

        from pyresample import utils
        area_id = 'meteosat_0deg'
        area_name = 'Meteosat 0 degree Service'
        proj_id = 'geos0'
        x_size = 3712
        y_size = 3712
        area_extent = [-5570248.477339261, -5567248.074173444, 5567248.074173444, 5570248.477339261]
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
        self.assertTrue(np.allclose(lons__, lon_expects, rtol = 0, atol = 1e-7))
        self.assertTrue(np.allclose(lats__, lat_expects, rtol = 0, atol = 1e-7))

        # test scalars
        lon__, lat__ = area.colrow2lonlat(1567, 2375)
        lon_expect = -8.125547604568746
        lat_expect = -14.345524111874646
        self.assertTrue(np.allclose(lon__, lon_expect, rtol = 0, atol = 1e-7))
        self.assertTrue(np.allclose(lat__, lat_expect, rtol = 0, atol = 1e-7))


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


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite


if __name__ == '__main__':
    unittest.main()
