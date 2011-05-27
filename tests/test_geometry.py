import unittest

import numpy as np

from pyresample import geometry, geo_filter


def tmp(f):
    f.tmp = True
    return f

class Test(unittest.TestCase):
    
    def test_lonlat_caching(self):
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
        
        lons1, lats1 = area_def.get_lonlats()
        lons2 = area_def.lons[:]
        lats2 = area_def.lats[:]
        lons3, lats3 = area_def.get_lonlats()
        self.failUnless(np.array_equal(lons1, lons2) and np.array_equal(lats1, lats2), 
                        'method and property lon lat calculation does not give same result')
        self.failIf(id(lons3) != id(lons2) or id(lats3) != id(lats2), 
                    'Caching of lon lat arrays does not work')
        
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
        self.failUnlessAlmostEqual(lon, 5.5028467120975835, 
                                   msg='lon retrieval from precomputated grid failed')
        self.failUnlessAlmostEqual(lat, 52.566998432390619, 
                                   msg='lat retrieval from precomputated grid failed')
        
    @tmp
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
        self.failUnlessAlmostEqual(cart_coords.sum(), 5872039989466.8457031,
                                   places=1,
                                   msg='Calculation of cartesian coordinates failed')
        
    def test_cartesian_caching(self):
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
        cart_coords1 = area_def.cartesian_coords[:]
        cart_coords2 = area_def.get_cartesian_coords()
        self.failIf(id(cart_coords1) != id(cart_coords2), 
                    msg='Caching of cartesian coordinates failed')
        
    
    def test_swath(self):
        lons1 = np.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats1 = np.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        
        swath_def = geometry.SwathDefinition(lons1, lats1)
        
        lons2, lats2 = swath_def.get_lonlats()
        
        self.failIf(id(lons1) != id(lons2) or id(lats1) != id(lats2), 
                    msg='Caching of swath coordinates failed')
        
    def test_slice_caching(self):
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
        
        cart_coords1 = area_def.cartesian_coords[200:350, 400:500]
        cart_coords2 = area_def.cartesian_coords[200:350, 400:500]
        
        self.failIf(id(cart_coords1) != id(cart_coords2), 
                    msg='Caching of sliced cartesian coordinates failed')
        
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
        self.failIf(area_def != area_def2, 'area_defs are not equal as expected')
         
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
        self.failIf(area_def == msg_area, 'area_defs are not expected to be equal')
       
    def test_swath_equal(self):
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = geometry.SwathDefinition(lons, lats)
        swath_def2 = geometry.SwathDefinition(lons, lats)
        self.failIf(swath_def != swath_def2, 'swath_defs are not equal as expected')
        
    def test_swath_not_equal(self):
        lats1 = np.array([65.9, 65.86, 65.82, 65.78])
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats2 = np.array([65.91, 65.85, 65.80, 65.75])
        swath_def = geometry.SwathDefinition(lons, lats1)
        swath_def2 = geometry.SwathDefinition(lons, lats2)
        self.failIf(swath_def == swath_def2, 'swath_defs are not expected to be equal')

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

        self.failIf(swath_def != area_def, "swath_def and area_def should be equal")

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

        self.failIf(area_def != swath_def, "swath_def and area_def should be equal")

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

        self.failIf(swath_def == area_def, "swath_def and area_def should be different")

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

        self.failIf(area_def == swath_def, "swath_def and area_def should be different")
        
    def test_concat_1d(self):
        lons1 = np.array([1, 2, 3])
        lats1 = np.array([1, 2, 3])
        lons2 = np.array([4, 5, 6])
        lats2 = np.array([4, 5, 6])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def_concat = swath_def1.concatenate(swath_def2) 
        expected = np.array([1, 2, 3, 4, 5, 6])
        self.failUnless(np.array_equal(swath_def_concat.lons.data, expected) and 
                        np.array_equal(swath_def_concat.lons.data, expected), 
                        'Failed to concatenate 1D swaths')

    def test_concat_2d(self):
        lons1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lats1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lons2 = np.array([[4, 5, 6], [6, 7, 8]])
        lats2 = np.array([[4, 5, 6], [6, 7, 8]])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def_concat = swath_def1.concatenate(swath_def2) 
        expected = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 5, 6], [6, 7, 8]])
        self.failUnless(np.array_equal(swath_def_concat.lons.data, expected) and 
                        np.array_equal(swath_def_concat.lons.data, expected), 
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
        self.failUnless(np.array_equal(swath_def1.lons.data, expected) and 
                        np.array_equal(swath_def1.lons.data, expected), 
                        'Failed to append 1D swaths')

    def test_append_2d(self):
        lons1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lats1 = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        lons2 = np.array([[4, 5, 6], [6, 7, 8]])
        lats2 = np.array([[4, 5, 6], [6, 7, 8]])
        swath_def1 = geometry.SwathDefinition(lons1, lats1)
        swath_def2 = geometry.SwathDefinition(lons2, lats2)
        swath_def1.append(swath_def2) 
        expected = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 5, 6], [6, 7, 8]])
        self.failUnless(np.array_equal(swath_def1.lons.data, expected) and 
                        np.array_equal(swath_def1.lons.data, expected), 
                        'Failed to append 2D swaths')

    def test_grid_filter_valid(self):
        lons = np.array([-170, -30, 30, 170])
        lats = np.array([20, -40, 50, -80])
        swath_def = geometry.SwathDefinition(lons, lats)
        filter_area = geometry.AreaDefinition('test', 'test', 'test', 
                                              {'proj' : 'eqc', 'lon_0' : 0.0, 'lat_0' : 0.0},
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
        self.failUnless(np.array_equal(valid_index, expected), 'Failed to find grid filter')
    
    def test_grid_filter(self):
        lons = np.array([-170, -30, 30, 170])
        lats = np.array([20, -40, 50, -80])
        swath_def = geometry.SwathDefinition(lons, lats)
        data = np.array([1, 2, 3, 4])
        filter_area = geometry.AreaDefinition('test', 'test', 'test', 
                                              {'proj' : 'eqc', 'lon_0' : 0.0, 'lat_0' : 0.0},
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
        self.failUnless(np.array_equal(data_f, expected), 'Failed grid filtering data')
        expected_lons = np.array([-170, 170])
        expected_lats = np.array([20, -80])
        self.failUnless(np.array_equal(swath_def_f.lons[:], expected_lons) 
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
                                              {'proj' : 'eqc', 'lon_0' : 0.0, 'lat_0' : 0.0},
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
        self.failUnless(np.array_equal(data_f, expected), 'Failed 2D grid filtering data')
        expected_lons = np.array([-170, 170, -170, 170])
        expected_lats = np.array([20, -80, 25, -75])
        self.failUnless(np.array_equal(swath_def_f.lons[:], expected_lons) 
                        and np.array_equal(swath_def_f.lats[:], expected_lats), 
                        'Failed finding 2D grid filtering lon lats')
    
    @tmp    
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
        proj_x_boundary = area_def.projection_x_coords.boundary
        expected = np.array([-1250912.72, -1010912.72, -770912.72, 
                             -530912.72, -290912.72, -50912.72, 189087.28, 
                             429087.28, 669087.28, 909087.28])
        self.failUnless(np.allclose(proj_x_boundary.side1, expected), 
                        'Failed to find proejction boundary')
        
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
        self.failUnlessAlmostEqual(sum(area_def.area_extent_ll), 
                                   122.06448093539757, 5, 
                                   'Failed to get lon and lats of area extent')
        