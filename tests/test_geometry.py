import unittest

import numpy as np

from pyresample import geometry


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
        self.failUnlessAlmostEqual(cart_coords.sum(), 5872042754516.1591797,
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
        
    
    @tmp
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

        
