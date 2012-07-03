from __future__ import with_statement

import os
import sys
import unittest
import warnings
if sys.version_info < (2, 6):
    warnings.simplefilter("ignore")
else:    
    warnings.simplefilter("always")
import numpy

from pyresample import kd_tree, utils, geometry, grid, data_reduce


def mp(f):
    f.mp = True
    return f

def quick(f):
    f.quick = True
    return f

def tmp(f):
    f.tmp = True
    return f

class Test(unittest.TestCase):

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

    tdata = numpy.array([1, 2, 3])
    tlons = numpy.array([11.280789, 12.649354, 12.080402])
    tlats = numpy.array([56.011037, 55.629675, 55.641535])
    tswath = geometry.SwathDefinition(lons=tlons, lats=tlats)
    #grid = numpy.ones((1, 1, 2))
    #grid[0, 0, 0] = 12.562036
    #grid[0, 0, 1] = 55.715613
    tgrid = geometry.CoordinateDefinition(lons=numpy.array([12.562036]), 
                                          lats=numpy.array([55.715613])) 
               
    def test_nearest_base(self):     
        res = kd_tree.resample_nearest(self.tswath,\
                                     self.tdata.ravel(), self.tgrid,\
                                     100000, reduce_data=False, segments=1)
        self.failUnless(res[0] == 2, 'Failed to calculate nearest neighbour')
    
    def test_gauss_base(self):
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(self.tswath, \
                                             self.tdata.ravel(), self.tgrid,\
                                             50000, 25000, reduce_data=False, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(self.tswath, \
                                             self.tdata.ravel(), self.tgrid,\
                                             50000, 25000, reduce_data=False, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour warning')
                self.failIf(('Searching' not in str(w[0].message)), 'Failed to create correct neighbour warning')    
        self.failUnlessAlmostEqual(res[0], 2.2020729, 5, \
                                   'Failed to calculate gaussian weighting')
        
    def test_custom_base(self):
        def wf(dist):
            return 1 - dist/100000.0
        
        if sys.version_info < (2, 6):
            res = kd_tree.resample_custom(self.tswath,\
                                         self.tdata.ravel(), self.tgrid,\
                                         50000, wf, reduce_data=False, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:     
                res = kd_tree.resample_custom(self.tswath,\
                                             self.tdata.ravel(), self.tgrid,\
                                             50000, wf, reduce_data=False, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour warning')
                self.failIf(('Searching' not in str(w[0].message)), 'Failed to create correct neighbour warning')        
        self.failUnlessAlmostEqual(res[0], 2.4356757, 5,\
                                   'Failed to calculate custom weighting')
    def test_nearest(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),\
                                     self.area_def, 50000, segments=1)        
        cross_sum = res.sum()        
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling nearest failed')
       
    def test_nearest_1d(self):
        data = numpy.fromfunction(lambda x, y: x * y, (800, 800))        
        lons = numpy.fromfunction(lambda x: 3 + x / 100. , (500,))
        lats = numpy.fromfunction(lambda x: 75 - x / 10., (500,))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(self.area_def, data.ravel(),
                                       swath_def, 50000, segments=1)
        cross_sum = res.sum()        
        expected = 35821299.0
        self.failUnlessEqual(res.shape, (500,),
                             msg='Swath resampling nearest 1d failed')
        self.failUnlessEqual(cross_sum, expected,
                             msg='Swath resampling nearest 1d failed')
    
    def test_nearest_empty(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),\
                                     self.area_def, 50000, segments=1)        
        cross_sum = res.sum()        
        expected = 0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling nearest empty failed')
    
    def test_nearest_empty_multi(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data_multi,\
                                     self.area_def, 50000, segments=1)                
        self.failUnlessEqual(res.shape, (800, 800, 3),\
                             msg='Swath resampling nearest empty multi failed')
    
    def test_nearest_empty_multi_masked(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data_multi,\
                                     self.area_def, 50000, segments=1,
                                     fill_value=None)                
        self.failUnlessEqual(res.shape, (800, 800, 3),
                             msg='Swath resampling nearest empty multi masked failed')
            
    def test_nearest_empty_masked(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),\
                                     self.area_def, 50000, segments=1, 
                                     fill_value=None)        
        cross_sum = res.mask.sum()        
        expected = res.size
        self.failUnless(cross_sum == expected,
                        msg='Swath resampling nearest empty masked failed')
    
    def test_nearest_segments(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),\
                                     self.area_def, 50000, segments=2)        
        cross_sum = res.sum()        
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling nearest segments failed')
    
    def test_nearest_remap(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),\
                                     self.area_def, 50000, segments=1)
        remap = kd_tree.resample_nearest(self.area_def, res.ravel(),\
                                       swath_def, 5000, segments=1)        
        cross_sum = remap.sum()
        expected = 22275.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Grid remapping nearest failed')
    
    @mp
    def test_nearest_mp(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),\
                                     self.area_def, 50000, nprocs=2, segments=1)
        cross_sum = res.sum()
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling mp nearest failed')
       
    def test_nearest_multi(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        res = kd_tree.resample_nearest(swath_def, data_multi,\
                                     self.area_def, 50000, segments=1)        
        cross_sum = res.sum()
        expected = 3 * 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath multi channel resampling nearest failed')
     
    def test_nearest_multi_unraveled(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.dstack((data, data, data))
        res = kd_tree.resample_nearest(swath_def, data_multi,\
                                     self.area_def, 50000, segments=1)        
        cross_sum = res.sum()
        expected = 3 * 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath multi channel resampling nearest failed')
        
    def test_gauss_sparse(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_gauss(swath_def, data.ravel(),\
                                     self.area_def, 50000, 25000, fill_value=-1, segments=1)        
        cross_sum = res.sum()        
        expected = 15387753.9852
        self.failUnlessAlmostEqual(cross_sum, expected, places=3,\
                                   msg='Swath gauss sparse nearest failed')
            
    def test_gauss(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-5, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(swath_def, data.ravel(),\
                                         self.area_def, 50000, 25000, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, data.ravel(),\
                                             self.area_def, 50000, 25000, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')        
        cross_sum = res.sum()        
        expected = 4872.81050892
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath resampling gauss failed')

    @tmp
    def test_gauss_fwhm(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-5, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(swath_def, data.ravel(),\
                                         self.area_def, 50000, utils.fwhm2sigma(41627.730557884883), segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, data.ravel(),\
                                             self.area_def, 50000, utils.fwhm2sigma(41627.730557884883), segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')        
        cross_sum = res.sum()        
        expected = 4872.81050892
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath resampling gauss failed')
        
    def test_gauss_multi(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(swath_def, data_multi,\
                                         self.area_def, 50000, [25000, 15000, 10000], segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, data_multi,\
                                             self.area_def, 50000, [25000, 15000, 10000], segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning') 
        cross_sum = res.sum()        
        expected = 1461.84313918
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel resampling gauss failed')
    
    def test_gauss_multi_mp(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(swath_def, data_multi,\
                                         self.area_def, 50000, [25000, 15000, 10000],\
                                         nprocs=2, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, data_multi,\
                                             self.area_def, 50000, [25000, 15000, 10000],\
                                             nprocs=2, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning') 
        cross_sum = res.sum()
        expected = 1461.84313918
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel resampling gauss failed') 
       
    def test_gauss_multi_mp_segments(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(swath_def, data_multi,\
                                         self.area_def, 50000, [25000, 15000, 10000],\
                                         nprocs=2, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, data_multi,\
                                             self.area_def, 50000, [25000, 15000, 10000],\
                                             nprocs=2, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')
        cross_sum = res.sum()
        expected = 1461.84313918
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel segments resampling gauss failed')
        
    def test_gauss_multi_mp_segments_empty(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 165 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        res = kd_tree.resample_gauss(swath_def, data_multi,\
                                     self.area_def, 50000, [25000, 15000, 10000],\
                                     nprocs=2, segments=1)
        cross_sum = res.sum()
        self.failUnless(cross_sum == 0,
                        msg=('Swath multi channel segments empty ' 
                             'resampling gauss failed')) 
    
    def test_custom(self):
        def wf(dist):
            return 1 - dist/100000.0
                    
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-5, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        if sys.version_info < (2, 6):
            res = kd_tree.resample_custom(swath_def, data.ravel(),\
                                          self.area_def, 50000, wf, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_custom(swath_def, data.ravel(),\
                                              self.area_def, 50000, wf, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')
        cross_sum = res.sum()
        expected = 4872.81050729
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath custom resampling failed')
     
    def test_custom_multi(self):
        def wf1(dist):
            return 1 - dist/100000.0
        
        def wf2(dist):
            return 1
        
        def wf3(dist):
            return numpy.cos(dist)**2
        
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        if sys.version_info < (2, 6):
            res = kd_tree.resample_custom(swath_def, data_multi,\
                                          self.area_def, 50000, [wf1, wf2, wf3], segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_custom(swath_def, data_multi,\
                                              self.area_def, 50000, [wf1, wf2, wf3], segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')
        cross_sum = res.sum()
        expected = 1461.842980746
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel custom resampling failed')
        
    def test_reduce(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        grid_lons, grid_lats = self.area_def.get_lonlats()
        lons, lats, data = data_reduce.swath_from_lonlat_grid(grid_lons, grid_lats, 
                                                              lons, lats, data, 
                                                              7000)
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Reduce data failed')
    
    def test_reduce_boundary(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        lons, lats, data = data_reduce.swath_from_lonlat_boundaries(self.area_def.lons.boundary,
                                                              self.area_def.lats.boundary, 
                                                              lons, lats, data, 
                                                              7000)
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Reduce data failed')
        
    def test_cartesian_reduce(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        #grid = utils.generate_cartesian_grid(self.area_def)
        grid = self.area_def.cartesian_coords[:]       
        lons, lats, data = data_reduce.swath_from_cartesian_grid(grid, lons, lats, data, 
                                                                 7000)
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Cartesian reduce data failed')
    
    def test_area_con_reduce(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        grid_lons, grid_lats = self.area_def.get_lonlats()
        valid_index = data_reduce.get_valid_index_from_lonlat_grid(grid_lons, grid_lats, 
                                                                   lons, lats, 7000) 
        data = data[valid_index]
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Reduce data failed')
       
    def test_area_con_cartesian_reduce(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        cart_grid = self.area_def.cartesian_coords[:]
        valid_index = data_reduce.get_valid_index_from_cartesian_grid(cart_grid, 
                                                                      lons, lats, 7000)
        data = data[valid_index]
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Cartesian reduce data failed')
               
    def test_masked_nearest(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        mask = numpy.ones((50, 10))
        mask[:, :5] = 0
        masked_data = numpy.ma.array(data, mask=mask)
        res = kd_tree.resample_nearest(swath_def, masked_data.ravel(), 
                                     self.area_def, 50000, segments=1)
        expected_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                    'test_files', 
                                                    'mask_test_nearest_mask.dat'), 
                                                    sep=' ').reshape((800, 800))
        expected_data = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                    'test_files', 
                                                    'mask_test_nearest_data.dat'), 
                                                    sep=' ').reshape((800, 800))        
        self.assertTrue(numpy.array_equal(expected_mask, res.mask), 
                        msg='Resampling of swath mask failed')
        self.assertTrue(numpy.array_equal(expected_data, res.data), 
                        msg='Resampling of swath masked data failed')
           
    def test_masked_nearest_1d(self):
        data = numpy.ones((800, 800))
        data[:400, :] = 2
        lons = numpy.fromfunction(lambda x: 3 + x / 100. , (500,))
        lats = numpy.fromfunction(lambda x: 75 - x / 10., (500,))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        mask = numpy.ones((800, 800))
        mask[400:, :] = 0
        masked_data = numpy.ma.array(data, mask=mask)
        res = kd_tree.resample_nearest(self.area_def, masked_data.ravel(),
                                       swath_def, 50000, segments=1)
        self.failUnlessEqual(res.mask.sum(), 108,
                             msg='Swath resampling masked nearest 1d failed')
        
    
    def test_masked_gauss(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        mask = numpy.ones((50, 10))
        mask[:, :5] = 0
        masked_data = numpy.ma.array(data, mask=mask)
        res = kd_tree.resample_gauss(swath_def, masked_data.ravel(),\
                                   self.area_def, 50000, 25000, segments=1)
        expected_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                    'test_files', 
                                                    'mask_test_mask.dat'), 
                                                    sep=' ').reshape((800, 800))
        expected_data = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                    'test_files', 
                                                    'mask_test_data.dat'), 
                                                    sep=' ').reshape((800, 800))
        expected = expected_data.sum()
        cross_sum = res.data.sum()
        
        self.assertTrue(numpy.array_equal(expected_mask, res.mask), 
                        msg='Gauss resampling of swath mask failed')
        self.failUnlessAlmostEqual(cross_sum, expected, places=3,\
                                   msg='Gauss resampling of swath masked data failed')
        
     
    def test_masked_fill_float(self):
        data = numpy.ones((50, 10))
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(), 
                                     self.area_def, 50000, fill_value=None, segments=1)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_fill_value.dat'), 
                                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask
        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                         msg='Failed to create fill mask on float data')
        
    def test_masked_fill_int(self):
        data = numpy.ones((50, 10)).astype('int')
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(), 
                                     self.area_def, 50000, fill_value=None, segments=1)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_fill_value.dat'), 
                                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask
        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                        msg='Failed to create fill mask on integer data')
        
    def test_masked_full(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        mask = numpy.ones((50, 10))
        mask[:, :5] = 0
        masked_data = numpy.ma.array(data, mask=mask)
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, 
                                    masked_data.ravel(), self.area_def, 50000,
                                    fill_value=None, segments=1)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_full_fill.dat'), 
                                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask

        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                         msg='Failed to create fill mask on masked data')
        
    def test_masked_full_multi(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        mask1 = numpy.ones((50, 10))
        mask1[:, :5] = 0
        mask2 = numpy.ones((50, 10))
        mask2[:, 5:] = 0
        mask3 = numpy.ones((50, 10))
        mask3[:25, :] = 0
        data_multi = numpy.column_stack((data.ravel(), data.ravel(), data.ravel()))
        mask_multi = numpy.column_stack((mask1.ravel(), mask2.ravel(), mask3.ravel()))
        masked_data = numpy.ma.array(data_multi, mask=mask_multi)
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, 
                                    masked_data, self.area_def, 50000,
                                    fill_value=None, segments=1)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_full_fill_multi.dat'), 
                                                         sep=' ').reshape((800, 800, 3))
        fill_mask = res.mask
        cross_sum = res.sum()
        expected = 357140.0
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Failed to resample masked data')        
        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                         msg='Failed to create fill mask on masked data')
        
    def test_nearest_from_sample(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        valid_input_index, valid_output_index, index_array, distance_array = \
                                    kd_tree.get_neighbour_info(swath_def, 
                                                             self.area_def, 
                                                             50000, neighbours=1, segments=1)
        res = kd_tree.get_sample_from_neighbour_info('nn', (800, 800), data.ravel(), 
                                                   valid_input_index, valid_output_index, 
                                                   index_array)        
        cross_sum = res.sum()        
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling from neighbour info nearest failed')
    
    def test_custom_multi_from_sample(self):
        def wf1(dist):
            return 1 - dist/100000.0
        
        def wf2(dist):
            return 1
        
        def wf3(dist):
            return numpy.cos(dist)**2
        
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        
        if sys.version_info < (2, 6):
            valid_input_index, valid_output_index, index_array, distance_array = \
                                        kd_tree.get_neighbour_info(swath_def, 
                                                                   self.area_def, 
                                                                   50000, segments=1)
        else:
            with warnings.catch_warnings(record=True) as w:
                valid_input_index, valid_output_index, index_array, distance_array = \
                                            kd_tree.get_neighbour_info(swath_def, 
                                                                       self.area_def, 
                                                                       50000, segments=1)
                self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
                self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')
            
        res = kd_tree.get_sample_from_neighbour_info('custom', (800, 800), 
                                                     data_multi, 
                                                     valid_input_index, valid_output_index, 
                                                     index_array, distance_array, 
                                                     weight_funcs=[wf1, wf2, wf3])
                        
        cross_sum = res.sum()
        
        expected = 1461.842980746
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel custom resampling from neighbour info failed 1')
        res = kd_tree.get_sample_from_neighbour_info('custom', (800, 800), 
                                                   data_multi, 
                                                   valid_input_index, valid_output_index, 
                                                   index_array, distance_array, 
                                                   weight_funcs=[wf1, wf2, wf3])
        
        # Look for error where input data has been manipulated    
        cross_sum = res.sum()
        expected = 1461.842980746
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel custom resampling from neighbour info failed 2')


    def test_masked_multi_from_sample(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        mask1 = numpy.ones((50, 10))
        mask1[:, :5] = 0
        mask2 = numpy.ones((50, 10))
        mask2[:, 5:] = 0
        mask3 = numpy.ones((50, 10))
        mask3[:25, :] = 0
        data_multi = numpy.column_stack((data.ravel(), data.ravel(), data.ravel()))
        mask_multi = numpy.column_stack((mask1.ravel(), mask2.ravel(), mask3.ravel()))
        masked_data = numpy.ma.array(data_multi, mask=mask_multi)
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
#        res = swath.resample_nearest(lons.ravel(), lats.ravel(), 
#                                    masked_data, self.area_def, 50000,
#                                    fill_value=None)
        valid_input_index, valid_output_index, index_array, distance_array = \
                                    kd_tree.get_neighbour_info(swath_def, 
                                                             self.area_def, 
                                                             50000, neighbours=1, segments=1)
        res = kd_tree.get_sample_from_neighbour_info('nn', (800, 800), 
                                                   masked_data, 
                                                   valid_input_index, 
                                                   valid_output_index, index_array,
                                                   fill_value=None)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_full_fill_multi.dat'), 
                                                         sep=' ').reshape((800, 800, 3))
        fill_mask = res.mask        
        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                         msg='Failed to create fill mask on masked data')
        
        

