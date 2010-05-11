import os
import unittest

import numpy

from pyresample import swath, utils, geometry, grid, data_reduce


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
    tgrid = numpy.ones((1, 1, 2))
    tgrid[0, 0, 0] = 12.562036
    tgrid[0, 0, 1] = 55.715613
            
        
    def test_nearest_base(self):     
        res = swath.resample_nearest(self.tlons.ravel(), self.tlats.ravel(),\
                                     self.tdata.ravel(), self.tgrid,\
                                     100000, reduce_data=False)
        self.failUnless(res[0] == 2, 'Failed to calculate nearest neighbour')
        
    def test_gauss_base(self):     
        res = swath.resample_gauss(self.tlons.ravel(), self.tlats.ravel(),\
                                     self.tdata.ravel(), self.tgrid,\
                                     50000, 25000, reduce_data=False)
        self.failUnlessAlmostEqual(res[0], 2.2020729, 5, \
                                   'Failed to calculate gaussian weighting')
    
    def test_custom_base(self):
        def wf(dist):
            return 1 - dist/100000.0
             
        res = swath.resample_custom(self.tlons.ravel(), self.tlats.ravel(),\
                                     self.tdata.ravel(), self.tgrid,\
                                     50000, wf, reduce_data=False)        
        self.failUnlessAlmostEqual(res[0], 2.4356757, 5,\
                                   'Failed to calculate custom weighting')
    @tmp
    def test_nearest(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), data.ravel(),\
                                     self.area_def, 50000)        
        cross_sum = res.sum()        
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling nearest failed')
    
    @mp
    def test_nearest_mp(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), data.ravel(),\
                                     self.area_def, 50000, nprocs=2)
        cross_sum = res.sum()
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling mp nearest failed')
        
    def test_nearest_multi(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), data_multi,\
                                     self.area_def, 50000)        
        cross_sum = res.sum()
        expected = 3 * 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath multi channel resampling nearest failed')
        
    def test_gauss_sparse(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        res = swath.resample_gauss(lons.ravel(), lats.ravel(), data.ravel(),\
                                     self.area_def, 50000, 25000, fill_value=-1)        
        cross_sum = res.sum()        
        expected = 15387753.9852
        self.failUnlessAlmostEqual(cross_sum, expected, places=3,\
                                   msg='Swath gauss sparse nearest failed')
            
    def test_gauss(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-5, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        res = swath.resample_gauss(lons.ravel(), lats.ravel(), data.ravel(),\
                                     self.area_def, 50000, 25000)        
        cross_sum = res.sum()        
        expected = 4872.81050892
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath resampling gauss failed')
        
    def test_gauss_multi(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        res = swath.resample_gauss(lons.ravel(), lats.ravel(), data_multi,\
                                     self.area_def, 50000, [25000, 15000, 10000])
        cross_sum = res.sum()        
        expected = 1461.84313918
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel resampling gauss failed')
    
    def test_gauss_multi_mp(self):
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-6, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        res = swath.resample_gauss(lons.ravel(), lats.ravel(), data_multi,\
                                     self.area_def, 50000, [25000, 15000, 10000],\
                                     nprocs=2)
        cross_sum = res.sum()
        expected = 1461.84313918
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel resampling gauss failed') 
       
    def test_custom(self):
        def wf(dist):
            return 1 - dist/100000.0
                    
        data = numpy.fromfunction(lambda y, x: (y + x)*10**-5, (5000, 100))        
        lons = numpy.fromfunction(lambda y, x: 3 + (10.0/100)*x, (5000, 100))
        lats = numpy.fromfunction(lambda y, x: 75 - (50.0/5000)*y, (5000, 100))
        res = swath.resample_custom(lons.ravel(), lats.ravel(), data.ravel(),\
                                     self.area_def, 50000, wf)
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
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        res = swath.resample_custom(lons.ravel(), lats.ravel(), data_multi,\
                                    self.area_def, 50000, [wf1, wf2, wf3])
        cross_sum = res.sum()
        expected = 1461.84298477
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
        
    def test_cartesian_reduce(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        grid = utils.generate_cartesian_grid(self.area_def)       
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
#        lons, lats, data = data_reduce.swath_from_lonlat_grid(grid_lons, grid_lats, 
#                                                              lons, lats, data, 
#                                                              7000)
        cart_grid = numpy.zeros((800, 800, 2))
        cart_grid[:, :, 0] = grid_lons
        cart_grid[:, :, 1] = grid_lats
        area_con = swath._AreaDefContainer(cart_grid)
        valid_index = area_con.get_valid_index(lons, lats, 7000)
        data = data[valid_index]
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Reduce data failed')
       
    def test_area_con_cartesian_reduce(self):
        data = numpy.fromfunction(lambda y, x: (y + x), (1000, 1000))
        lons = numpy.fromfunction(lambda y, x: -180 + (360.0/1000)*x, (1000, 1000))
        lats = numpy.fromfunction(lambda y, x: -90 + (180.0/1000)*y, (1000, 1000))
        cart_grid = utils.generate_cartesian_grid(self.area_def)       
#        lons, lats, data = data_reduce.swath_from_cartesian_grid(grid, lons, lats, data, 
#                                                                 7000)
        area_con = swath._AreaDefContainer(cart_grid)
        valid_index = area_con.get_valid_index(lons, lats, 7000)
        data = data[valid_index]
        cross_sum = data.sum()
        expected = 20514375.0
        self.failUnlessAlmostEqual(cross_sum, expected, msg='Cartesian reduce data failed')
        
    def test_nearest_cartesian(self):
        data = numpy.fromfunction(lambda y, x: y*x, (50, 10))        
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        cart_grid = utils.generate_cartesian_grid(self.area_def)
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), data.ravel(),
                                     cart_grid, 50000, reduce_data=False)
        cross_sum = res.sum()
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling nearest from cartesian grid failed')
        
    def test_masked_nearest(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        mask = numpy.ones((50, 10))
        mask[:, :5] = 0
        masked_data = numpy.ma.array(data, mask=mask)
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), masked_data.ravel(), 
                                     self.area_def, 50000)
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
            
    def test_masked_gauss(self):
        data = numpy.ones((50, 10))
        data[:, 5:] = 2
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10)) 
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        mask = numpy.ones((50, 10))
        mask[:, :5] = 0
        masked_data = numpy.ma.array(data, mask=mask)
        res = swath.resample_gauss(lons.ravel(), lats.ravel(), masked_data.ravel(),\
                                   self.area_def, 50000, 25000)
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
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), data.ravel(), 
                                     self.area_def, 50000, fill_value=None)
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
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), data.ravel(), 
                                     self.area_def, 50000, fill_value=None)
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
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), 
                                    masked_data.ravel(), self.area_def, 50000,
                                    fill_value=None)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_full_fill.dat'), 
                                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask
#        mask = res.mask.astype('float64')
#        mask.tofile('/home/esn/dvl/test/pyresample_masked/mask_test_full_fill.dat', sep=' ')
#        import h5py  
#        h5out = h5py.File('/home/esn/data/avhrr_test/test_mf.h5', 'w')
#        h5out.create_dataset('data', data=res.data.copy(), compression=1)
#        h5out.create_dataset('mask', data=res.mask.astype(numpy.int).copy(), compression=1)
#        h5out.create_dataset('exp_mask', data=expected_fill_mask, compression=1)
#        h5out['data'].attrs['CLASS'] = 'IMAGE'
#        h5out['mask'].attrs['CLASS'] = 'IMAGE'
#        h5out['exp_mask'].attrs['CLASS'] = 'IMAGE'                
#        h5out.close()

        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                         msg='Failed to create fill mask on masked data')
        
    @tmp
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
        res = swath.resample_nearest(lons.ravel(), lats.ravel(), 
                                    masked_data, self.area_def, 50000,
                                    fill_value=None)
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
        valid_index, index_array, distance_array = \
                                    swath.get_neighbour_info(lons.ravel(), lats.ravel(), 
                                                             self.area_def, 
                                                             50000, neighbours=1)
        res = swath.get_sample_from_neighbour_info('nn', (800, 800), data.ravel(), 
                                                   valid_index, index_array)        
        cross_sum = res.sum()        
        expected = 15874591.0
        self.failUnlessEqual(cross_sum, expected,\
                             msg='Swath resampling from neighbour info nearest failed')
    
    @tmp    
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
        data_multi = numpy.column_stack((data.ravel(), data.ravel(),\
                                         data.ravel()))
        valid_index, index_array, distance_array = \
                                    swath.get_neighbour_info(lons.ravel(), lats.ravel(), 
                                                             self.area_def, 
                                                             50000)
        res = swath.get_sample_from_neighbour_info('custom', (800, 800), 
                                                   data_multi, 
                                                   valid_index, index_array, distance_array, 
                                                   weight_funcs=[wf1, wf2, wf3])
            
        cross_sum = res.sum()
        expected = 1461.84298477
        self.failUnlessAlmostEqual(cross_sum, expected,\
                                   msg='Swath multi channel custom resampling from neighbour info failed 1')
        res = swath.get_sample_from_neighbour_info('custom', (800, 800), 
                                                   data_multi, 
                                                   valid_index, index_array, distance_array, 
                                                   weight_funcs=[wf1, wf2, wf3])
            
        cross_sum = res.sum()
        expected = 1461.84298477
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
#        res = swath.resample_nearest(lons.ravel(), lats.ravel(), 
#                                    masked_data, self.area_def, 50000,
#                                    fill_value=None)
        valid_index, index_array, distance_array = \
                                    swath.get_neighbour_info(lons.ravel(), lats.ravel(), 
                                                             self.area_def, 
                                                             50000, neighbours=1)
        res = swath.get_sample_from_neighbour_info('nn', (800, 800), 
                                                   masked_data, 
                                                   valid_index, index_array,
                                                   fill_value=None)
        expected_fill_mask = numpy.fromfile(os.path.join(os.path.dirname(__file__), 
                                                         'test_files', 
                                                         'mask_test_full_fill_multi.dat'), 
                                                         sep=' ').reshape((800, 800, 3))
        fill_mask = res.mask        
        self.assertTrue(numpy.array_equal(fill_mask, expected_fill_mask), 
                         msg='Failed to create fill mask on masked data')
        
        

