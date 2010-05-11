import os
import unittest

import numpy

from pyresample import image, geometry, grid, utils

def mask(f):
    f.mask = True
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

    def test_image(self):        
        data = numpy.fromfunction(lambda y, x: y*x*10**-6, (3712, 3712))
        msg_con = image.ImageContainerQuick(data, self.msg_area)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.failUnlessAlmostEqual(cross_sum, expected, msg='ImageContainer resampling quick failed')
        
    def test_return_type(self):
        data = numpy.ones((3712, 3712)).astype('int')
        msg_con = image.ImageContainerQuick(data, self.msg_area)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        self.assertTrue(data.dtype is res.dtype, msg='Failed to maintain input data type')
    
    @mask
    def test_masked_image(self):
        data = numpy.zeros((3712, 3712))
        mask = numpy.zeros((3712, 3712))
        mask[:, 1865:] = 1
        data_masked = numpy.ma.array(data, mask=mask)
        msg_con = image.ImageContainerQuick(data_masked, self.msg_area)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        resampled_mask = res.mask.astype('int')
        expected = numpy.fromfile(os.path.join(os.path.dirname(__file__), 'test_files', 'mask_grid.dat'), 
                                  sep=' ').reshape((800, 800))
        self.assertTrue(numpy.array_equal(resampled_mask, expected), msg='Failed to resample masked array')

    @mask
    def test_masked_image_fill(self):
        data = numpy.zeros((3712, 3712))
        mask = numpy.zeros((3712, 3712))
        mask[:, 1865:] = 1
        data_masked = numpy.ma.array(data, mask=mask)
        msg_con = image.ImageContainerQuick(data_masked, self.msg_area)
        area_con = msg_con.resample(self.area_def, fill_value=None)
        res = area_con.image_data
        resampled_mask = res.mask.astype('int')
        expected = numpy.fromfile(os.path.join(os.path.dirname(__file__), 'test_files', 'mask_grid.dat'), 
                                  sep=' ').reshape((800, 800))
        self.assertTrue(numpy.array_equal(resampled_mask, expected), msg='Failed to resample masked array')
        
    def test_nearest_neighbour(self):
        data = numpy.fromfunction(lambda y, x: y*x*10**-6, (3712, 3712))
        msg_con = image.ImageContainerNearest(data, self.msg_area, 50000)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 399936.783062
        self.failUnlessAlmostEqual(cross_sum, expected, 
                                   msg='ImageContainer resampling nearest neighbour failed')
        
    def test_nearest_neighbour_multi(self):
        data1 = numpy.fromfunction(lambda y, x: y*x*10**-6, (3712, 3712))
        data2 = numpy.fromfunction(lambda y, x: y*x*10**-6, (3712, 3712)) * 2
        data = numpy.dstack((data1, data2))
        msg_con = image.ImageContainerNearest(data, self.msg_area, 50000)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum1 = res[:, :, 0].sum()
        expected1 = 399936.783062
        self.failUnlessAlmostEqual(cross_sum1, expected1, 
                                   msg='ImageContainer resampling nearest neighbour multi failed')        
        cross_sum2 = res[:, :, 1].sum()
        expected2 = 399936.783062 * 2
        self.failUnlessAlmostEqual(cross_sum2, expected2, 
                                   msg='ImageContainer resampling nearest neighbour multi failed')
        
    def test_nearest_neighbour_multi_preproc(self):
        data1 = numpy.fromfunction(lambda y, x: y*x*10**-6, (3712, 3712))
        data2 = numpy.fromfunction(lambda y, x: y*x*10**-6, (3712, 3712)) * 2
        data = numpy.dstack((data1, data2))
        msg_con = image.ImageContainer(data, self.msg_area)
        #area_con = msg_con.resample_area_nearest_neighbour(self.area_def, 50000)
        row_indices, col_indices = \
            utils.generate_nearest_neighbour_linesample_arrays(self.msg_area, 
                                                               self.area_def, 
                                                               50000)
        res = msg_con.get_array_from_linesample(row_indices, col_indices)
        cross_sum1 = res[:, :, 0].sum()
        expected1 = 399936.783062
        self.failUnlessAlmostEqual(cross_sum1, expected1, 
                                   msg='ImageContainer resampling nearest neighbour multi preproc failed')        
        cross_sum2 = res[:, :, 1].sum()
        expected2 = 399936.783062 * 2
        self.failUnlessAlmostEqual(cross_sum2, expected2, 
                                   msg='ImageContainer resampling nearest neighbour multi preproc failed')
        
        
        


