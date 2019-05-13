import unittest
import numpy as np

from pyresample._spatial_mp import Proj

import pyresample.bilinear as bil
from pyresample import geometry, utils, kd_tree


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pts_irregular = (np.array([[-1., 1.], ]),
                             np.array([[1., 2.], ]),
                             np.array([[-2., -1.], ]),
                             np.array([[2., -4.], ]))
        cls.pts_vert_parallel = (np.array([[-1., 1.], ]),
                                 np.array([[1., 2.], ]),
                                 np.array([[-1., -1.], ]),
                                 np.array([[1., -2.], ]))
        cls.pts_both_parallel = (np.array([[-1., 1.], ]),
                                 np.array([[1., 1.], ]),
                                 np.array([[-1., -1.], ]),
                                 np.array([[1., -1.], ]))

        # Area definition with four pixels
        target_def = geometry.AreaDefinition('areaD',
                                             'Europe (3km, HRV, VTC)',
                                             'areaD',
                                             {'a': '6378144.0',
                                              'b': '6356759.0',
                                              'lat_0': '50.00',
                                              'lat_ts': '50.00',
                                              'lon_0': '8.00',
                                              'proj': 'stere'},
                                             4, 4,
                                             [-1370912.72,
                                              -909968.64000000001,
                                              1029087.28,
                                              1490031.3600000001])

        # Input data around the target pixel at 0.63388324, 55.08234642,
        in_shape = (100, 100)
        cls.data1 = np.ones((in_shape[0], in_shape[1]))
        cls.data2 = 2. * cls.data1
        cls.data3 = cls.data1 + 9.5
        lons, lats = np.meshgrid(np.linspace(-25., 40., num=in_shape[0]),
                                 np.linspace(45., 75., num=in_shape[1]))
        cls.swath_def = geometry.SwathDefinition(lons=lons, lats=lats)

        radius = 50e3
        cls.neighbours = 32
        input_idxs, output_idxs, idx_ref, dists = \
            kd_tree.get_neighbour_info(cls.swath_def, target_def,
                                       radius, neighbours=cls.neighbours,
                                       nprocs=1)
        input_size = input_idxs.sum()
        index_mask = (idx_ref == input_size)
        idx_ref = np.where(index_mask, 0, idx_ref)

        cls.input_idxs = input_idxs
        cls.target_def = target_def
        cls.idx_ref = idx_ref

    def test_calc_abc(self):
        # No np.nan inputs
        pt_1, pt_2, pt_3, pt_4 = self.pts_irregular
        res = bil._calc_abc(pt_1, pt_2, pt_3, pt_4, 0.0, 0.0)
        self.assertFalse(np.isnan(res[0]))
        self.assertFalse(np.isnan(res[1]))
        self.assertFalse(np.isnan(res[2]))
        # np.nan input -> np.nan output
        res = bil._calc_abc(np.array([[np.nan, np.nan]]),
                            pt_2, pt_3, pt_4, 0.0, 0.0)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))
        self.assertTrue(np.isnan(res[2]))

    def test_get_ts_irregular(self):
        res = bil._get_ts_irregular(self.pts_irregular[0],
                                    self.pts_irregular[1],
                                    self.pts_irregular[2],
                                    self.pts_irregular[3],
                                    0., 0.)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = bil._get_ts_irregular(self.pts_vert_parallel[0],
                                    self.pts_vert_parallel[1],
                                    self.pts_vert_parallel[2],
                                    self.pts_vert_parallel[3],
                                    0., 0.)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))

    def test_get_ts_uprights_parallel(self):
        res = bil._get_ts_uprights_parallel(self.pts_vert_parallel[0],
                                            self.pts_vert_parallel[1],
                                            self.pts_vert_parallel[2],
                                            self.pts_vert_parallel[3],
                                            0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_ts_parallellogram(self):
        res = bil._get_ts_parallellogram(self.pts_both_parallel[0],
                                         self.pts_both_parallel[1],
                                         self.pts_both_parallel[2],
                                         0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_ts(self):
        out_x = np.array([[0.]])
        out_y = np.array([[0.]])
        res = bil._get_ts(self.pts_irregular[0],
                          self.pts_irregular[1],
                          self.pts_irregular[2],
                          self.pts_irregular[3],
                          out_x, out_y)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = bil._get_ts(self.pts_both_parallel[0],
                          self.pts_both_parallel[1],
                          self.pts_both_parallel[2],
                          self.pts_both_parallel[3],
                          out_x, out_y)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)
        res = bil._get_ts(self.pts_vert_parallel[0],
                          self.pts_vert_parallel[1],
                          self.pts_vert_parallel[2],
                          self.pts_vert_parallel[3],
                          out_x, out_y)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_solve_quadratic(self):
        res = bil._solve_quadratic(1, 0, 0)
        self.assertEqual(res[0], 0.0)
        res = bil._solve_quadratic(1, 2, 1)
        self.assertTrue(np.isnan(res[0]))
        res = bil._solve_quadratic(1, 2, 1, min_val=-2.)
        self.assertEqual(res[0], -1.0)
        # Test that small adjustments work
        pt_1, pt_2, pt_3, pt_4 = self.pts_vert_parallel
        pt_1 = self.pts_vert_parallel[0].copy()
        pt_1[0][0] += 1e-7
        res = bil._calc_abc(pt_1, pt_2, pt_3, pt_4, 0.0, 0.0)
        res = bil._solve_quadratic(res[0], res[1], res[2])
        self.assertAlmostEqual(res[0], 0.5, 5)
        res = bil._calc_abc(pt_1, pt_3, pt_2, pt_4, 0.0, 0.0)
        res = bil._solve_quadratic(res[0], res[1], res[2])
        self.assertAlmostEqual(res[0], 0.5, 5)

    def test_get_output_xy(self):
        proj = Proj(self.target_def.proj_str)
        out_x, out_y = bil._get_output_xy(self.target_def, proj)
        self.assertTrue(out_x.all())
        self.assertTrue(out_y.all())

    def test_get_input_xy(self):
        proj = Proj(self.target_def.proj_str)
        in_x, in_y = bil._get_output_xy(self.swath_def, proj)
        self.assertTrue(in_x.all())
        self.assertTrue(in_y.all())

    def test_get_bounding_corners(self):
        proj = Proj(self.target_def.proj_str)
        out_x, out_y = bil._get_output_xy(self.target_def, proj)
        in_x, in_y = bil._get_input_xy(self.swath_def, proj,
                                       self.input_idxs, self.idx_ref)
        res = bil._get_bounding_corners(in_x, in_y, out_x, out_y,
                                        self.neighbours, self.idx_ref)
        for i in range(len(res) - 1):
            pt_ = res[i]
            for j in range(2):
                # Only the sixth output location has four valid corners
                self.assertTrue(np.isfinite(pt_[5, j]))

    def test_get_bil_info(self):
        def _check_ts(t__, s__):
            for i in range(len(t__)):
                # Just check the exact value for one pixel
                if i == 5:
                    self.assertAlmostEqual(t__[i], 0.730659147133, 5)
                    self.assertAlmostEqual(s__[i], 0.310314173004, 5)
                # These pixels are outside the area
                elif i in [12, 13, 14, 15]:
                    self.assertTrue(np.isnan(t__[i]))
                    self.assertTrue(np.isnan(s__[i]))
                # All the others should have values between 0.0 and 1.0
                else:
                    self.assertTrue(t__[i] >= 0.0)
                    self.assertTrue(s__[i] >= 0.0)
                    self.assertTrue(t__[i] <= 1.0)
                    self.assertTrue(s__[i] <= 1.0)

        t__, s__, input_idxs, idx_arr = bil.get_bil_info(self.swath_def,
                                                         self.target_def,
                                                         50e5, neighbours=32,
                                                         nprocs=1,
                                                         reduce_data=False)
        _check_ts(t__, s__)

        t__, s__, input_idxs, idx_arr = bil.get_bil_info(self.swath_def,
                                                         self.target_def,
                                                         50e5, neighbours=32,
                                                         nprocs=1,
                                                         reduce_data=True)
        _check_ts(t__, s__)

    def test_get_sample_from_bil_info(self):
        t__, s__, input_idxs, idx_arr = bil.get_bil_info(self.swath_def,
                                                         self.target_def,
                                                         50e5, neighbours=32,
                                                         nprocs=1)
        # Sample from data1
        res = bil.get_sample_from_bil_info(self.data1.ravel(), t__, s__,
                                           input_idxs, idx_arr)
        self.assertEqual(res[5], 1.)
        # Sample from data2
        res = bil.get_sample_from_bil_info(self.data2.ravel(), t__, s__,
                                           input_idxs, idx_arr)
        self.assertEqual(res[5], 2.)
        # Reshaping
        res = bil.get_sample_from_bil_info(self.data2.ravel(), t__, s__,
                                           input_idxs, idx_arr,
                                           output_shape=self.target_def.shape)
        res = res.shape
        self.assertEqual(res[0], self.target_def.shape[0])
        self.assertEqual(res[1], self.target_def.shape[1])

        # Test rounding that is happening for certain values
        res = bil.get_sample_from_bil_info(self.data3.ravel(), t__, s__,
                                           input_idxs, idx_arr,
                                           output_shape=self.target_def.shape)
        # Four pixels are outside of the data
        self.assertEqual(np.isnan(res).sum(), 4)

    def test_resample_bilinear(self):
        # Single array
        res = bil.resample_bilinear(self.data1,
                                    self.swath_def,
                                    self.target_def,
                                    50e5, neighbours=32,
                                    nprocs=1)
        self.assertEqual(res.shape, self.target_def.shape)
        # There are 12 pixels with value 1, all others are zero
        self.assertEqual(res.sum(), 12)
        self.assertEqual((res == 0).sum(), 4)

        # Single array with masked output
        res = bil.resample_bilinear(self.data1,
                                    self.swath_def,
                                    self.target_def,
                                    50e5, neighbours=32,
                                    nprocs=1, fill_value=None)
        self.assertTrue(hasattr(res, 'mask'))
        # There should be 12 valid pixels
        self.assertEqual(self.target_def.size - res.mask.sum(), 12)

        # Two stacked arrays
        data = np.dstack((self.data1, self.data2))
        res = bil.resample_bilinear(data,
                                    self.swath_def,
                                    self.target_def)
        shp = res.shape
        self.assertEqual(shp[0:2], self.target_def.shape)
        self.assertEqual(shp[-1], 2)


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
