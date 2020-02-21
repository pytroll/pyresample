"""Test bilinear interpolation."""
import unittest
import numpy as np
from unittest import mock


class TestNumpyBilinear(unittest.TestCase):
    """Test Numpy-based bilinear interpolation."""

    @classmethod
    def setUpClass(cls):
        """Do some setup for the test class."""
        from pyresample import geometry, kd_tree

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
        """Test calculation of quadratic coefficients."""
        from pyresample.bilinear import _calc_abc

        # No np.nan inputs
        pt_1, pt_2, pt_3, pt_4 = self.pts_irregular
        res = _calc_abc(pt_1, pt_2, pt_3, pt_4, 0.0, 0.0)
        self.assertFalse(np.isnan(res[0]))
        self.assertFalse(np.isnan(res[1]))
        self.assertFalse(np.isnan(res[2]))
        # np.nan input -> np.nan output
        res = _calc_abc(np.array([[np.nan, np.nan]]),
                        pt_2, pt_3, pt_4, 0.0, 0.0)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))
        self.assertTrue(np.isnan(res[2]))

    def test_get_ts_irregular(self):
        """Test calculations for irregular corner locations."""
        from pyresample.bilinear import _get_ts_irregular

        res = _get_ts_irregular(self.pts_irregular[0],
                                self.pts_irregular[1],
                                self.pts_irregular[2],
                                self.pts_irregular[3],
                                0., 0.)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = _get_ts_irregular(self.pts_vert_parallel[0],
                                self.pts_vert_parallel[1],
                                self.pts_vert_parallel[2],
                                self.pts_vert_parallel[3],
                                0., 0.)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))

    def test_get_ts_uprights_parallel(self):
        """Test calculation when uprights are parallel."""
        from pyresample.bilinear import _get_ts_uprights_parallel

        res = _get_ts_uprights_parallel(self.pts_vert_parallel[0],
                                        self.pts_vert_parallel[1],
                                        self.pts_vert_parallel[2],
                                        self.pts_vert_parallel[3],
                                        0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_ts_parallellogram(self):
        """Test calculation when the corners form a parallellogram."""
        from pyresample.bilinear import _get_ts_parallellogram

        res = _get_ts_parallellogram(self.pts_both_parallel[0],
                                     self.pts_both_parallel[1],
                                     self.pts_both_parallel[2],
                                     0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_ts(self):
        """Test get_ts()."""
        from pyresample.bilinear import _get_ts

        out_x = np.array([[0.]])
        out_y = np.array([[0.]])
        res = _get_ts(self.pts_irregular[0],
                      self.pts_irregular[1],
                      self.pts_irregular[2],
                      self.pts_irregular[3],
                      out_x, out_y)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = _get_ts(self.pts_both_parallel[0],
                      self.pts_both_parallel[1],
                      self.pts_both_parallel[2],
                      self.pts_both_parallel[3],
                      out_x, out_y)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)
        res = _get_ts(self.pts_vert_parallel[0],
                      self.pts_vert_parallel[1],
                      self.pts_vert_parallel[2],
                      self.pts_vert_parallel[3],
                      out_x, out_y)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_solve_quadratic(self):
        """Test solving quadratic equation."""
        from pyresample.bilinear import (_solve_quadratic, _calc_abc)

        res = _solve_quadratic(1, 0, 0)
        self.assertEqual(res[0], 0.0)
        res = _solve_quadratic(1, 2, 1)
        self.assertTrue(np.isnan(res[0]))
        res = _solve_quadratic(1, 2, 1, min_val=-2.)
        self.assertEqual(res[0], -1.0)
        # Test that small adjustments work
        pt_1, pt_2, pt_3, pt_4 = self.pts_vert_parallel
        pt_1 = self.pts_vert_parallel[0].copy()
        pt_1[0][0] += 1e-7
        res = _calc_abc(pt_1, pt_2, pt_3, pt_4, 0.0, 0.0)
        res = _solve_quadratic(res[0], res[1], res[2])
        self.assertAlmostEqual(res[0], 0.5, 5)
        res = _calc_abc(pt_1, pt_3, pt_2, pt_4, 0.0, 0.0)
        res = _solve_quadratic(res[0], res[1], res[2])
        self.assertAlmostEqual(res[0], 0.5, 5)

    def test_get_output_xy(self):
        """Test calculation of output xy-coordinates."""
        from pyresample.bilinear import _get_output_xy
        from pyresample._spatial_mp import Proj

        proj = Proj(self.target_def.proj_str)
        out_x, out_y = _get_output_xy(self.target_def, proj)
        self.assertTrue(out_x.all())
        self.assertTrue(out_y.all())

    def test_get_input_xy(self):
        """Test calculation of input xy-coordinates."""
        from pyresample.bilinear import _get_input_xy
        from pyresample._spatial_mp import Proj

        proj = Proj(self.target_def.proj_str)
        in_x, in_y = _get_input_xy(self.swath_def, proj,
                                   self.input_idxs, self.idx_ref)
        self.assertTrue(in_x.all())
        self.assertTrue(in_y.all())

    def test_get_bounding_corners(self):
        """Test calculation of bounding corners."""
        from pyresample.bilinear import (_get_output_xy,
                                         _get_input_xy,
                                         _get_bounding_corners)
        from pyresample._spatial_mp import Proj

        proj = Proj(self.target_def.proj_str)
        out_x, out_y = _get_output_xy(self.target_def, proj)
        in_x, in_y = _get_input_xy(self.swath_def, proj,
                                   self.input_idxs, self.idx_ref)
        res = _get_bounding_corners(in_x, in_y, out_x, out_y,
                                    self.neighbours, self.idx_ref)
        for i in range(len(res) - 1):
            pt_ = res[i]
            for j in range(2):
                # Only the sixth output location has four valid corners
                self.assertTrue(np.isfinite(pt_[5, j]))

    def test_get_bil_info(self):
        """Test calculation of bilinear resampling indices."""
        from pyresample.bilinear import get_bil_info

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

        t__, s__, input_idxs, idx_arr = get_bil_info(self.swath_def,
                                                     self.target_def,
                                                     50e5, neighbours=32,
                                                     nprocs=1,
                                                     reduce_data=False)
        _check_ts(t__, s__)

        t__, s__, input_idxs, idx_arr = get_bil_info(self.swath_def,
                                                     self.target_def,
                                                     50e5, neighbours=32,
                                                     nprocs=1,
                                                     reduce_data=True)
        _check_ts(t__, s__)

    def test_get_sample_from_bil_info(self):
        """Test resampling using resampling indices."""
        from pyresample.bilinear import get_bil_info, get_sample_from_bil_info

        t__, s__, input_idxs, idx_arr = get_bil_info(self.swath_def,
                                                     self.target_def,
                                                     50e5, neighbours=32,
                                                     nprocs=1)
        # Sample from data1
        res = get_sample_from_bil_info(self.data1.ravel(), t__, s__,
                                       input_idxs, idx_arr)
        self.assertEqual(res[5], 1.)
        # Sample from data2
        res = get_sample_from_bil_info(self.data2.ravel(), t__, s__,
                                       input_idxs, idx_arr)
        self.assertEqual(res[5], 2.)
        # Reshaping
        res = get_sample_from_bil_info(self.data2.ravel(), t__, s__,
                                       input_idxs, idx_arr,
                                       output_shape=self.target_def.shape)
        res = res.shape
        self.assertEqual(res[0], self.target_def.shape[0])
        self.assertEqual(res[1], self.target_def.shape[1])

        # Test rounding that is happening for certain values
        res = get_sample_from_bil_info(self.data3.ravel(), t__, s__,
                                       input_idxs, idx_arr,
                                       output_shape=self.target_def.shape)
        # Four pixels are outside of the data
        self.assertEqual(np.isnan(res).sum(), 4)

    def test_resample_bilinear(self):
        """Test whole bilinear resampling."""
        from pyresample.bilinear import resample_bilinear

        # Single array
        res = resample_bilinear(self.data1,
                                self.swath_def,
                                self.target_def,
                                50e5, neighbours=32,
                                nprocs=1)
        self.assertEqual(res.shape, self.target_def.shape)
        # There are 12 pixels with value 1, all others are zero
        self.assertEqual(res.sum(), 12)
        self.assertEqual((res == 0).sum(), 4)

        # Single array with masked output
        res = resample_bilinear(self.data1,
                                self.swath_def,
                                self.target_def,
                                50e5, neighbours=32,
                                nprocs=1, fill_value=None)
        self.assertTrue(hasattr(res, 'mask'))
        # There should be 12 valid pixels
        self.assertEqual(self.target_def.size - res.mask.sum(), 12)

        # Two stacked arrays
        data = np.dstack((self.data1, self.data2))
        res = resample_bilinear(data,
                                self.swath_def,
                                self.target_def)
        shp = res.shape
        self.assertEqual(shp[0:2], self.target_def.shape)
        self.assertEqual(shp[-1], 2)


class TestXarrayBilinear(unittest.TestCase):
    """Test Xarra/Dask -based bilinear interpolation."""

    def setUp(self):
        """Do some setup for common things."""
        import dask.array as da
        from xarray import DataArray
        from pyresample import geometry, kd_tree

        self.pts_irregular = (np.array([[-1., 1.], ]),
                              np.array([[1., 2.], ]),
                              np.array([[-2., -1.], ]),
                              np.array([[2., -4.], ]))
        self.pts_vert_parallel = (np.array([[-1., 1.], ]),
                                  np.array([[1., 2.], ]),
                                  np.array([[-1., -1.], ]),
                                  np.array([[1., -2.], ]))
        self.pts_both_parallel = (np.array([[-1., 1.], ]),
                                  np.array([[1., 1.], ]),
                                  np.array([[-1., -1.], ]),
                                  np.array([[1., -1.], ]))

        # Area definition with four pixels
        self.target_def = geometry.AreaDefinition('areaD',
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
        self.data1 = DataArray(da.ones((in_shape[0], in_shape[1])), dims=('y', 'x'))
        self.data2 = 2. * self.data1
        self.data3 = self.data1 + 9.5
        lons, lats = np.meshgrid(np.linspace(-25., 40., num=in_shape[0]),
                                 np.linspace(45., 75., num=in_shape[1]))
        self.source_def = geometry.SwathDefinition(lons=lons, lats=lats)

        self.radius = 50e3
        self.neighbours = 32
        valid_input_index, output_idxs, index_array, dists = \
            kd_tree.get_neighbour_info(self.source_def, self.target_def,
                                       self.radius, neighbours=self.neighbours,
                                       nprocs=1)
        input_size = valid_input_index.sum()
        index_mask = (index_array == input_size)
        index_array = np.where(index_mask, 0, index_array)

        self.valid_input_index = valid_input_index
        self.index_array = index_array

        shp = self.source_def.shape
        self.cols, self.lines = np.meshgrid(np.arange(shp[1]),
                                            np.arange(shp[0]))

    def test_init(self):
        """Test that the resampler has been initialized correctly."""
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        # With defaults
        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)
        self.assertTrue(resampler.source_geo_def == self.source_def)
        self.assertTrue(resampler.target_geo_def == self.target_def)
        self.assertEqual(resampler.radius_of_influence, self.radius)
        self.assertEqual(resampler.neighbours, 32)
        self.assertEqual(resampler.epsilon, 0)
        self.assertTrue(resampler.reduce_data)
        # These should be None
        self.assertIsNone(resampler.valid_input_index)
        self.assertIsNone(resampler.valid_output_index)
        self.assertIsNone(resampler.index_array)
        self.assertIsNone(resampler.distance_array)
        self.assertIsNone(resampler.bilinear_t)
        self.assertIsNone(resampler.bilinear_s)
        self.assertIsNone(resampler.slices_x)
        self.assertIsNone(resampler.slices_y)
        self.assertIsNone(resampler.mask_slices)
        self.assertIsNone(resampler.out_coords_x)
        self.assertIsNone(resampler.out_coords_y)
        # self.slices_{x,y} are used in self.slices dict
        self.assertTrue(resampler.slices['x'] is resampler.slices_x)
        self.assertTrue(resampler.slices['y'] is resampler.slices_y)
        # self.out_coords_{x,y} are used in self.out_coords dict
        self.assertTrue(resampler.out_coords['x'] is resampler.out_coords_x)
        self.assertTrue(resampler.out_coords['y'] is resampler.out_coords_y)

        # Override defaults
        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius, neighbours=16,
                                            epsilon=0.1, reduce_data=False)
        self.assertEqual(resampler.neighbours, 16)
        self.assertEqual(resampler.epsilon, 0.1)
        self.assertFalse(resampler.reduce_data)

    def test_get_bil_info(self):
        """Test calculation of bilinear info."""
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        def _check_ts(t__, s__, nans):
            for i, _ in enumerate(t__):
                # Just check the exact value for one pixel
                if i == 5:
                    self.assertAlmostEqual(t__[i], 0.730659147133, 5)
                    self.assertAlmostEqual(s__[i], 0.310314173004, 5)
                # These pixels are outside the area
                elif i in nans:
                    self.assertTrue(np.isnan(t__[i]))
                    self.assertTrue(np.isnan(s__[i]))
                # All the others should have values between 0.0 and 1.0
                else:
                    self.assertTrue(t__[i] >= 0.0)
                    self.assertTrue(s__[i] >= 0.0)
                    self.assertTrue(t__[i] <= 1.0)
                    self.assertTrue(s__[i] <= 1.0)

        # Data reduction enabled (default)
        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius, reduce_data=True)
        (t__, s__, slices, mask_slices, out_coords) = resampler.get_bil_info()
        _check_ts(t__.compute(), s__.compute(), [3, 10, 12, 13, 14, 15])

        # Nothing should be masked based on coordinates
        self.assertTrue(np.all(~mask_slices))
        # Four values per output location
        self.assertEqual(mask_slices.shape, (self.target_def.size, 4))

        # self.slices_{x,y} are used in self.slices dict so they
        # should be the same (object)
        self.assertTrue(isinstance(slices, dict))
        self.assertTrue(resampler.slices['x'] is resampler.slices_x)
        self.assertTrue(np.all(resampler.slices['x'] == slices['x']))
        self.assertTrue(resampler.slices['y'] is resampler.slices_y)
        self.assertTrue(np.all(resampler.slices['y'] == slices['y']))

        # self.slices_{x,y} are used in self.slices dict so they
        # should be the same (object)
        self.assertTrue(isinstance(out_coords, dict))
        self.assertTrue(resampler.out_coords['x'] is resampler.out_coords_x)
        self.assertTrue(np.all(resampler.out_coords['x'] == out_coords['x']))
        self.assertTrue(resampler.out_coords['y'] is resampler.out_coords_y)
        self.assertTrue(np.all(resampler.out_coords['y'] == out_coords['y']))

        # Also some other attributes should have been set
        self.assertTrue(t__ is resampler.bilinear_t)
        self.assertTrue(s__ is resampler.bilinear_s)
        self.assertIsNotNone(resampler.valid_output_index)
        self.assertIsNotNone(resampler.index_array)
        self.assertIsNotNone(resampler.valid_input_index)

        # Data reduction disabled
        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius, reduce_data=False)
        (t__, s__, slices, mask_slices, out_coords) = resampler.get_bil_info()
        _check_ts(t__.compute(), s__.compute(), [10, 12, 13, 14, 15])

    def test_get_sample_from_bil_info(self):
        """Test bilinear interpolation as a whole."""
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)
        _ = resampler.get_bil_info()

        # Sample from data1
        res = resampler.get_sample_from_bil_info(self.data1)
        res = res.compute()
        # Check couple of values
        self.assertEqual(res.values[1, 1], 1.)
        self.assertTrue(np.isnan(res.values[0, 3]))
        # Check that the values haven't gone down or up a lot
        self.assertAlmostEqual(np.nanmin(res.values), 1.)
        self.assertAlmostEqual(np.nanmax(res.values), 1.)
        # Check that dimensions are the same
        self.assertEqual(res.dims, self.data1.dims)

        # Sample from data1, custom fill value
        res = resampler.get_sample_from_bil_info(self.data1, fill_value=-1.0)
        res = res.compute()
        self.assertEqual(np.nanmin(res.values), -1.)

        # Sample from integer data
        res = resampler.get_sample_from_bil_info(self.data1.astype(np.uint8),
                                                 fill_value=None)
        res = res.compute()
        # Five values should be filled with zeros, which is the
        # default fill_value for integer data
        self.assertEqual(np.sum(res == 0), 6)

    @mock.patch('pyresample.bilinear.xarr.setattr')
    def test_compute_indices(self, mock_setattr):
        """Test running .compute() for indices."""
        from pyresample.bilinear.xarr import (XArrayResamplerBilinear,
                                              CACHE_INDICES)

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)

        # Set indices to Numpy arrays
        for idx in CACHE_INDICES:
            setattr(resampler, idx, np.array([]))
        resampler._compute_indices()
        # None of the indices shouldn't have been reassigned
        mock_setattr.assert_not_called()

        # Set indices to a Mock object
        arr = mock.MagicMock()
        for idx in CACHE_INDICES:
            setattr(resampler, idx, arr)
        resampler._compute_indices()
        # All the indices should have been reassigned
        self.assertEqual(mock_setattr.call_count, len(CACHE_INDICES))
        # The compute should have been called the same amount of times
        self.assertEqual(arr.compute.call_count, len(CACHE_INDICES))

    def test_add_missing_coordinates(self):
        """Test coordinate updating."""
        import dask.array as da
        from xarray import DataArray
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)
        bands = ['R', 'G', 'B']
        data = DataArray(da.ones((3, 10, 10)), dims=('bands', 'y', 'x'),
                         coords={'bands': bands,
                                 'y': np.arange(10), 'x': np.arange(10)})
        resampler._add_missing_coordinates(data)
        # X and Y coordinates should not change
        self.assertIsNone(resampler.out_coords_x)
        self.assertIsNone(resampler.out_coords_y)
        self.assertIsNone(resampler.out_coords['x'])
        self.assertIsNone(resampler.out_coords['y'])
        self.assertTrue('bands' in resampler.out_coords)
        self.assertTrue(np.all(resampler.out_coords['bands'] == bands))

    def test_slice_data(self):
        """Test slicing the data."""
        import dask.array as da
        from xarray import DataArray
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)

        # Too many dimensions
        data = DataArray(da.ones((1, 3, 10, 10)))
        with self.assertRaises(ValueError):
            _ = resampler._slice_data(data, np.nan)

        # 2D data
        data = DataArray(da.ones((10, 10)))
        resampler.slices_x = np.random.randint(0, 10, (100, 4))
        resampler.slices_y = np.random.randint(0, 10, (100, 4))
        resampler.mask_slices = np.zeros((100, 4), dtype=np.bool)
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertEqual(p_1.shape, (100, ))
        self.assertTrue(p_1.shape == p_2.shape == p_3.shape == p_4.shape)
        self.assertTrue(np.all(p_1 == 1.0) and np.all(p_2 == 1.0) and
                        np.all(p_3 == 1.0) and np.all(p_4 == 1.0))

        # 2D data with masking
        resampler.mask_slices = np.ones((100, 4), dtype=np.bool)
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertTrue(np.all(np.isnan(p_1)) and np.all(np.isnan(p_2)) and
                        np.all(np.isnan(p_3)) and np.all(np.isnan(p_4)))

        # 3D data
        data = DataArray(da.ones((3, 10, 10)))
        resampler.slices_x = np.random.randint(0, 10, (100, 4))
        resampler.slices_y = np.random.randint(0, 10, (100, 4))
        resampler.mask_slices = np.zeros((100, 4), dtype=np.bool)
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertEqual(p_1.shape, (3, 100))
        self.assertTrue(p_1.shape == p_2.shape == p_3.shape == p_4.shape)

        # 3D data with masking
        resampler.mask_slices = np.ones((100, 4), dtype=np.bool)
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertTrue(np.all(np.isnan(p_1)) and np.all(np.isnan(p_2)) and
                        np.all(np.isnan(p_3)) and np.all(np.isnan(p_4)))

    @mock.patch('pyresample.bilinear.xarr.np.meshgrid')
    def test_get_slices(self, meshgrid):
        """Test slice array creation."""
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        meshgrid.return_value = (self.cols, self.lines)

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)
        resampler.valid_input_index = self.valid_input_index
        resampler.index_array = self.index_array

        resampler._get_slices()
        self.assertIsNotNone(resampler.out_coords_x)
        self.assertIsNotNone(resampler.out_coords_y)
        self.assertTrue(resampler.out_coords_x is resampler.out_coords['x'])
        self.assertTrue(resampler.out_coords_y is resampler.out_coords['y'])
        self.assertTrue(np.allclose(
            resampler.out_coords_x,
            [-1070912.72, -470912.72, 129087.28, 729087.28]))
        self.assertTrue(np.allclose(
            resampler.out_coords_y,
            [1190031.36,  590031.36,   -9968.64, -609968.64]))

        self.assertIsNotNone(resampler.slices_x)
        self.assertIsNotNone(resampler.slices_y)
        self.assertTrue(resampler.slices_x is resampler.slices['x'])
        self.assertTrue(resampler.slices_y is resampler.slices['y'])
        self.assertTrue(resampler.slices_x.shape == (self.target_def.size, 32))
        self.assertTrue(resampler.slices_y.shape == (self.target_def.size, 32))
        self.assertEqual(np.sum(resampler.slices_x), 12471)
        self.assertEqual(np.sum(resampler.slices_y), 2223)

        self.assertFalse(np.any(resampler.mask_slices))

        # Ensure that source geo def is used in masking
        # Setting target_geo_def to 0-size shouldn't cause any masked values
        resampler.target_geo_def = np.array([])
        resampler._get_slices()
        self.assertFalse(np.any(resampler.mask_slices))
        # Setting source area def to 0-size should mask all values
        resampler.source_geo_def = np.array([[]])
        resampler._get_slices()
        self.assertTrue(np.all(resampler.mask_slices))

    @mock.patch('pyresample.bilinear.xarr.KDTree')
    def test_create_resample_kdtree(self, KDTree):
        """Test that KDTree creation is called."""
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)

        vii, kdtree = resampler._create_resample_kdtree()
        self.assertEqual(np.sum(vii), 2700)
        self.assertEqual(vii.size, self.source_def.size)
        KDTree.assert_called_once()

    @mock.patch('pyresample.bilinear.xarr.query_no_distance')
    def test_query_resample_kdtree(self, qnd):
        """Test that query_no_distance is called in _query_resample_kdtree()."""
        from pyresample.bilinear.xarr import XArrayResamplerBilinear

        resampler = XArrayResamplerBilinear(self.source_def, self.target_def,
                                            self.radius)
        res, none = resampler._query_resample_kdtree(1, 2, 3, 4,
                                                     reduce_data=5)
        qnd.assert_called_with(2, 3, 4, 1, resampler.neighbours,
                               resampler.epsilon,
                               resampler.radius_of_influence)

    def test_get_input_xy_dask(self):
        """Test computation of input X and Y coordinates in target proj."""
        import dask.array as da
        from pyresample.bilinear.xarr import _get_input_xy_dask
        from pyresample._spatial_mp import Proj

        proj = Proj(self.target_def.proj_str)
        in_x, in_y = _get_input_xy_dask(self.source_def, proj,
                                        da.from_array(self.valid_input_index),
                                        da.from_array(self.index_array))

        self.assertTrue(in_x.shape, (self.target_def.size, 32))
        self.assertTrue(in_y.shape, (self.target_def.size, 32))
        self.assertTrue(in_x.all())
        self.assertTrue(in_y.all())

    def test_mask_coordinates_dask(self):
        """Test masking of invalid coordinates."""
        import dask.array as da
        from pyresample.bilinear.xarr import _mask_coordinates_dask

        lons, lats = _mask_coordinates_dask(
            da.from_array([-200., 0., 0., 0., 200.]),
            da.from_array([0., -100., 0, 100., 0.]))
        lons, lats = da.compute(lons, lats)
        self.assertTrue(lons[2] == lats[2] == 0.0)
        self.assertEqual(np.sum(np.isnan(lons)), 4)
        self.assertEqual(np.sum(np.isnan(lats)), 4)

    def test_get_bounding_corners_dask(self):
        """Test finding surrounding bounding corners."""
        import dask.array as da
        from pyresample.bilinear.xarr import (_get_input_xy_dask,
                                              _get_bounding_corners_dask)
        from pyresample._spatial_mp import Proj
        from pyresample import CHUNK_SIZE

        proj = Proj(self.target_def.proj_str)
        out_x, out_y = self.target_def.get_proj_coords(chunks=CHUNK_SIZE)
        out_x = da.ravel(out_x)
        out_y = da.ravel(out_y)
        in_x, in_y = _get_input_xy_dask(self.source_def, proj,
                                        da.from_array(self.valid_input_index),
                                        da.from_array(self.index_array))
        pt_1, pt_2, pt_3, pt_4, ia_ = _get_bounding_corners_dask(
            in_x, in_y, out_x, out_y,
            self.neighbours,
            da.from_array(self.index_array))

        self.assertTrue(pt_1.shape == pt_2.shape ==
                        pt_3.shape == pt_4.shape ==
                        (self.target_def.size, 2))
        self.assertTrue(ia_.shape == (self.target_def.size, 4))

        # Check which of the locations has four valid X/Y pairs by
        # finding where there are non-NaN values
        res = da.sum(pt_1 + pt_2 + pt_3 + pt_4, axis=1).compute()
        self.assertEqual(np.sum(~np.isnan(res)), 10)

    def test_get_corner_dask(self):
        """Test finding the closest corners."""
        import dask.array as da
        from pyresample.bilinear.xarr import (_get_corner_dask,
                                              _get_input_xy_dask)
        from pyresample import CHUNK_SIZE
        from pyresample._spatial_mp import Proj

        proj = Proj(self.target_def.proj_str)
        in_x, in_y = _get_input_xy_dask(self.source_def, proj,
                                        da.from_array(self.valid_input_index),
                                        da.from_array(self.index_array))
        out_x, out_y = self.target_def.get_proj_coords(chunks=CHUNK_SIZE)
        out_x = da.ravel(out_x)
        out_y = da.ravel(out_y)

        # Some copy&paste from the code to get the input
        out_x_tile = np.reshape(np.tile(out_x, self.neighbours),
                                (self.neighbours, out_x.size)).T
        out_y_tile = np.reshape(np.tile(out_y, self.neighbours),
                                (self.neighbours, out_y.size)).T
        x_diff = out_x_tile - in_x
        y_diff = out_y_tile - in_y
        stride = np.arange(x_diff.shape[0])

        # Use lower left source pixels for testing
        valid = (x_diff > 0) & (y_diff > 0)
        x_3, y_3, idx_3 = _get_corner_dask(stride, valid, in_x, in_y,
                                           da.from_array(self.index_array))

        self.assertTrue(x_3.shape == y_3.shape == idx_3.shape ==
                        (self.target_def.size, ))
        # Four locations have no data to the lower left of them (the
        # bottom row of the area
        self.assertEqual(np.sum(np.isnan(x_3.compute())), 4)

    @mock.patch('pyresample.bilinear.xarr._get_ts_parallellogram_dask')
    @mock.patch('pyresample.bilinear.xarr._get_ts_uprights_parallel_dask')
    @mock.patch('pyresample.bilinear.xarr._get_ts_irregular_dask')
    def test_get_ts_dask(self, irregular, uprights, parallellogram):
        """Test that the three separate functions are called."""
        from pyresample.bilinear.xarr import _get_ts_dask

        # All valid values
        t_irr = np.array([0.1, 0.2, 0.3])
        s_irr = np.array([0.1, 0.2, 0.3])
        irregular.return_value = (t_irr, s_irr)
        t__, s__ = _get_ts_dask(1, 2, 3, 4, 5, 6)
        irregular.assert_called_once()
        uprights.assert_not_called()
        parallellogram.assert_not_called()
        self.assertTrue(np.allclose(t__.compute(), t_irr))
        self.assertTrue(np.allclose(s__.compute(), s_irr))

        # NaN in the first step, good value for that location from the
        # second step
        t_irr = np.array([0.1, 0.2, np.nan])
        s_irr = np.array([0.1, 0.2, np.nan])
        irregular.return_value = (t_irr, s_irr)
        t_upr = np.array([3, 3, 0.3])
        s_upr = np.array([3, 3, 0.3])
        uprights.return_value = (t_upr, s_upr)
        t__, s__ = _get_ts_dask(1, 2, 3, 4, 5, 6)
        self.assertEqual(irregular.call_count, 2)
        uprights.assert_called_once()
        parallellogram.assert_not_called()
        # Only the last value of the first step should have been replaced
        t_res = np.array([0.1, 0.2, 0.3])
        s_res = np.array([0.1, 0.2, 0.3])
        self.assertTrue(np.allclose(t__.compute(), t_res))
        self.assertTrue(np.allclose(s__.compute(), s_res))

        # Two NaNs in the first step, one of which are found by the
        # second, and the last bad value is replaced by the third step
        t_irr = np.array([0.1, np.nan, np.nan])
        s_irr = np.array([0.1, np.nan, np.nan])
        irregular.return_value = (t_irr, s_irr)
        t_upr = np.array([3, np.nan, 0.3])
        s_upr = np.array([3, np.nan, 0.3])
        uprights.return_value = (t_upr, s_upr)
        t_par = np.array([4, 0.2, 0.3])
        s_par = np.array([4, 0.2, 0.3])
        parallellogram.return_value = (t_par, s_par)
        t__, s__ = _get_ts_dask(1, 2, 3, 4, 5, 6)
        self.assertEqual(irregular.call_count, 3)
        self.assertEqual(uprights.call_count, 2)
        parallellogram.assert_called_once()
        # Only the last two values should have been replaced
        t_res = np.array([0.1, 0.2, 0.3])
        s_res = np.array([0.1, 0.2, 0.3])
        self.assertTrue(np.allclose(t__.compute(), t_res))
        self.assertTrue(np.allclose(s__.compute(), s_res))

        # Too large and small values should be set to NaN
        t_irr = np.array([1.00001, -0.00001, 1e6])
        s_irr = np.array([1.00001, -0.00001, -1e6])
        irregular.return_value = (t_irr, s_irr)
        # Second step also returns invalid values
        t_upr = np.array([1.00001, 0.2, np.nan])
        s_upr = np.array([-0.00001, 0.2, np.nan])
        uprights.return_value = (t_upr, s_upr)
        # Third step has one new valid value, the last will stay invalid
        t_par = np.array([0.1, 0.2, 4.0])
        s_par = np.array([0.1, 0.2, 4.0])
        parallellogram.return_value = (t_par, s_par)
        t__, s__ = _get_ts_dask(1, 2, 3, 4, 5, 6)

        t_res = np.array([0.1, 0.2, np.nan])
        s_res = np.array([0.1, 0.2, np.nan])
        self.assertTrue(np.allclose(t__.compute(), t_res, equal_nan=True))
        self.assertTrue(np.allclose(s__.compute(), s_res, equal_nan=True))

    def test_get_ts_irregular_dask(self):
        """Test calculations for irregular corner locations."""
        from pyresample.bilinear.xarr import _get_ts_irregular_dask

        res = _get_ts_irregular_dask(self.pts_irregular[0],
                                     self.pts_irregular[1],
                                     self.pts_irregular[2],
                                     self.pts_irregular[3],
                                     0., 0.)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = _get_ts_irregular_dask(self.pts_vert_parallel[0],
                                     self.pts_vert_parallel[1],
                                     self.pts_vert_parallel[2],
                                     self.pts_vert_parallel[3],
                                     0., 0.)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))

    def test_get_ts_uprights_parallel(self):
        """Test calculation when uprights are parallel."""
        from pyresample.bilinear import _get_ts_uprights_parallel

        res = _get_ts_uprights_parallel(self.pts_vert_parallel[0],
                                        self.pts_vert_parallel[1],
                                        self.pts_vert_parallel[2],
                                        self.pts_vert_parallel[3],
                                        0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_ts_parallellogram(self):
        """Test calculation when the corners form a parallellogram."""
        from pyresample.bilinear import _get_ts_parallellogram

        res = _get_ts_parallellogram(self.pts_both_parallel[0],
                                     self.pts_both_parallel[1],
                                     self.pts_both_parallel[2],
                                     0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_calc_abc(self):
        """Test calculation of quadratic coefficients."""
        from pyresample.bilinear.xarr import _calc_abc_dask

        # No np.nan inputs
        pt_1, pt_2, pt_3, pt_4 = self.pts_irregular
        res = _calc_abc_dask(pt_1, pt_2, pt_3, pt_4, 0.0, 0.0)
        self.assertFalse(np.isnan(res[0]))
        self.assertFalse(np.isnan(res[1]))
        self.assertFalse(np.isnan(res[2]))
        # np.nan input -> np.nan output
        res = _calc_abc_dask(np.array([[np.nan, np.nan]]),
                             pt_2, pt_3, pt_4, 0.0, 0.0)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))
        self.assertTrue(np.isnan(res[2]))

    def test_solve_quadratic(self):
        """Test solving quadratic equation."""
        from pyresample.bilinear.xarr import (_solve_quadratic_dask,
                                              _calc_abc_dask)

        res = _solve_quadratic_dask(1, 0, 0).compute()
        self.assertEqual(res, 0.0)
        res = _solve_quadratic_dask(1, 2, 1).compute()
        self.assertTrue(np.isnan(res))
        res = _solve_quadratic_dask(1, 2, 1, min_val=-2.).compute()
        self.assertEqual(res, -1.0)
        # Test that small adjustments work
        pt_1, pt_2, pt_3, pt_4 = self.pts_vert_parallel
        pt_1 = self.pts_vert_parallel[0].copy()
        pt_1[0][0] += 1e-7
        res = _calc_abc_dask(pt_1, pt_2, pt_3, pt_4, 0.0, 0.0)
        res = _solve_quadratic_dask(res[0], res[1], res[2]).compute()
        self.assertAlmostEqual(res[0], 0.5, 5)
        res = _calc_abc_dask(pt_1, pt_3, pt_2, pt_4, 0.0, 0.0)
        res = _solve_quadratic_dask(res[0], res[1], res[2]).compute()
        self.assertAlmostEqual(res[0], 0.5, 5)

    def test_query_no_distance(self):
        """Test KDTree querying."""
        from pyresample.bilinear.xarr import query_no_distance

        kdtree = mock.MagicMock()
        kdtree.query.return_value = (1, 2)
        lons, lats = self.target_def.get_lonlats()
        voi = (lons >= -180) & (lons <= 180) & (lats <= 90) & (lats >= -90)
        res = query_no_distance(lons, lats, voi, kdtree, self.neighbours,
                                0., self.radius)
        # Only the second value from the query is returned
        self.assertEqual(res, 2)
        kdtree.query.assert_called_once()

    def test_get_valid_input_index_dask(self):
        """Test finding valid indices for reduced input data."""
        from pyresample.bilinear.xarr import _get_valid_input_index_dask

        # Do not reduce data
        vii, lons, lats = _get_valid_input_index_dask(self.source_def,
                                                      self.target_def,
                                                      False, self.radius)
        self.assertEqual(vii.shape, (self.source_def.size, ))
        self.assertTrue(vii.dtype == np.bool)
        # No data has been reduced, whole input is used
        self.assertTrue(vii.compute().all())

        # Reduce data
        vii, lons, lats = _get_valid_input_index_dask(self.source_def,
                                                      self.target_def,
                                                      True, self.radius)
        # 2700 valid input points
        self.assertEqual(vii.compute().sum(), 2700)

    def test_create_empty_bil_info(self):
        """Test creation of empty bilinear info."""
        from pyresample.bilinear.xarr import _create_empty_bil_info

        t__, s__, vii, ia_ = _create_empty_bil_info(self.source_def,
                                                    self.target_def)
        self.assertEqual(t__.shape, (self.target_def.size,))
        self.assertEqual(s__.shape, (self.target_def.size,))
        self.assertEqual(ia_.shape, (self.target_def.size, 4))
        self.assertTrue(ia_.dtype == np.int32)
        self.assertEqual(vii.shape, (self.source_def.size,))
        self.assertTrue(vii.dtype == np.bool)

    def test_lonlat2xyz(self):
        """Test conversion from geographic to cartesian 3D coordinates."""
        from pyresample.bilinear.xarr import lonlat2xyz
        from pyresample import CHUNK_SIZE

        lons, lats = self.target_def.get_lonlats(chunks=CHUNK_SIZE)
        res = lonlat2xyz(lons, lats)
        self.assertEqual(res.shape, (self.target_def.size, 3))
        vals = [3188578.91069278, -612099.36103276, 5481596.63569999]
        self.assertTrue(np.allclose(res.compute()[0, :], vals))
