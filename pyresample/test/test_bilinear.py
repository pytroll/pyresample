#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2020 Pyresample developers.
#
# This file is part of Pyresample
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test bilinear interpolation."""
import unittest
from unittest import mock

import numpy as np
from pyproj import Proj


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
        cls.data3_1d = np.ravel(cls.data3)

        lons, lats = np.meshgrid(np.linspace(-25., 40., num=in_shape[0]),
                                 np.linspace(45., 75., num=in_shape[1]))
        cls.source_def = geometry.SwathDefinition(lons=lons, lats=lats)
        cls.source_def_1d = geometry.SwathDefinition(lons=np.ravel(lons),
                                                     lats=np.ravel(lats))

        cls.radius = 50e3
        cls._neighbours = 32
        input_idxs, output_idxs, idx_ref, dists = \
            kd_tree.get_neighbour_info(cls.source_def, target_def,
                                       cls.radius, neighbours=cls._neighbours,
                                       nprocs=1)
        input_size = input_idxs.sum()
        index_mask = (idx_ref == input_size)
        idx_ref = np.where(index_mask, 0, idx_ref)

        cls.input_idxs = input_idxs
        cls.target_def = target_def
        cls.idx_ref = idx_ref

    def test_init(self):
        """Test that the resampler has been initialized correctly."""
        from pyresample.bilinear import NumpyBilinearResampler

        resampler = NumpyBilinearResampler(self.source_def, self.target_def,
                                           self.radius)
        self.assertTrue(resampler._source_geo_def == self.source_def)
        self.assertTrue(resampler._target_geo_def == self.target_def)
        self.assertEqual(resampler._radius_of_influence, self.radius)
        self.assertEqual(resampler._neighbours, 32)
        self.assertEqual(resampler._epsilon, 0)
        self.assertTrue(resampler._reduce_data)
        # These should be None
        self.assertIsNone(resampler._valid_input_index)
        self.assertIsNone(resampler._index_array)
        self.assertIsNone(resampler._distance_array)
        self.assertIsNone(resampler.bilinear_t)
        self.assertIsNone(resampler.bilinear_s)
        self.assertIsNone(resampler.slices_x)
        self.assertIsNone(resampler.slices_y)
        self.assertIsNone(resampler.mask_slices)
        self.assertIsNone(resampler.out_coords_x)
        self.assertIsNone(resampler.out_coords_y)

        # Override defaults
        resampler = NumpyBilinearResampler(self.source_def, self.target_def,
                                           self.radius, neighbours=16,
                                           epsilon=0.1, reduce_data=False)
        self.assertEqual(resampler._neighbours, 16)
        self.assertEqual(resampler._epsilon, 0.1)
        self.assertFalse(resampler._reduce_data)

    def test_calc_abc(self):
        """Test calculation of quadratic coefficients."""
        from pyresample.bilinear._base import _calc_abc

        # No np.nan inputs
        res = _calc_abc(self.pts_irregular, 0.0, 0.0)
        self.assertFalse(np.isnan(res[0]))
        self.assertFalse(np.isnan(res[1]))
        self.assertFalse(np.isnan(res[2]))
        # np.nan input -> np.nan output
        pt_1, pt_2, pt_3, pt_4 = self.pts_irregular
        corner_points = (np.array([[np.nan, np.nan]]), pt_2, pt_3, pt_4)
        res = _calc_abc(corner_points, 0.0, 0.0)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))
        self.assertTrue(np.isnan(res[2]))

    def test_get_fractional_distances_irregular(self):
        """Test calculations for irregular corner locations."""
        from pyresample.bilinear._base import _get_fractional_distances_irregular

        res = _get_fractional_distances_irregular(self.pts_irregular, 0., 0.)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = _get_fractional_distances_irregular(self.pts_vert_parallel, 0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertTrue(np.isnan(res[1]))

    def test_get_fractional_distances_uprights_parallel(self):
        """Test calculation when uprights are parallel."""
        from pyresample.bilinear._base import _get_fractional_distances_uprights_parallel

        res = _get_fractional_distances_uprights_parallel(self.pts_vert_parallel, 0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_fractional_distances_parallellogram(self):
        """Test calculation when the corners form a parallellogram."""
        from pyresample.bilinear._base import _get_fractional_distances_parallellogram

        res = _get_fractional_distances_parallellogram(self.pts_both_parallel[:3], 0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_fractional_distances(self):
        """Test get_ts()."""
        from pyresample.bilinear._base import _get_fractional_distances

        out_x = np.array([[0.]])
        out_y = np.array([[0.]])
        res = _get_fractional_distances(self.pts_irregular, out_x, out_y)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = _get_fractional_distances(self.pts_both_parallel, out_x, out_y)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)
        res = _get_fractional_distances(self.pts_vert_parallel, out_x, out_y)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_fractional_distances_division_by_zero(self):
        """Test that the correct result is found even when there's a division by zero when solving t and s."""
        from pyresample.bilinear._base import _get_fractional_distances
        corner_points = [np.array([[-64.9936752319336, -5.140199184417725]]),
                         np.array([[-64.98487091064453, -5.142156600952148]]),
                         np.array([[-64.98683166503906, -5.151054859161377]]),
                         np.array([[-64.97802734375, -5.153012275695801]])]
        out_x = np.array([-64.985])
        out_y = np.array([-5.145])

        t__, s__ = _get_fractional_distances(corner_points, out_x, out_y)
        np.testing.assert_allclose(t__, np.array([0.30769689]))
        np.testing.assert_allclose(s__, np.array([0.74616628]))

    def test_solve_quadratic(self):
        """Test solving quadratic equation."""
        from pyresample.bilinear._base import _calc_abc, _solve_quadratic

        res = _solve_quadratic(1, 0, 0)
        self.assertEqual(res, 0.0)
        res = _solve_quadratic(1, 2, 1)
        self.assertTrue(np.isnan(res))
        res = _solve_quadratic(1, 2, 1, min_val=-2.)
        self.assertEqual(res, -1.0)
        # Test that small adjustments work
        pt_1, pt_2, pt_3, pt_4 = self.pts_vert_parallel
        pt_1 = self.pts_vert_parallel[0].copy()
        pt_1[0][0] += 1e-7
        corner_points = (pt_1, pt_2, pt_3, pt_4)
        res = _calc_abc(corner_points, 0.0, 0.0)
        res = _solve_quadratic(res[0], res[1], res[2])
        self.assertAlmostEqual(res[0], 0.5, 5)
        corner_points = (pt_1, pt_3, pt_2, pt_4)
        res = _calc_abc(corner_points, 0.0, 0.0)
        res = _solve_quadratic(res[0], res[1], res[2])
        self.assertAlmostEqual(res[0], 0.5, 5)

    def test_get_input_xy(self):
        """Test calculation of input xy-coordinates."""
        from pyresample.bilinear._base import _get_input_xy

        proj = Proj(self.target_def.proj_str)
        in_x, in_y = _get_input_xy(self.source_def, proj,
                                   self.input_idxs, self.idx_ref)
        self.assertTrue(in_x.all())
        self.assertTrue(in_y.all())

    def test_get_four_closest_corners(self):
        """Test calculation of bounding corners."""
        from pyresample.bilinear._base import _get_four_closest_corners, _get_input_xy, _get_output_xy

        proj = Proj(self.target_def.proj_str)
        out_x, out_y = _get_output_xy(self.target_def)
        in_x, in_y = _get_input_xy(self.source_def, proj,
                                   self.input_idxs, self.idx_ref)
        (pt_1, pt_2, pt_3, pt_4), ia_ = _get_four_closest_corners(
            in_x, in_y, out_x, out_y,
            self._neighbours,
            self.idx_ref)

        self.assertTrue(pt_1.shape == pt_2.shape ==
                        pt_3.shape == pt_4.shape ==
                        (self.target_def.size, 2))
        self.assertTrue(ia_.shape == (self.target_def.size, 4))

        # Check which of the locations has four valid X/Y pairs by
        # finding where there are non-NaN values
        res = np.sum(pt_1 + pt_2 + pt_3 + pt_4, axis=1)
        self.assertEqual(np.sum(~np.isnan(res)), 10)

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

        t__, s__, _, _ = get_bil_info(self.source_def,
                                      self.target_def,
                                      50e5, neighbours=32,
                                      nprocs=1,
                                      reduce_data=False)
        _check_ts(t__, s__)

        t__, s__, _, _ = get_bil_info(self.source_def,
                                      self.target_def,
                                      50e5, neighbours=32,
                                      nprocs=2,
                                      reduce_data=True)
        _check_ts(t__, s__)

    def test_get_sample_from_bil_info(self):
        """Test resampling using resampling indices."""
        from pyresample.bilinear import get_bil_info, get_sample_from_bil_info

        t__, s__, input_idxs, idx_arr = get_bil_info(self.source_def,
                                                     self.target_def,
                                                     50e5, neighbours=32,
                                                     nprocs=1)
        # Sample from data1
        res = get_sample_from_bil_info(self.data1.ravel(), t__, s__,
                                       input_idxs, idx_arr)
        self.assertAlmostEqual(res.ravel()[5], 1.)
        # Sample from data2
        res = get_sample_from_bil_info(self.data2.ravel(), t__, s__,
                                       input_idxs, idx_arr)
        self.assertAlmostEqual(res.ravel()[5], 2.)
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
        self.assertAlmostEqual(np.isnan(res).sum(), 4)

        # Masked array as input, result should be plain Numpy array
        data = np.ma.masked_all(self.data1.shape)
        res = get_sample_from_bil_info(data.ravel(), t__, s__,
                                       input_idxs, idx_arr)
        assert not hasattr(res, 'mask')

    def test_get_sample_from_bil_info_1d(self):
        """Test resampling using resampling indices for 1D data."""
        from pyresample.bilinear import get_bil_info, get_sample_from_bil_info

        t__, s__, input_idxs, idx_arr = get_bil_info(self.source_def_1d,
                                                     self.target_def,
                                                     50e5, neighbours=32,
                                                     nprocs=1)
        # Sample from 1D data
        res = get_sample_from_bil_info(self.data3_1d, t__, s__,
                                       input_idxs, idx_arr)
        self.assertAlmostEqual(np.nanmin(res), 10.5)
        self.assertAlmostEqual(np.nanmax(res), 10.5)
        # Four pixels are outside of the data
        self.assertEqual(np.isnan(res).sum(), 4)

    def test_resample_bilinear(self):
        """Test whole bilinear resampling."""
        from pyresample.bilinear import resample_bilinear

        # Single array
        res = resample_bilinear(self.data1,
                                self.source_def,
                                self.target_def,
                                50e5, neighbours=32,
                                nprocs=1)
        self.assertEqual(res.shape, self.target_def.shape)
        # There are 12 pixels with value 1, all others are zero
        self.assertEqual(res.sum(), 12)
        self.assertEqual((res == 0).sum(), 4)

        # Single array with masked output
        res = resample_bilinear(self.data1,
                                self.source_def,
                                self.target_def,
                                50e5, neighbours=32,
                                nprocs=1, fill_value=None)
        self.assertTrue(hasattr(res, 'mask'))
        # There should be 12 valid pixels
        self.assertEqual(self.target_def.size - res.mask.sum(), 12)

        # Two stacked arrays, multiprocessing
        data = np.dstack((self.data1, self.data2))
        res = resample_bilinear(data,
                                self.source_def,
                                self.target_def,
                                nprocs=2)
        shp = res.shape
        self.assertEqual(shp[0:2], self.target_def.shape)
        self.assertEqual(shp[-1], 2)

    def test_class_resample_method(self):
        """Test the 'resampler.resample()' method."""
        from pyresample.bilinear import NumpyBilinearResampler

        resampler = NumpyBilinearResampler(self.source_def,
                                           self.target_def,
                                           50e5,
                                           neighbours=32,
                                           epsilon=0)

        # Single array, no fill value
        res = resampler.resample(self.data1)
        self.assertEqual(res.shape, self.target_def.shape)
        # There are 12 pixels with value 1, all others are zero
        self.assertEqual(res.sum(), 12)
        self.assertEqual((res == 0).sum(), 4)

        # Single array with masked output
        res = resampler.resample(self.data1, fill_value=None)
        self.assertTrue(hasattr(res, 'mask'))
        # There should be 12 valid pixels
        self.assertEqual(self.target_def.size - res.mask.sum(), 12)

        # Two stacked arrays, multiprocessing
        data = np.dstack((self.data1, self.data2))
        res = resampler.resample(data, nprocs=2)
        shp = res.shape
        self.assertEqual(shp[0:2], self.target_def.shape)
        self.assertEqual(shp[-1], 2)

    def test_create_empty_bil_info(self):
        """Test creation of empty bilinear info."""
        from pyresample.bilinear import NumpyBilinearResampler

        resampler = NumpyBilinearResampler(self.source_def, self.target_def,
                                           self.radius)

        resampler._create_empty_bil_info()
        self.assertEqual(resampler.bilinear_t.shape, (self.target_def.size,))
        self.assertEqual(resampler.bilinear_s.shape, (self.target_def.size,))
        self.assertEqual(resampler._index_array.shape, (self.target_def.size, 4))
        self.assertTrue(resampler._index_array.dtype == np.int32)
        self.assertEqual(resampler._valid_input_index.shape, (self.source_def.size,))
        self.assertTrue(resampler._valid_input_index.dtype == bool)


class TestXarrayBilinear(unittest.TestCase):
    """Test Xarra/Dask -based bilinear interpolation."""

    def setUp(self):
        """Do some setup for common things."""
        import dask.array as da
        from xarray import DataArray

        from pyresample import geometry, kd_tree

        self.pts_irregular = (da.array([[-1., 1.], ]),
                              da.array([[1., 2.], ]),
                              da.array([[-2., -1.], ]),
                              da.array([[2., -4.], ]))
        self.pts_vert_parallel = (da.array([[-1., 1.], ]),
                                  da.array([[1., 2.], ]),
                                  da.array([[-1., -1.], ]),
                                  da.array([[1., -2.], ]))
        self.pts_both_parallel = (da.array([[-1., 1.], ]),
                                  da.array([[1., 1.], ]),
                                  da.array([[-1., -1.], ]),
                                  da.array([[1., -1.], ]))

        # Area definition with 4x4 pixels
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
        # Area definition with 8x8 pixels, does not intersect with source data
        self.target_def_outside = geometry.AreaDefinition('area_outside',
                                                          'area outside the input data',
                                                          'ease_sh',
                                                          {'a': '6371228.0',
                                                           'lat_0': '-90.0',
                                                           'lon_0': '0.0',
                                                           'proj': 'laea'},
                                                          8, 8,
                                                          [-5326849.0625, -5326849.0625,
                                                           5326849.0625, 5326849.0625])

        # Area that partially overlaps the source data
        self.target_def_partial = geometry.AreaDefinition('area_partial_overlap',
                                                          'Europe (3km, HRV, VTC)',
                                                          'areaD',
                                                          {'a': '6378144.0',
                                                           'b': '6356759.0',
                                                           'lat_0': '50.00',
                                                           'lat_ts': '50.00',
                                                           'lon_0': '8.00',
                                                           'proj': 'stere'},
                                                          4, 4,
                                                          [59559.320999999996,
                                                           -909968.64000000001,
                                                           2920503.401,
                                                           1490031.3600000001])

        # Input data around the target pixel at 0.63388324, 55.08234642,
        in_shape = (100, 100)
        self.data1 = DataArray(da.ones((in_shape[0], in_shape[1])), dims=('y', 'x'))
        self.data2 = 2. * self.data1
        self.data3 = self.data1 + 9.5

        lons, lats = np.meshgrid(np.linspace(-25., 40., num=in_shape[0]),
                                 np.linspace(45., 75., num=in_shape[1]))
        self.source_def = geometry.SwathDefinition(lons=lons, lats=lats)
        self.source_def_1d = geometry.SwathDefinition(lons=np.ravel(lons),
                                                      lats=np.ravel(lats))

        self.radius = 50e3
        self._neighbours = 32
        valid_input_index, output_idxs, index_array, dists = \
            kd_tree.get_neighbour_info(self.source_def, self.target_def,
                                       self.radius, neighbours=self._neighbours,
                                       nprocs=1)
        input_size = valid_input_index.sum()
        index_mask = (index_array == input_size)
        index_array = np.where(index_mask, 0, index_array)

        self._valid_input_index = valid_input_index
        self._index_array = index_array

        shp = self.source_def.shape
        self.cols, self.lines = np.meshgrid(np.arange(shp[1]),
                                            np.arange(shp[0]))

    def test_init(self):
        """Test that the resampler has been initialized correctly."""
        from pyresample.bilinear import XArrayBilinearResampler

        # With defaults
        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        self.assertTrue(resampler._source_geo_def == self.source_def)
        self.assertTrue(resampler._target_geo_def == self.target_def)
        self.assertEqual(resampler._radius_of_influence, self.radius)
        self.assertEqual(resampler._neighbours, 32)
        self.assertEqual(resampler._epsilon, 0)
        self.assertTrue(resampler._reduce_data)
        # These should be None
        self.assertIsNone(resampler._valid_input_index)
        self.assertIsNone(resampler._index_array)
        self.assertIsNone(resampler._distance_array)
        self.assertIsNone(resampler.bilinear_t)
        self.assertIsNone(resampler.bilinear_s)
        self.assertIsNone(resampler.slices_x)
        self.assertIsNone(resampler.slices_y)
        self.assertIsNone(resampler.mask_slices)
        self.assertIsNone(resampler.out_coords_x)
        self.assertIsNone(resampler.out_coords_y)
        # self._out_coords_{x,y} are used in self._out_coords dict
        self.assertTrue(np.all(resampler._out_coords['x'] == resampler.out_coords_x))
        self.assertTrue(np.all(resampler._out_coords['y'] == resampler.out_coords_y))

        # Override defaults
        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius, neighbours=16,
                                            epsilon=0.1, reduce_data=False)
        self.assertEqual(resampler._neighbours, 16)
        self.assertEqual(resampler._epsilon, 0.1)
        self.assertFalse(resampler._reduce_data)

    def test_get_bil_info(self):
        """Test calculation of bilinear info."""
        from pyresample.bilinear import XArrayBilinearResampler

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
        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius, reduce_data=True)
        resampler.get_bil_info()
        t__ = resampler.bilinear_t
        s__ = resampler.bilinear_s
        mask_slices = resampler.mask_slices
        _check_ts(t__, s__, [3, 10, 12, 13, 14, 15])

        # Nothing should be masked based on coordinates
        self.assertTrue(np.all(~mask_slices))
        # Four values per output location
        self.assertEqual(mask_slices.shape, (self.target_def.size, 4))

        # Also some other attributes should have been set
        self.assertTrue(t__ is resampler.bilinear_t)
        self.assertTrue(s__ is resampler.bilinear_s)
        self.assertIsNotNone(resampler._index_array)
        self.assertIsNotNone(resampler._valid_input_index)
        self.assertIsNotNone(resampler.out_coords_x)
        self.assertIsNotNone(resampler.out_coords_y)
        self.assertTrue(np.allclose(
            resampler.out_coords_x,
            [-1070912.72, -470912.72, 129087.28, 729087.28]))
        self.assertTrue(np.allclose(
            resampler.out_coords_y,
            [1190031.36, 590031.36, -9968.64, -609968.64]))

        # Data reduction disabled
        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius, reduce_data=False)
        resampler.get_bil_info()
        t__ = resampler.bilinear_t
        s__ = resampler.bilinear_s
        mask_slices = resampler.mask_slices
        _check_ts(t__, s__, [10, 12, 13, 14, 15])

        # Target area and source data do not overlap
        resampler = XArrayBilinearResampler(self.source_def, self.target_def_outside,
                                            self.radius, reduce_data=False)
        resampler.get_bil_info()
        self.assertEqual(resampler.bilinear_t.shape, (self.target_def_outside.size,))
        self.assertEqual(resampler.bilinear_s.shape, (self.target_def_outside.size,))
        self.assertEqual(resampler.slices_x.shape, (self.target_def_outside.size, 4))
        self.assertEqual(resampler.slices_y.shape, (self.target_def_outside.size, 4))
        self.assertEqual(resampler.out_coords_x.shape, (self.target_def_outside.shape[1],))
        self.assertEqual(resampler.out_coords_y.shape, (self.target_def_outside.shape[0],))
        self.assertEqual(resampler.mask_slices.shape, (self.target_def_outside.size, 4))

    def test_get_sample_from_bil_info(self):
        """Test bilinear interpolation as a whole."""
        import dask.array as da
        from xarray import DataArray

        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        resampler.get_bil_info()

        # Sample from data1
        res = resampler.get_sample_from_bil_info(self.data1)
        res = res.compute()
        # Check couple of values
        self.assertAlmostEqual(res.values[1, 1], 1.)
        self.assertTrue(np.isnan(res.values[0, 3]))
        # Check that the values haven't gone down or up a lot
        self.assertAlmostEqual(np.nanmin(res.values), 1.)
        self.assertAlmostEqual(np.nanmax(res.values), 1.)
        # Check that dimensions are the same
        self.assertEqual(res.dims, self.data1.dims)

        # Sample from data1, custom fill value
        res = resampler.get_sample_from_bil_info(self.data1, fill_value=-1.0)
        res = res.compute()
        self.assertAlmostEqual(np.nanmin(res.values), -1.)

        # Sample from integer data
        res = resampler.get_sample_from_bil_info(self.data1.astype(np.uint8),
                                                 fill_value=None)
        res = res.compute()
        # Five values should be filled with zeros, which is the
        # default fill_value for integer data
        self.assertAlmostEqual(np.sum(res == 0), 6)

        # Output coordinates should have been set
        self.assertTrue(isinstance(resampler._out_coords, dict))
        self.assertTrue(np.all(resampler._out_coords['x'] == resampler.out_coords_x))
        self.assertTrue(np.all(resampler._out_coords['y'] == resampler.out_coords_y))

        # 3D data
        data = da.moveaxis(da.dstack((self.data1, self.data1)), -1, 0)
        data = DataArray(data, dims=('bands', 'y', 'x'))
        res = resampler.get_sample_from_bil_info(data)
        assert res.shape == (2,) + self.target_def.shape
        assert res.dims == data.dims

    def test_add_missing_coordinates(self):
        """Test coordinate updating."""
        import dask.array as da
        from xarray import DataArray

        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        bands = ['R', 'G', 'B']
        data = DataArray(da.ones((3, 10, 10)), dims=('bands', 'y', 'x'),
                         coords={'bands': bands,
                                 'y': np.arange(10), 'x': np.arange(10)})
        resampler._add_missing_coordinates(data)
        # X and Y coordinates should not change
        self.assertIsNone(resampler.out_coords_x)
        self.assertIsNone(resampler.out_coords_y)
        self.assertIsNone(resampler._out_coords['x'])
        self.assertIsNone(resampler._out_coords['y'])
        self.assertTrue('bands' in resampler._out_coords)
        self.assertTrue(np.all(resampler._out_coords['bands'] == bands))

        # Available coordinates from self.out_coords_x and self.out_coords_y
        # should be set to self._out_coords
        resampler.out_coords_x = [1]
        resampler.out_coords_y = [2]
        resampler._add_missing_coordinates(data)
        self.assertEqual(resampler._out_coords['x'], resampler.out_coords_x)
        self.assertEqual(resampler._out_coords['y'], resampler.out_coords_y)

    def test_slice_data(self):
        """Test slicing the data."""
        import dask.array as da
        from xarray import DataArray

        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        resampler.get_bil_info()

        # Too many dimensions
        data = DataArray(da.ones((1, 3) + self.source_def.shape))
        with self.assertRaises(ValueError):
            _ = resampler._slice_data(data, np.nan)

        # 2D data
        data = DataArray(da.ones(self.source_def.shape))
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertEqual(p_1.shape, resampler.bilinear_s.shape)
        self.assertTrue(p_1.shape == p_2.shape == p_3.shape == p_4.shape)
        self.assertTrue(np.all(p_1 == 1.0) and np.all(p_2 == 1.0) and
                        np.all(p_3 == 1.0) and np.all(p_4 == 1.0))

        # 2D data with masking
        resampler.mask_slices[:, :] = True
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertTrue(np.all(np.isnan(p_1)) and np.all(np.isnan(p_2)) and
                        np.all(np.isnan(p_3)) and np.all(np.isnan(p_4)))
        # 3D data
        data = DataArray(da.ones((3,) + self.source_def.shape))
        resampler.mask_slices[:, :] = False
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertEqual(p_1.shape, (3,) + resampler.bilinear_s.shape)
        self.assertTrue(p_1.shape == p_2.shape == p_3.shape == p_4.shape)

        # 3D data with masking
        resampler.mask_slices[:, :] = True
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertTrue(np.all(np.isnan(p_1)) and np.all(np.isnan(p_2)) and
                        np.all(np.isnan(p_3)) and np.all(np.isnan(p_4)))

    def test_slice_data_1d(self):
        """Test slicing 1D data."""
        import dask.array as da
        from xarray import DataArray

        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def_1d, self.target_def,
                                            self.radius)
        resampler.get_bil_info()

        # 1D data
        data = DataArray(da.ones(self.source_def_1d.shape))
        p_1, p_2, p_3, p_4 = resampler._slice_data(data, np.nan)
        self.assertEqual(p_1.shape, resampler.bilinear_s.shape)
        self.assertTrue(p_1.shape == p_2.shape == p_3.shape == p_4.shape)
        self.assertTrue(np.all(p_1 == 1.0) and np.all(p_2 == 1.0) and
                        np.all(p_3 == 1.0) and np.all(p_4 == 1.0))

    def test_get_sample_from_bil_info_1d(self):
        """Test resampling using resampling indices for 1D data."""
        import dask.array as da
        from xarray import DataArray

        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def_1d, self.target_def,
                                            50e5)
        resampler.get_bil_info()

        # Sample from 1D data
        data = DataArray(da.ones(self.source_def_1d.shape), dims=('y'))
        res = resampler.get_sample_from_bil_info(data)  # noqa
        assert 'x' in res.dims
        assert 'y' in res.dims

        # Four pixels are outside of the data
        self.assertEqual(np.isnan(res).sum().compute(), 4)

    @mock.patch('pyresample.bilinear.xarr.np.meshgrid')
    def test_get_slices(self, meshgrid):
        """Test slice array creation."""
        from pyresample.bilinear import XArrayBilinearResampler

        meshgrid.return_value = (self.cols, self.lines)

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        resampler._valid_input_index = self._valid_input_index
        resampler._index_array = self._index_array

        resampler._get_slices()

        self.assertIsNotNone(resampler.slices_x)
        self.assertIsNotNone(resampler.slices_y)
        self.assertTrue(resampler.slices_x.shape == (self.target_def.size, 32))
        self.assertTrue(resampler.slices_y.shape == (self.target_def.size, 32))
        self.assertEqual(np.sum(resampler.slices_x), 12471)
        self.assertEqual(np.sum(resampler.slices_y), 2223)

        self.assertFalse(np.any(resampler.mask_slices))

        # Ensure that source geo def is used in masking
        # Setting target_geo_def to 0-size shouldn't cause any masked values
        resampler._target_geo_def = np.array([])
        resampler._get_slices()
        self.assertFalse(np.any(resampler.mask_slices))
        # Setting source area def to 0-size should mask all values
        resampler._source_geo_def = np.array([[]])
        resampler._get_slices()
        self.assertTrue(np.all(resampler.mask_slices))

    @mock.patch('pyresample.bilinear._base.KDTree')
    def test_create_resample_kdtree(self, KDTree):
        """Test that KDTree creation is called."""
        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)

        vii, kdtree = resampler._create_resample_kdtree()
        self.assertEqual(np.sum(vii), 2700)
        self.assertEqual(vii.size, self.source_def.size)
        KDTree.assert_called_once()

    @mock.patch('pyresample.bilinear._base.BilinearBase._reduce_index_array')
    @mock.patch('pyresample.bilinear._base._query_no_distance')
    def test_get_index_array(self, qnd, ria):
        """Test that query_no_distance is called in __get_index_array()."""
        from pyresample.bilinear import XArrayBilinearResampler

        qnd.return_value = 'foo'
        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        resampler._target_lons = 1
        resampler._target_lats = 2
        resampler._resample_kdtree = 3
        resampler._get_index_array()
        qnd.assert_called_with(1, 2, True, 3, resampler._neighbours,
                               resampler._epsilon,
                               resampler._radius_of_influence)
        ria.assert_called_with(qnd.return_value)

    def test_get_input_xy(self):
        """Test computation of input X and Y coordinates in target proj."""
        from pyresample.bilinear.xarr import _get_input_xy

        proj = Proj(self.target_def.proj_str)
        in_x, in_y = _get_input_xy(self.source_def, proj,
                                   self._valid_input_index,
                                   self._index_array)

        self.assertTrue(in_x.shape, (self.target_def.size, 32))
        self.assertTrue(in_y.shape, (self.target_def.size, 32))
        self.assertTrue(in_x.all())
        self.assertTrue(in_y.all())

    def test_get_four_closest_corners(self):
        """Test finding surrounding bounding corners."""
        import dask.array as da

        from pyresample import CHUNK_SIZE
        from pyresample.bilinear._base import _get_four_closest_corners
        from pyresample.bilinear.xarr import _get_input_xy

        proj = Proj(self.target_def.proj_str)
        out_x, out_y = self.target_def.get_proj_coords(chunks=CHUNK_SIZE)
        out_x = da.ravel(out_x)
        out_y = da.ravel(out_y)
        in_x, in_y = _get_input_xy(self.source_def, proj,
                                   self._valid_input_index,
                                   self._index_array)
        (pt_1, pt_2, pt_3, pt_4), ia_ = _get_four_closest_corners(
            in_x, in_y, out_x, out_y,
            self._neighbours,
            self._index_array)

        self.assertTrue(pt_1.shape == pt_2.shape ==
                        pt_3.shape == pt_4.shape ==
                        (self.target_def.size, 2))
        self.assertTrue(ia_.shape == (self.target_def.size, 4))

        # Check which of the locations has four valid X/Y pairs by
        # finding where there are non-NaN values
        res = da.sum(pt_1 + pt_2 + pt_3 + pt_4, axis=1).compute()
        self.assertEqual(np.sum(~np.isnan(res)), 10)

    def test_get_corner(self):
        """Test finding the closest corners."""
        import dask.array as da

        from pyresample import CHUNK_SIZE
        from pyresample.bilinear._base import _get_corner, _get_input_xy

        proj = Proj(self.target_def.proj_str)
        in_x, in_y = _get_input_xy(self.source_def, proj,
                                   self._valid_input_index,
                                   self._index_array)
        out_x, out_y = self.target_def.get_proj_coords(chunks=CHUNK_SIZE)
        out_x = da.ravel(out_x)
        out_y = da.ravel(out_y)

        # Some copy&paste from the code to get the input
        out_x_tile = np.reshape(np.tile(out_x, self._neighbours),
                                (self._neighbours, out_x.size)).T
        out_y_tile = np.reshape(np.tile(out_y, self._neighbours),
                                (self._neighbours, out_y.size)).T
        x_diff = out_x_tile - in_x
        y_diff = out_y_tile - in_y
        stride = np.arange(x_diff.shape[0])

        # Use lower left source pixels for testing
        valid = (x_diff > 0) & (y_diff > 0)
        x_3, y_3, idx_3 = _get_corner(stride, valid, in_x, in_y,
                                      self._index_array)

        self.assertTrue(x_3.shape == y_3.shape == idx_3.shape ==
                        (self.target_def.size, ))
        # Four locations have no data to the lower left of them (the
        # bottom row of the area
        self.assertEqual(np.sum(np.isnan(x_3.compute())), 4)

    @mock.patch('pyresample.bilinear._base._get_fractional_distances_parallellogram')
    @mock.patch('pyresample.bilinear._base._get_fractional_distances_uprights_parallel')
    @mock.patch('pyresample.bilinear._base._get_fractional_distances_irregular')
    def test_get_fractional_distances(self, irregular, uprights, parallellogram):
        """Test that the three separate functions are called."""
        import dask.array as da

        from pyresample.bilinear._base import _get_fractional_distances

        # All valid values
        t_irr = da.array([0.1, 0.2, 0.3])
        s_irr = da.array([0.1, 0.2, 0.3])
        irregular.return_value = (t_irr, s_irr)
        t__, s__ = _get_fractional_distances((1, 2, 3, 4), 5, 6)
        irregular.assert_called_once()
        uprights.assert_not_called()
        parallellogram.assert_not_called()
        self.assertTrue(np.allclose(t__.compute(), t_irr))
        self.assertTrue(np.allclose(s__.compute(), s_irr))

        # NaN in the first step, good value for that location from the
        # second step
        t_irr = da.array([0.1, 0.2, np.nan])
        s_irr = da.array([0.1, 0.2, np.nan])
        irregular.return_value = (t_irr, s_irr)
        t_upr = da.array([3, 3, 0.3])
        s_upr = da.array([3, 3, 0.3])
        uprights.return_value = (t_upr, s_upr)
        t__, s__ = _get_fractional_distances((1, 2, 3, 4), 5, 6)
        self.assertEqual(irregular.call_count, 2)
        uprights.assert_called_once()
        parallellogram.assert_not_called()
        # Only the last value of the first step should have been replaced
        t_res = da.array([0.1, 0.2, 0.3])
        s_res = da.array([0.1, 0.2, 0.3])
        self.assertTrue(np.allclose(t__.compute(), t_res))
        self.assertTrue(np.allclose(s__.compute(), s_res))

        # Two NaNs in the first step, one of which are found by the
        # second, and the last bad value is replaced by the third step
        t_irr = da.array([0.1, np.nan, np.nan])
        s_irr = da.array([0.1, np.nan, np.nan])
        irregular.return_value = (t_irr, s_irr)
        t_upr = da.array([3, np.nan, 0.3])
        s_upr = da.array([3, np.nan, 0.3])
        uprights.return_value = (t_upr, s_upr)
        t_par = da.array([4, 0.2, 0.3])
        s_par = da.array([4, 0.2, 0.3])
        parallellogram.return_value = (t_par, s_par)
        t__, s__ = _get_fractional_distances((1, 2, 3, 4), 5, 6)
        self.assertEqual(irregular.call_count, 3)
        self.assertEqual(uprights.call_count, 2)
        parallellogram.assert_called_once()
        # Only the last two values should have been replaced
        t_res = da.array([0.1, 0.2, 0.3])
        s_res = da.array([0.1, 0.2, 0.3])
        self.assertTrue(np.allclose(t__.compute(), t_res))
        self.assertTrue(np.allclose(s__.compute(), s_res))

        # Too large and small values should be set to NaN
        t_irr = da.array([1.00001, -0.00001, 1e6])
        s_irr = da.array([1.00001, -0.00001, -1e6])
        irregular.return_value = (t_irr, s_irr)
        # Second step also returns invalid values
        t_upr = da.array([1.00001, 0.2, np.nan])
        s_upr = da.array([-0.00001, 0.2, np.nan])
        uprights.return_value = (t_upr, s_upr)
        # Third step has one new valid value, the last will stay invalid
        t_par = da.array([0.1, 0.2, 4.0])
        s_par = da.array([0.1, 0.2, 4.0])
        parallellogram.return_value = (t_par, s_par)
        t__, s__ = _get_fractional_distances((1, 2, 3, 4), 5, 6)

        t_res = da.array([0.1, 0.2, np.nan])
        s_res = da.array([0.1, 0.2, np.nan])
        self.assertTrue(np.allclose(t__.compute(), t_res, equal_nan=True))
        self.assertTrue(np.allclose(s__.compute(), s_res, equal_nan=True))

    def test_get_fractional_distances_division_by_zero(self):
        """Test that the correct result is found even when there's a division by zero when solving t and s."""
        from pyresample.bilinear._base import _get_fractional_distances
        corner_points = [np.array([[-64.9936752319336, -5.140199184417725]]),
                         np.array([[-64.98487091064453, -5.142156600952148]]),
                         np.array([[-64.98683166503906, -5.151054859161377]]),
                         np.array([[-64.97802734375, -5.153012275695801]])]
        out_x = np.array([-64.985])
        out_y = np.array([-5.145])

        t__, s__ = _get_fractional_distances(corner_points, out_x, out_y)
        np.testing.assert_allclose(t__, np.array([0.30769689]))
        np.testing.assert_allclose(s__, np.array([0.74616628]))

    def test_get_fractional_distances_irregular(self):
        """Test calculations for irregular corner locations."""
        from pyresample.bilinear._base import _get_fractional_distances_irregular

        res = _get_fractional_distances_irregular(self.pts_irregular, 0., 0.)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)
        res = _get_fractional_distances_irregular(
            self.pts_vert_parallel, 0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertTrue(np.isnan(res[1]))

    def test_get_fractional_distances_uprights_parallel(self):
        """Test calculation when uprights are parallel."""
        from pyresample.bilinear._base import _get_fractional_distances_uprights_parallel

        res = _get_fractional_distances_uprights_parallel(self.pts_vert_parallel, 0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_get_fractional_distances_parallellogram(self):
        """Test calculation when the corners form a parallellogram."""
        from pyresample.bilinear._base import _get_fractional_distances_parallellogram

        res = _get_fractional_distances_parallellogram(self.pts_both_parallel[:3], 0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_calc_abc(self):
        """Test calculation of quadratic coefficients."""
        from pyresample.bilinear._base import _calc_abc

        # No np.nan inputs
        res = _calc_abc(self.pts_irregular, 0.0, 0.0)
        self.assertFalse(np.isnan(res[0]))
        self.assertFalse(np.isnan(res[1]))
        self.assertFalse(np.isnan(res[2]))
        # np.nan input -> np.nan output
        pt_1, pt_2, pt_3, pt_4 = self.pts_irregular
        corner_points = (np.array([[np.nan, np.nan]]), pt_2, pt_3, pt_4)
        res = _calc_abc(corner_points, 0.0, 0.0)
        self.assertTrue(np.isnan(res[0]))
        self.assertTrue(np.isnan(res[1]))
        self.assertTrue(np.isnan(res[2]))

    def test_solve_quadratic(self):
        """Test solving quadratic equation."""
        import dask.array as da

        from pyresample.bilinear._base import _calc_abc, _solve_quadratic

        res = _solve_quadratic(1, 0, 0)
        self.assertEqual(res, 0.0)
        res = _solve_quadratic(1, 2, 1)
        self.assertTrue(np.isnan(res))
        res = _solve_quadratic(1, 2, 1, min_val=-2.)
        self.assertEqual(res, -1.0)
        # Test that small adjustments work
        pt_1, pt_2, pt_3, pt_4 = self.pts_vert_parallel
        pt_1 = self.pts_vert_parallel[0].compute()
        pt_1[0][0] += 1e-7
        pt_1 = da.from_array(pt_1)
        corner_points = (pt_1, pt_2, pt_3, pt_4)
        res = _calc_abc(corner_points, 0.0, 0.0)
        res = _solve_quadratic(res[0], res[1], res[2]).compute()
        self.assertAlmostEqual(res[0], 0.5, 5)
        corner_points = (pt_1, pt_3, pt_2, pt_4)
        res = _calc_abc(corner_points, 0.0, 0.0)
        res = _solve_quadratic(res[0], res[1], res[2]).compute()
        self.assertAlmostEqual(res[0], 0.5, 5)

    def test_query_no_distance(self):
        """Test KDTree querying."""
        from pyresample.bilinear._base import _query_no_distance

        kdtree = mock.MagicMock()
        kdtree.query.return_value = (1, 2)
        lons, lats = self.target_def.get_lonlats()
        voi = np.ravel(
            (lons >= -180) & (lons <= 180) & (lats <= 90) & (lats >= -90))
        res = _query_no_distance(lons, lats, voi, kdtree, self._neighbours,
                                 0., self.radius)
        # Only the second value from the query is returned
        self.assertEqual(res, 2)
        kdtree.query.assert_called_once()

    def test_get_valid_input_index(self):
        """Test finding valid indices for reduced input data."""
        from pyresample.bilinear._base import _get_valid_input_index

        # Do not reduce data
        vii, lons, lats = _get_valid_input_index(self.source_def,
                                                 self.target_def,
                                                 False, self.radius)
        self.assertEqual(vii.shape, (self.source_def.size, ))
        self.assertTrue(vii.dtype == bool)
        # No data has been reduced, whole input is used
        self.assertTrue(vii.all())

        # Reduce data
        vii, lons, lats = _get_valid_input_index(self.source_def,
                                                 self.target_def,
                                                 True, self.radius)
        # 2700 valid input points
        self.assertEqual(vii.sum(), 2700)

    def test_create_empty_bil_info(self):
        """Test creation of empty bilinear info."""
        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)

        resampler._create_empty_bil_info()
        self.assertEqual(resampler.bilinear_t.shape, (self.target_def.size,))
        self.assertEqual(resampler.bilinear_s.shape, (self.target_def.size,))
        self.assertEqual(resampler._index_array.shape, (self.target_def.size, 4))
        self.assertTrue(resampler._index_array.dtype == np.int32)
        self.assertEqual(resampler._valid_input_index.shape, (self.source_def.size,))
        self.assertTrue(resampler._valid_input_index.dtype == bool)

    def test_lonlat2xyz(self):
        """Test conversion from geographic to cartesian 3D coordinates."""
        from pyresample import CHUNK_SIZE
        from pyresample.future.resamplers._transform_utils import lonlat2xyz

        lons, lats = self.target_def.get_lonlats(chunks=CHUNK_SIZE)
        res = lonlat2xyz(lons, lats)
        self.assertEqual(res.shape, (self.target_def.size, 3))
        vals = [3188578.91069278, -612099.36103276, 5481596.63569999]
        self.assertTrue(np.allclose(res.compute()[0, :], vals))

    def test_class_resample_method(self):
        """Test the 'resampler.resample()' method."""
        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def,
                                            self.target_def,
                                            50e5,
                                            neighbours=32,
                                            epsilon=0)

        # Single array, no fill value
        res = resampler.resample(self.data1)
        self.assertEqual(res.shape, self.target_def.shape)
        # There are 12 pixels with value 1, all others are NaN
        res = res.compute()
        self.assertEqual(np.nansum(res), 12)
        self.assertEqual(np.isnan(res).sum(), 4)

        # Single array with fill value
        res = resampler.resample(self.data1, fill_value=0)
        res = res.compute()
        self.assertEqual(np.sum(res), 12)
        self.assertEqual((res == 0).sum(), 4)

    def test_save_and_load_bil_info(self):
        """Test saving and loading the resampling info."""
        import os
        import shutil
        from tempfile import mkdtemp

        from pyresample.bilinear import CACHE_INDICES, XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                            self.radius)
        resampler.get_bil_info()

        try:
            tempdir = mkdtemp()
            filename = os.path.join(tempdir, "test.zarr")

            resampler.save_resampling_info(filename)

            assert os.path.exists(filename)

            new_resampler = XArrayBilinearResampler(self.source_def, self.target_def,
                                                    self.radius)
            new_resampler.load_resampling_info(filename)

            for attr in CACHE_INDICES:
                orig = getattr(resampler, attr)
                reloaded = getattr(new_resampler, attr).compute()
                np.testing.assert_array_equal(orig, reloaded)
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

    def test_get_sample_from_cached_bil_info(self):
        """Test getting data using pre-calculated resampling info."""
        import os
        import shutil
        from tempfile import mkdtemp

        from pyresample.bilinear import XArrayBilinearResampler

        resampler = XArrayBilinearResampler(self.source_def, self.target_def_partial,
                                            self.radius)
        resampler.get_bil_info()

        try:
            tempdir = mkdtemp()
            filename = os.path.join(tempdir, "test.zarr")

            resampler.save_resampling_info(filename)

            assert os.path.exists(filename)

            new_resampler = XArrayBilinearResampler(self.source_def, self.target_def_partial,
                                                    self.radius)
            new_resampler.load_resampling_info(filename)
            _ = new_resampler.get_sample_from_bil_info(self.data1)
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)


def test_check_fill_value():
    """Test that fill_value replacement/adjustment works."""
    from pyresample.bilinear._base import _check_fill_value

    # None + integer dtype -> 0
    assert _check_fill_value(None, np.uint8) == 0
    # None + float dtype -> np.nan
    assert np.isnan(_check_fill_value(None, np.double))

    # integer fill value + integer dtype -> no change
    assert _check_fill_value(3, np.uint8) == 3
    # np.nan + integer dtype -> 0
    assert _check_fill_value(np.nan, np.uint8) == 0
    # float fill value + integer dtype -> int(fill_value)
    assert _check_fill_value(3.3, np.uint16) == 3

    # float fill value + float dtype -> no change
    assert _check_fill_value(3.3, np.float32)


def test_target_has_invalid_coordinates():
    """Test bilinear resampling to area that has invalid coordinates.

    The area used here is in geos projection that has space pixels in the corners.
    """
    import dask.array as da
    import xarray as xr

    from pyresample.bilinear import XArrayBilinearResampler

    # NumpyBilinearResampler
    from pyresample.geometry import AreaDefinition, GridDefinition

    geos_def = AreaDefinition('geos',
                              'GEO area with space in corners',
                              'geos',
                              {'proj': 'geos',
                               'lon_0': '0.0',
                               'a': '6378169.0',
                               'b': '6356583.8',
                               'h': '35785831.0'},
                              640, 640,
                              [-5432229.931711678,
                               -5429229.528545862,
                               5429229.528545862,
                               5432229.931711678])
    lats = np.linspace(-89, 89, 179)
    lons = np.linspace(-179, 179, 359)
    lats = np.repeat(lats[:, None], 359, axis=1)
    lons = np.repeat(lons[None, :], 179, axis=0)
    grid_def = GridDefinition(lons=lons, lats=lats)

    data_xr = xr.DataArray(da.random.uniform(0, 1, lons.shape), dims=["y", "x"])

    resampler = XArrayBilinearResampler(grid_def,
                                        geos_def,
                                        500e3,
                                        reduce_data=False)
    res = resampler.resample(data_xr)
    res = res.compute()
    assert not np.all(np.isnan(res))
    assert np.any(np.isnan(res))
