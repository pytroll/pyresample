#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Pyresample developers
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
"""Test the bucket resampler."""

import unittest
from unittest.mock import MagicMock, patch

import dask
import dask.array as da
import numpy as np
import xarray as xr

from pyresample import bucket, create_area_def
from pyresample.geometry import AreaDefinition
from pyresample.test.utils import CustomScheduler


class Test(unittest.TestCase):
    """Test bucket resampler."""

    adef = AreaDefinition('eurol', 'description', '',
                          {'ellps': 'WGS84',
                           'lat_0': '90.0',
                           'lat_ts': '60.0',
                           'lon_0': '0.0',
                           'proj': 'stere'}, 2560, 2048,
                          (-3780000.0, -7644000.0, 3900000.0, -1500000.0))
    chunks = 2
    lons = da.from_array(np.array([[25., 25.], [25., 25.]]),
                         chunks=chunks)
    lats = da.from_array(np.array([[60., 60.00001], [60.2, 60.3]]),
                         chunks=chunks)

    def setUp(self):
        self.resampler = bucket.BucketResampler(self.adef, self.lons, self.lats)

    @patch('pyresample.bucket.Proj')
    @patch('pyresample.bucket.BucketResampler._get_indices')
    def test_init(self, get_indices, prj):
        resampler = bucket.BucketResampler(self.adef, self.lons, self.lats)
        get_indices.assert_called_once()
        prj.assert_called_once_with(self.adef.proj_dict)
        self.assertTrue(hasattr(resampler, 'target_area'))
        self.assertTrue(hasattr(resampler, 'source_lons'))
        self.assertTrue(hasattr(resampler, 'source_lats'))
        self.assertTrue(hasattr(resampler, 'x_idxs'))
        self.assertTrue(hasattr(resampler, 'y_idxs'))
        self.assertTrue(hasattr(resampler, 'idxs'))
        self.assertTrue(hasattr(resampler, 'get_sum'))
        self.assertTrue(hasattr(resampler, 'get_count'))
        self.assertTrue(hasattr(resampler, 'get_min'))
        self.assertTrue(hasattr(resampler, 'get_max'))
        self.assertTrue(hasattr(resampler, 'get_average'))
        self.assertTrue(hasattr(resampler, 'get_fractions'))
        self.assertIsNone(resampler.counts)

    def test_round_to_resolution(self):
        """Test rounding to given resolution."""
        # Scalar, integer resolution
        self.assertEqual(bucket.round_to_resolution(5.5, 2.), 6)
        # Scalar, non-integer resolution
        self.assertEqual(bucket.round_to_resolution(5.5, 1.7), 5.1)
        # List
        self.assertTrue(np.all(bucket.round_to_resolution([4.2, 5.6], 2) ==
                               np.array([4., 6.])))
        # Numpy array
        self.assertTrue(np.all(bucket.round_to_resolution(np.array([4.2, 5.6]), 2) ==
                               np.array([4., 6.])))
        # Dask array
        self.assertTrue(
            np.all(bucket.round_to_resolution(da.array([4.2, 5.6]), 2) ==
                   np.array([4., 6.])))

    def test_get_proj_coordinates(self):
        """Test calculation of projection coordinates."""
        prj = MagicMock()
        prj.return_value = ([3.1, 3.1, 3.1], [4.8, 4.8, 4.8])
        lons = [1., 1., 1.]
        lats = [2., 2., 2.]
        self.resampler.prj = prj
        result = self.resampler._get_proj_coordinates(lons, lats)
        prj.assert_called_once_with(lons, lats)
        self.assertTrue(isinstance(result, np.ndarray))
        np.testing.assert_equal(result, np.array([[3.1, 3.1, 3.1],
                                                  [4.8, 4.8, 4.8]]))

    def test_get_bucket_indices(self):
        """Test calculation of array indices."""
        # Ensure nothing is calculated
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            self.resampler._get_indices()
        x_idxs, y_idxs = da.compute(self.resampler.x_idxs,
                                    self.resampler.y_idxs)
        np.testing.assert_equal(x_idxs, np.array([1710, 1710, 1707, 1705]))
        np.testing.assert_equal(y_idxs, np.array([465, 465, 459, 455]))

        # Additional small test case
        adef = create_area_def(
            area_id='test',
            projection={'proj': 'latlong'},
            width=2, height=2,
            center=(0, 0),
            resolution=10)
        lons = da.from_array(
            np.array([-10.0, -9.9, -0.1, 0, 0.1, 9.9, 10.0, -10.1, 0]),
            chunks=2)
        lats = da.from_array(
            np.array([-10.0, -9.9, -0.1, 0, 0.1, 9.9, 10.0, 0, 10.1]),
            chunks=2)
        resampler = bucket.BucketResampler(source_lats=lats,
                                           source_lons=lons,
                                           target_area=adef)
        resampler._get_indices()
        np.testing.assert_equal(resampler.x_idxs, np.array([-1, 0, 0, 1, 1, 1, -1, -1, -1]))
        np.testing.assert_equal(resampler.y_idxs, np.array([-1, 1, 1, 1, 0, 0, -1, -1, -1]))

    def _get_sum_result(self, data, **kwargs):
        """Compute the bucket average with kwargs and check that no dask computation is performed."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_sum(data, **kwargs)
        return result.compute()

    def test_get_sum_valid_data(self):
        """Test drop-in-a-bucket sum for valid data input."""
        data = da.from_array(np.array([[2., 3.], [7., 16.]]),
                             chunks=self.chunks)

        result = self._get_sum_result(data)

        # first two values are in same bin
        self.assertEqual(np.count_nonzero(result == 5), 1)
        # others are in separate bins
        self.assertEqual(np.count_nonzero(result == 7), 1)
        self.assertEqual(np.count_nonzero(result == 16), 1)

        self.assertEqual(result.shape, self.adef.shape)

        # Test that also xarray.DataArrays work (same output)
        data = xr.DataArray(data)
        np.testing.assert_array_equal(result, self._get_sum_result(data))

    def test_get_sum_nan_data_skipna_false(self):
        """Test drop-in-a-bucket sum for data input with nan and skipna False."""
        data = da.from_array(np.array([[2., np.nan], [5., np.nan]]),
                             chunks=self.chunks)

        result = self._get_sum_result(data, skipna=False)
        # 2 + nan is nan, all-nan bin is nan
        self.assertEqual(np.count_nonzero(np.isnan(result)), 2)
        # rest is 0
        self.assertEqual(np.nanmin(result), 0)

    def test_get_sum_nan_data_skipna_true(self):
        """Test drop-in-a-bucket sum for data input with nan and skipna True."""
        data = da.from_array(np.array([[2., np.nan], [5., np.nan]]),
                             chunks=self.chunks)

        result = self._get_sum_result(data, skipna=True)
        # 2 + nan is 2
        self.assertEqual(np.count_nonzero(result == 2.), 1)
        # all-nan and rest is 0
        self.assertEqual(np.count_nonzero(np.isnan(result)), 0)
        self.assertEqual(np.nanmin(result), 0)

    def test_get_count(self):
        """Test drop-in-a-bucket sum."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_count()
        result = result.compute()
        self.assertTrue(np.max(result) == 2)
        self.assertEqual(np.sum(result == 1), 2)
        self.assertEqual(np.sum(result == 2), 1)
        self.assertTrue(self.resampler.counts is not None)

    def _get_min_result(self, data, **kwargs):
        """Compute the bucket average with kwargs and check that no dask computation is performed."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            result = self.resampler.get_min(data, **kwargs)
        return result.compute()

    def test_get_min(self):
        """Test min bucket resampling."""
        data = da.from_array(np.array([[2, 11], [5, np.nan]]),
                             chunks=self.chunks)
        result = self._get_min_result(data)
        # test multiple entries average
        self.assertEqual(np.count_nonzero(result == 2), 1)
        # test single entry average
        self.assertEqual(np.count_nonzero(result == 5), 1)
        # test that minimum of bucket with only nan is nan, and empty buckets are nan
        self.assertEqual(np.count_nonzero(~np.isnan(result)), 2)

    def _get_max_result(self, data, **kwargs):
        """Compute the bucket average with kwargs and check that no dask computation is performed."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            result = self.resampler.get_max(data, **kwargs)
        return result.compute()

    def test_get_max(self):
        """Test max bucket resampling."""
        data = da.from_array(np.array([[2, 11], [5, np.nan]]),
                             chunks=self.chunks)
        result = self._get_max_result(data)
        # test multiple entries average
        self.assertEqual(np.count_nonzero(result == 11), 1)
        # test single entry average
        self.assertEqual(np.count_nonzero(result == 5), 1)
        # test that minimum of bucket with only nan is nan, and empty buckets are nan
        self.assertEqual(np.count_nonzero(~np.isnan(result)), 2)

    def _get_average_result(self, data, **kwargs):
        """Compute the bucket average with kwargs and check that no dask computation is performed."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_average(data, **kwargs)
        return result.compute()

    def test_get_average_basic(self):
        """Test averaging bucket resampling."""
        data = da.from_array(np.array([[2, 11], [5, np.nan]]),
                             chunks=self.chunks)
        result = self._get_average_result(data)
        # test multiple entries average
        self.assertEqual(np.count_nonzero(result == 6.5), 1)
        # test single entry average
        self.assertEqual(np.count_nonzero(result == 5), 1)
        # test that average of bucket with only nan is nan, and empty buckets are nan
        self.assertEqual(np.count_nonzero(~np.isnan(result)), 2)

    def test_get_average_with_fillvalue_for_output(self):
        """Test averaging bucket resampling with defined fill_value for output."""
        data = da.from_array(np.array([[2, 11], [5, np.nan]]),
                             chunks=self.chunks)
        # test fill_value other than np.nan
        result = self._get_average_result(data, fill_value=-1)
        # check that all empty buckets are fill_value
        self.assertEqual(np.count_nonzero(result != -1), 2)

    def test_get_average_skipna_true(self):
        """Test averaging bucket resampling with skipna True."""
        # test skipna
        data = da.from_array(np.array([[2, np.nan], [np.nan, np.nan]]),
                             chunks=self.chunks)
        result = self._get_average_result(data, skipna=True)
        # test that average of 2 and np.nan is 2 for skipna=True
        self.assertEqual(np.count_nonzero(result == 2), 1)

    def test_get_average_skipna_false(self):
        """Test averaging bucket resampling with skipna False."""
        data = da.from_array(np.array([[2, np.nan], [np.nan, np.nan]]),
                             chunks=self.chunks)
        result = self._get_average_result(data, skipna=False)
        # test that average of 2 and np.nan is nan for skipna=False
        self.assertTrue(np.all(np.isnan(result)))

    def test_get_average_only_nan_input(self):
        """Test averaging bucket resampling with only NaN as input."""
        data = da.from_array(np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                             chunks=self.chunks)
        result = self._get_average_result(data, skipna=True)
        # test that average of np.nan and np.nan is np.nan for both skipna
        self.assertTrue(np.all(np.isnan(result)))
        np.testing.assert_array_equal(result, self._get_average_result(data, skipna=False))

    def test_get_average_with_fill_value_in_input(self):
        """Test averaging bucket resampling with fill_value in input and skipna True."""
        # test that fill_value in input is recognised as missing value
        data = da.from_array(np.array([[2, -1], [-1, np.nan]]),
                             chunks=self.chunks)
        result = self._get_average_result(data, fill_value=-1, skipna=True)
        # test that average of 2 and -1 (missing value) is 2
        self.assertEqual(np.count_nonzero(result == 2), 1)
        # test than all other buckets are -1
        self.assertEqual(np.count_nonzero(result != -1), 1)

    def test_resample_bucket_fractions(self):
        """Test fraction calculations for categorical data."""
        data = da.from_array(np.array([[2, 4], [2, 2]]),
                             chunks=self.chunks)
        categories = [1, 2, 3, 4]
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_fractions(data, categories=categories)
        self.assertEqual(set(categories), set(result.keys()))
        res = result[1].compute()
        self.assertTrue(np.nanmax(res) == 0.)
        res = result[2].compute()
        self.assertTrue(np.nanmax(res) == 1.)
        self.assertTrue(np.nanmin(res) == 0.5)
        res = result[3].compute()
        self.assertTrue(np.nanmax(res) == 0.)
        res = result[4].compute()
        self.assertTrue(np.nanmax(res) == 0.5)
        self.assertTrue(np.nanmin(res) == 0.)
        # There should be NaN values
        self.assertTrue(np.any(np.isnan(res)))

        # Use a fill value
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_fractions(data, categories=categories,
                                                  fill_value=-1)

        # There should not be any NaN values
        for i in categories:
            res = result[i].compute()
            self.assertFalse(np.any(np.isnan(res)))
            self.assertTrue(np.min(res) == -1)

        # No categories given, need to compute the data once to get
        # the categories
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            _ = self.resampler.get_fractions(data, categories=None)
