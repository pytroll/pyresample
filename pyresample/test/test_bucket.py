import unittest
import numpy as np
import dask.array as da
import dask
import xarray as xr
from unittest.mock import MagicMock, patch

from pyresample import create_area_def
from pyresample.geometry import AreaDefinition
from pyresample import bucket
from pyresample.test.utils import CustomScheduler


class Test(unittest.TestCase):

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
        self.assertTrue(hasattr(resampler, 'get_average'))
        self.assertTrue(hasattr(resampler, 'get_fractions'))
        self.assertIsNone(resampler.counts)

    def test_round_to_resolution(self):
        """Test rounding to given resolution"""
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

    def test_get_sum(self):
        """Test drop-in-a-bucket sum."""
        data = da.from_array(np.array([[2., 2.], [2., 2.]]),
                             chunks=self.chunks)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_sum(data)

        result = result.compute()
        # One bin with two hits, so max value is 2.0
        self.assertTrue(np.max(result) == 4.)
        # Two bins with the same value
        self.assertEqual(np.sum(result == 2.), 2)
        # One bin with double the value
        self.assertEqual(np.sum(result == 4.), 1)
        self.assertEqual(result.shape, self.adef.shape)

        # Test that also Xarray.DataArrays work
        data = xr.DataArray(data)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_sum(data)
        # One bin with two hits, so max value is 2.0
        self.assertTrue(np.max(result) == 4.)
        # Two bins with the same value
        self.assertEqual(np.sum(result == 2.), 2)
        # One bin with double the value
        self.assertEqual(np.sum(result == 4.), 1)
        self.assertEqual(result.shape, self.adef.shape)

        # Test masking all-NaN bins
        data = da.from_array(np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                             chunks=self.chunks)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_sum(data, mask_all_nan=True)
        self.assertTrue(np.all(np.isnan(result)))
        # By default all-NaN bins have a value of 0.0
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_sum(data)
        self.assertEqual(np.nanmax(result), 0.0)

    def test_get_count(self):
        """Test drop-in-a-bucket sum."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_count()
        result = result.compute()
        self.assertTrue(np.max(result) == 2)
        self.assertEqual(np.sum(result == 1), 2)
        self.assertEqual(np.sum(result == 2), 1)
        self.assertTrue(self.resampler.counts is not None)

    def test_get_average(self):
        """Test averaging bucket resampling."""
        data = da.from_array(np.array([[2., 4.], [3., np.nan]]),
                             chunks=self.chunks)
        # Without pre-calculated indices
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_average(data)
        result = result.compute()
        self.assertEqual(np.nanmax(result), 3.)
        self.assertTrue(np.any(np.isnan(result)))
        # Use a fill value other than np.nan
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_average(data, fill_value=-1)
        result = result.compute()
        self.assertEqual(np.max(result), 3.)
        self.assertEqual(np.min(result), -1)
        self.assertFalse(np.any(np.isnan(result)))

        # Test masking all-NaN bins
        data = da.from_array(np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                             chunks=self.chunks)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_average(data, mask_all_nan=True)
        self.assertTrue(np.all(np.isnan(result)))
        # By default all-NaN bins have a value of NaN
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = self.resampler.get_average(data)
        self.assertTrue(np.all(np.isnan(result)))

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
            result = self.resampler.get_fractions(data, categories=None)
