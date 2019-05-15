import unittest
import numpy as np
import dask.array as da
import dask
import xarray as xr
try:
    from unittest.mock import MagicMock, patch
except ImportError:
    # separate mock package py<3.3
    from mock import MagicMock, patch

from pyresample.geometry import AreaDefinition
import pyresample.bucket as bucket
from pyresample.test.utils import CustomScheduler


class Test(unittest.TestCase):

    adef = AreaDefinition('eurol', 'description', '',
                          {'ellps': 'WGS84',
                           'lat_0': '90.0',
                           'lat_ts': '60.0',
                           'lon_0': '0.0',
                           'proj': 'stere'}, 2560, 2048,
                          (-3780000.0, -7644000.0, 3900000.0, -1500000.0))

    lons = da.from_array(np.array([[25., 25.], [25., 25.]]))
    lats = da.from_array(np.array([[60., 60.00001], [60.2, 60.3]]))

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
        x_res, y_res = 0.5, 0.5
        result = bucket._get_proj_coordinates(lons, lats, x_res, y_res, prj)
        prj.assert_called_once_with(lons, lats)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(np.all(result == np.array([[3., 3., 3.],
                                                   [5., 5., 5.]])))

    def test_get_bucket_indices(self):
        """Test calculation of array indices."""
        # Ensure nothing is calculated
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            x_idxs, y_idxs = bucket.get_bucket_indices(self.adef, self.lons,
                                                       self.lats)
        x_idxs, y_idxs = da.compute(x_idxs, y_idxs)
        self.assertTrue(np.all(x_idxs == np.array([1709, 1709, 1706, 1705])))
        self.assertTrue(np.all(y_idxs == np.array([465, 465, 458, 455])))

    def test_get_sum_from_bucket_indices(self):
        """Test drop-in-a-bucket sum."""
        x_idxs, y_idxs = bucket.get_bucket_indices(self.adef, self.lons,
                                                   self.lats)
        data = da.from_array(np.array([[2., 2.], [2., 2.]]))
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = bucket.get_sum_from_bucket_indices(data, x_idxs, y_idxs,
                                                        self.adef.shape)
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
            result = bucket.get_sum_from_bucket_indices(data, x_idxs, y_idxs,
                                                        self.adef.shape)
        # One bin with two hits, so max value is 2.0
        self.assertTrue(np.max(result) == 4.)
        # Two bins with the same value
        self.assertEqual(np.sum(result == 2.), 2)
        # One bin with double the value
        self.assertEqual(np.sum(result == 4.), 1)
        self.assertEqual(result.shape, self.adef.shape)

    def test_get_count_from_bucket_indices(self):
        """Test drop-in-a-bucket sum."""
        x_idxs, y_idxs = bucket.get_bucket_indices(self.adef, self.lons,
                                                   self.lats)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = bucket.get_count_from_bucket_indices(x_idxs, y_idxs,
                                                          self.adef.shape)
        result = result.compute()
        self.assertTrue(np.max(result) == 2)
        self.assertEqual(np.sum(result == 1), 2)
        self.assertEqual(np.sum(result == 2), 1)

    def test_resample_bucket_average(self):
        """Test averaging bucket resampling."""
        data = da.from_array(np.array([[2., 4.], [2., 2.]]))
        # Without pre-calculated indices
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = bucket.resample_bucket_average(self.adef,
                                                    data, self.lons, self.lats)
        result = result.compute()
        self.assertEqual(np.nanmax(result), 3.)
        self.assertTrue(np.any(np.isnan(result)))
        # Use a fill value other than np.nan
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = bucket.resample_bucket_average(self.adef,
                                                    data, self.lons, self.lats,
                                                    fill_value=-1)
        result = result.compute()
        self.assertEqual(np.max(result), 3.)
        self.assertEqual(np.min(result), -1)
        self.assertFalse(np.any(np.isnan(result)))
        # Pre-calculate the indices
        x_idxs, y_idxs = bucket.get_bucket_indices(self.adef, self.lons,
                                                   self.lats)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            result = bucket.resample_bucket_average(self.adef,
                                                    data, self.lons, self.lats,
                                                    x_idxs=x_idxs,
                                                    y_idxs=y_idxs)

    def test_resample_bucket_fractions(self):
        """Test fraction calculations for categorical data."""
        data = da.from_array(np.array([[2, 4], [2, 2]]))
        categories = [1, 2, 3, 4]
        # Without pre-calculated indices
        with dask.config.set(scheduler=CustomScheduler(max_computes=10)):
            result = bucket.resample_bucket_fractions(self.adef,
                                                      data, self.lons,
                                                      self.lats,
                                                      categories)
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


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
