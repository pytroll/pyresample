import unittest
import numpy as np
import dask.array as da

from pyresample._spatial_mp import Proj

import pyresample.bucket as bucket


class Test(unittest.TestCase):

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


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
