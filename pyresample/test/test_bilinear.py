import unittest
import numpy as np

from pyproj import Proj

import pyresample.bilinear as bil
from pyresample import geometry, utils


class Test(unittest.TestCase):

    pts_irregular = (np.array([[-1., 1.], ]),
                     np.array([[1., 2.], ]),
                     np.array([[-2., -1.], ]),
                     np.array([[2., -4.], ]))
    pts_vert_parallel = (np.array([[-1., 1.], ]),
                         np.array([[1., 2.], ]),
                         np.array([[-1., -1.], ]),
                         np.array([[1., -2.], ]))
    pts_both_parallel = (np.array([[-1., 1.], ]),
                         np.array([[1., 1.], ]),
                         np.array([[-1., -1.], ]),
                         np.array([[1., -1.], ]))

    def test_find_vert_parallels(self):
        res = bil._find_vert_parallels(*self.pts_both_parallel)
        self.assertTrue(res[0])
        res = bil._find_vert_parallels(*self.pts_vert_parallel)
        self.assertTrue(res[0])
        res = bil._find_vert_parallels(*self.pts_irregular)
        self.assertFalse(res[0])

    def test_find_horiz_parallels(self):
        res = bil._find_horiz_parallels(*self.pts_both_parallel)
        self.assertTrue(res[0])
        res = bil._find_horiz_parallels(*self.pts_vert_parallel)
        self.assertFalse(res[0])
        res = bil._find_horiz_parallels(*self.pts_irregular)
        self.assertFalse(res[0])

    def test_get_ts_irregular(self):
        res = bil._get_ts_irregular(self.pts_irregular[0],
                                    self.pts_irregular[1],
                                    self.pts_irregular[2],
                                    self.pts_irregular[3],
                                    0., 0.)
        self.assertEqual(res[0], 0.375)
        self.assertEqual(res[1], 0.5)

    def test_get_ts_uprights_parallel(self):
        res = bil._get_ts_uprights_parallel(self.pts_vert_parallel[0],
                                            self.pts_vert_parallel[1],
                                            self.pts_vert_parallel[2],
                                            self.pts_vert_parallel[3],
                                            0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertAlmostEqual(res[1], 0.6513878, 5)

    def test_get_ts_parallellogram(self):
        res = bil._get_ts_parallellogram(self.pts_both_parallel[0],
                                         self.pts_both_parallel[1],
                                         self.pts_both_parallel[2],
                                         0., 0.)
        self.assertEqual(res[0], 0.5)
        self.assertEqual(res[1], 0.5)

    def test_solve_quadratic(self):
        res = bil._solve_quadratic(1, 0, 0)
        self.assertEqual(res[0], 0.0)
        res = bil._solve_quadratic(1, 2, 1)
        self.assertTrue(np.isnan(res[0]))
        res = bil._solve_quadratic(1, 2, 1, min_val=-2.)
        self.assertEqual(res[0], -1.0)


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
