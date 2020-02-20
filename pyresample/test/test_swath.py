import os
import sys
import unittest
import warnings
warnings.simplefilter("always")

import numpy as np
from pyresample.test.utils import catch_warnings
from pyresample import kd_tree, geometry


class Test(unittest.TestCase):

    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'test_files', 'ssmis_swath.npz'))
    data = np.load(filename)['data']
    lons = data[:, 0].astype(np.float64)
    lats = data[:, 1].astype(np.float64)
    tb37v = data[:, 2].astype(np.float64)

    # screen out the fill values
    fvalue = -10000000000.0
    valid_fov = (lons != fvalue) * (lats != fvalue) * (tb37v != fvalue)
    lons = lons[valid_fov]
    lats = lats[valid_fov]
    tb37v = tb37v[valid_fov]

    def test_self_map(self):
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)
        with catch_warnings() as w:
            res = kd_tree.resample_gauss(swath_def, self.tb37v.copy(), swath_def,
                                         radius_of_influence=70000, sigmas=56500)
            self.assertFalse(
                len(w) != 1, 'Failed to create neighbour radius warning')
            self.assertFalse(('Possible more' not in str(
                w[0].message)), 'Failed to create correct neighbour radius warning')

        if sys.platform == 'darwin':
            # OSX seems to get slightly different results for `_spatial_mp.Cartesian`
            truth_value = 668848.144817
        else:
            truth_value = 668848.082208
        self.assertAlmostEqual(res.sum() / 100., truth_value, 1,
                               msg='Failed self mapping swath for 1 channel')

    def test_self_map_multi(self):
        data = np.column_stack((self.tb37v, self.tb37v, self.tb37v))
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)

        with catch_warnings() as w:
            res = kd_tree.resample_gauss(swath_def, data, swath_def,
                                         radius_of_influence=70000, sigmas=[56500, 56500, 56500])
            self.assertFalse(
                len(w) != 1, 'Failed to create neighbour radius warning')
            self.assertFalse(('Possible more' not in str(
                w[0].message)), 'Failed to create correct neighbour radius warning')

        if sys.platform == 'darwin':
            # OSX seems to get slightly different results for `_spatial_mp.Cartesian`
            truth_value = 668848.144817
        else:
            truth_value = 668848.082208
        self.assertAlmostEqual(res[:, 0].sum() / 100., truth_value, 1,
                               msg='Failed self mapping swath multi for channel 1')
        self.assertAlmostEqual(res[:, 1].sum() / 100., truth_value, 1,
                               msg='Failed self mapping swath multi for channel 2')
        self.assertAlmostEqual(res[:, 2].sum() / 100., truth_value, 1,
                               msg='Failed self mapping swath multi for channel 3')


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
