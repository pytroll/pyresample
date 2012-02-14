from __future__ import with_statement

import os
import unittest
import warnings
warnings.simplefilter("always")

import numpy as np

from pyresample import kd_tree, geometry


def tmp(f):
    f.tmp = True
    return f

class Test(unittest.TestCase):
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                               'test_files', 'ssmis_swath.npz'))
    data = np.load(filename)['data']
    lons = data[:, 0]
    lats = data[:, 1]
    tb37v = data[:, 2]
               
    def test_self_map(self):
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)
        with warnings.catch_warnings(record=True) as w:
            res = kd_tree.resample_gauss(swath_def, self.tb37v.copy(), swath_def, 
                                         radius_of_influence=70000, sigmas=56500)
            self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
            self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')
        self.failUnlessAlmostEqual(res.sum(), 66884817.0303, 3, 
                                msg='Failed self mapping swath for 1 channel')
    
    @tmp                            
    def test_self_map_multi(self):
        data = np.column_stack((self.tb37v, self.tb37v, self.tb37v))
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)
        with warnings.catch_warnings(record=True) as w:
            res = kd_tree.resample_gauss(swath_def, data, swath_def, 
                                         radius_of_influence=70000, sigmas=[56500, 56500, 56500])
            self.failIf(len(w) != 1, 'Failed to create neighbour radius warning')
            self.failIf(('Possible more' not in str(w[0].message)), 'Failed to create correct neighbour radius warning')
        self.failUnlessAlmostEqual(res[:, 0].sum(), 66884817.0303, 3, 
                                   msg='Failed self mapping swath multi for channel 1')
        self.failUnlessAlmostEqual(res[:, 1].sum(), 66884817.0303, 3, 
                                   msg='Failed self mapping swath multi for channel 2')
        self.failUnlessAlmostEqual(res[:, 2].sum(), 66884817.0303, 3, 
                                   msg='Failed self mapping swath multi for channel 3')            
    
