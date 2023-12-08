#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2021 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test resampling swath definitions."""

import os
import unittest
import warnings

import numpy as np

from pyresample import geometry, kd_tree
from pyresample.test.utils import TEST_FILES_PATH, catch_warnings

warnings.simplefilter("always")


class Test(unittest.TestCase):
    """Tests for swath definitions."""

    filename = os.path.abspath(os.path.join(TEST_FILES_PATH, 'ssmis_swath.npz'))
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
            self.assertEqual(len(w), 1,
                             'Failed to create neighbour radius warning')
            self.assertFalse(('Possible more' not in str(
                w[0].message)), 'Failed to create correct neighbour radius warning')

        # only compare the whole number as different OSes and versions of numpy
        # can produce slightly different results
        truth_value = 668848.0
        self.assertAlmostEqual(res.sum() / 100., truth_value, 0,
                               msg='Failed self mapping swath for 1 channel')

    def test_self_map_multi(self):
        data = np.column_stack((self.tb37v, self.tb37v, self.tb37v))
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)

        with catch_warnings() as w:
            res = kd_tree.resample_gauss(swath_def, data, swath_def,
                                         radius_of_influence=70000, sigmas=[56500, 56500, 56500])
            self.assertEqual(len(w), 1,
                             'Failed to create neighbour radius warning')
            self.assertFalse(('Possible more' not in str(
                w[0].message)), 'Failed to create correct neighbour radius warning')

        truth_value = 668848.0
        self.assertAlmostEqual(res[:, 0].sum() / 100., truth_value, 0,
                               msg='Failed self mapping swath multi for channel 1')
        self.assertAlmostEqual(res[:, 1].sum() / 100., truth_value, 0,
                               msg='Failed self mapping swath multi for channel 2')
        self.assertAlmostEqual(res[:, 2].sum() / 100., truth_value, 0,
                               msg='Failed self mapping swath multi for channel 3')
