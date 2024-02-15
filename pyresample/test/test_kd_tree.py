#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
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
"""Test kd_tree operations."""
import os
import unittest
from unittest import mock

import numpy as np
import pytest

from pyresample import geometry, kd_tree, utils
from pyresample.test.utils import TEST_FILES_PATH, catch_warnings


class Test(unittest.TestCase):
    """Test nearest neighbor resampling on numpy arrays."""

    @classmethod
    def setUpClass(cls):
        cls.area_def = geometry.AreaDefinition('areaD',
                                               'Europe (3km, HRV, VTC)',
                                               'areaD',
                                               {'a': '6378144.0',
                                                'b': '6356759.0',
                                                'lat_0': '50.00',
                                                'lat_ts': '50.00',
                                                'lon_0': '8.00',
                                                'proj': 'stere'},
                                               800,
                                               800,
                                               [-1370912.72,
                                                   -909968.64000000001,
                                                   1029087.28,
                                                   1490031.3600000001])

        cls.tdata = np.array([1, 2, 3])
        cls.tlons = np.array([11.280789, 12.649354, 12.080402])
        cls.tlats = np.array([56.011037, 55.629675, 55.641535])
        cls.tswath = geometry.SwathDefinition(lons=cls.tlons, lats=cls.tlats)
        cls.tgrid = geometry.CoordinateDefinition(
            lons=np.array([12.562036]), lats=np.array([55.715613]))

    def test_nearest_base(self):
        res = kd_tree.resample_nearest(self.tswath,
                                       self.tdata.ravel(), self.tgrid,
                                       100000, reduce_data=False, segments=1)
        self.assertTrue(res[0] == 2)

    def test_gauss_base(self):
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_gauss(self.tswath,
                                         self.tdata.ravel(), self.tgrid,
                                         50000, 25000, reduce_data=False, segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Searching' not in str(w[0].message)))
        self.assertAlmostEqual(res[0], 2.2020729, 5)

    def test_custom_base(self):
        def wf(dist):
            return 1 - dist / 100000.0

        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_custom(self.tswath,
                                          self.tdata.ravel(), self.tgrid,
                                          50000, wf, reduce_data=False, segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Searching' not in str(w[0].message)))
        self.assertAlmostEqual(res[0], 2.4356757, 5)

    def test_gauss_uncert(self):
        sigma = utils.fwhm2sigma(41627.730557884883)
        with catch_warnings(UserWarning) as w:
            res, stddev, count = kd_tree.resample_gauss(self.tswath, self.tdata,
                                                        self.tgrid, 100000, sigma,
                                                        with_uncert=True)
            self.assertTrue(len(w) > 0)
            self.assertTrue((any('Searching' in str(_w.message) for _w in w)))

        expected_res = 2.20206560694
        expected_stddev = 0.707115076173
        expected_count = 3
        self.assertAlmostEqual(res[0], expected_res, 5)
        self.assertAlmostEqual(stddev[0], expected_stddev, 5)
        self.assertEqual(count[0], expected_count)

    def test_custom_uncert(self):
        def wf(dist):
            return 1 - dist / 100000.0

        with catch_warnings(UserWarning) as w:
            res, stddev, counts = kd_tree.resample_custom(self.tswath,
                                                          self.tdata, self.tgrid,
                                                          100000, wf, with_uncert=True)
            self.assertTrue(len(w) > 0)
            self.assertTrue((any('Searching' in str(_w.message) for _w in w)))

        self.assertAlmostEqual(res[0], 2.32193149, 5)
        self.assertAlmostEqual(stddev[0], 0.81817972, 5)
        self.assertEqual(counts[0], 3)

    def test_nearest(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, segments=1)
        cross_sum = res.sum()
        expected = 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_complex(self):
        data = np.fromfunction(lambda y, x: y + complex("j") * x, (50, 10), dtype=np.complex128)
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, segments=1)
        assert np.issubdtype(res.dtype, np.complex128)
        cross_sum = res.sum()
        assert cross_sum.real == 3530219.0
        assert cross_sum.imag == 688723.0

    def test_nearest_masked_swath_target(self):
        """Test that a masked array works as a target."""
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        mask = np.ones_like(lons, dtype=np.bool_)
        mask[::2, ::2] = False
        swath_def = geometry.SwathDefinition(
            lons=np.ma.masked_array(lons, mask=mask),
            lats=np.ma.masked_array(lats, mask=False)
        )
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       swath_def, 50000, segments=3)
        cross_sum = res.sum()
        # expected = 12716  # if masks aren't respected
        expected = 12000
        self.assertEqual(cross_sum, expected)

    def test_nearest_1d(self):
        data = np.fromfunction(lambda x, y: x * y, (800, 800))
        lons = np.fromfunction(lambda x: 3 + x / 100., (500,))
        lats = np.fromfunction(lambda x: 75 - x / 10., (500,))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(self.area_def, data.ravel(),
                                       swath_def, 50000, segments=1)
        cross_sum = res.sum()
        expected = 35821299.0
        self.assertEqual(res.shape, (500,))
        self.assertEqual(cross_sum, expected)

    def test_nearest_empty(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, segments=1)
        cross_sum = res.sum()
        expected = 0
        self.assertEqual(cross_sum, expected)

    def test_nearest_empty_multi(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data_multi,
                                       self.area_def, 50000, segments=1)
        self.assertEqual(res.shape, (800, 800, 3),
                         msg='Swath resampling nearest empty multi failed')

    def test_nearest_empty_multi_masked(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data_multi,
                                       self.area_def, 50000, segments=1,
                                       fill_value=None)
        self.assertEqual(res.shape, (800, 800, 3))

    def test_nearest_empty_masked(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 165 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, segments=1,
                                       fill_value=None)
        cross_sum = res.mask.sum()
        expected = res.size
        self.assertTrue(cross_sum == expected)

    def test_nearest_segments(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, segments=2)
        cross_sum = res.sum()
        expected = 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_remap(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, segments=1)
        remap = kd_tree.resample_nearest(self.area_def, res.ravel(),
                                         swath_def, 5000, segments=1)
        cross_sum = remap.sum()
        expected = 22275.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_mp(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, nprocs=2, segments=1)
        cross_sum = res.sum()
        expected = 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_multi(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        res = kd_tree.resample_nearest(swath_def, data_multi,
                                       self.area_def, 50000, segments=1)
        cross_sum = res.sum()
        expected = 3 * 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_multi_unraveled(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.dstack((data, data, data))
        res = kd_tree.resample_nearest(swath_def, data_multi,
                                       self.area_def, 50000, segments=1)
        cross_sum = res.sum()
        expected = 3 * 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_gauss_sparse(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_gauss(swath_def, data.ravel(),
                                     self.area_def, 50000, 25000, fill_value=-1, segments=1)
        cross_sum = res.sum()
        expected = 15387753.9852
        self.assertAlmostEqual(cross_sum, expected, places=3)

    def test_gauss(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -5, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_gauss(swath_def, data.ravel(),
                                         self.area_def, 50000, 25000, segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Possible more' not in str(w[0].message)))
        cross_sum = res.sum()
        expected = 4872.8100353517921
        self.assertAlmostEqual(cross_sum, expected)

    def test_gauss_fwhm(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -5, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_gauss(swath_def, data.ravel(),
                                         self.area_def, 50000, utils.fwhm2sigma(41627.730557884883), segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Possible more' not in str(w[0].message)))
        cross_sum = res.sum()
        expected = 4872.8100353517921
        self.assertAlmostEqual(cross_sum, expected)

    def test_gauss_multi(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_gauss(swath_def, data_multi,
                                         self.area_def, 50000, [25000, 15000, 10000], segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Possible more' not in str(w[0].message)))
        cross_sum = res.sum()
        expected = 1461.8429990248171
        self.assertAlmostEqual(cross_sum, expected)

    def test_gauss_multi_uncert(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        with catch_warnings(UserWarning) as w:
            # The assertion below checks if there is only one warning raised
            # and whether it contains a specific message from pyresample
            # On python 2.7.9+ the resample_gauss method raises multiple deprecation warnings
            # that cause to fail, so we ignore the unrelated warnings.
            res, stddev, counts = kd_tree.resample_gauss(swath_def, data_multi,
                                                         self.area_def, 50000, [
                                                             25000, 15000, 10000],
                                                         segments=1, with_uncert=True)
            self.assertTrue(len(w) >= 1)
            self.assertTrue(
                any(['Possible more' in str(x.message) for x in w]))
        cross_sum = res.sum()
        cross_sum_counts = counts.sum()
        expected = 1461.8429990248171
        expected_stddev = [0.44621800779801657, 0.44363137712896705,
                           0.43861019464274459]
        expected_counts = 4934802.0
        self.assertTrue(res.shape == stddev.shape and stddev.shape == counts.shape and counts.shape == (800, 800, 3))
        self.assertAlmostEqual(cross_sum, expected)

        for i, e_stddev in enumerate(expected_stddev):
            cross_sum_stddev = stddev[:, :, i].sum()
            self.assertAlmostEqual(cross_sum_stddev, e_stddev)
        self.assertAlmostEqual(cross_sum_counts, expected_counts)

    def test_gauss_multi_mp(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_gauss(swath_def, data_multi,
                                         self.area_def, 50000, [
                                             25000, 15000, 10000],
                                         nprocs=2, segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Possible more' not in str(w[0].message)))
        cross_sum = res.sum()
        expected = 1461.8429990248171
        self.assertAlmostEqual(cross_sum, expected)

    def test_gauss_multi_mp_segments(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_gauss(swath_def, data_multi,
                                         self.area_def, 50000, [
                                             25000, 15000, 10000],
                                         nprocs=2, segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse('Possible more' not in str(w[0].message))
        cross_sum = res.sum()
        expected = 1461.8429990248171
        self.assertAlmostEqual(cross_sum, expected)

    def test_gauss_multi_mp_segments_empty(self):
        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 165 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        res = kd_tree.resample_gauss(swath_def, data_multi,
                                     self.area_def, 50000, [
                                         25000, 15000, 10000],
                                     nprocs=2, segments=1)
        cross_sum = res.sum()
        self.assertTrue(cross_sum == 0)

    def test_custom(self):
        def wf(dist):
            return 1 - dist / 100000.0

        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -5, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_custom(swath_def, data.ravel(),
                                          self.area_def, 50000, wf, segments=1)
            # PyProj proj/CRS and "more than 8 neighbours" are warned about
            self.assertFalse(len(w) > 2)
            neighbour_warn = False
            for warn in w:
                if 'Possible more' in str(warn.message):
                    neighbour_warn = True
                    break
            self.assertTrue(neighbour_warn)
            if len(w) == 2:
                proj_crs_warn = False
                for warn in w:
                    if 'important projection information' in str(warn.message):
                        proj_crs_warn = True
                        break
                self.assertTrue(proj_crs_warn)

        cross_sum = res.sum()
        expected = 4872.8100347930776
        self.assertAlmostEqual(cross_sum, expected)

    def test_custom_multi(self):
        def wf1(dist):
            return 1 - dist / 100000.0

        def wf2(dist):
            return 1

        def wf3(dist):
            return np.cos(dist) ** 2

        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))
        with catch_warnings(UserWarning) as w:
            res = kd_tree.resample_custom(swath_def, data_multi,
                                          self.area_def, 50000, [wf1, wf2, wf3], segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse('Possible more' not in str(w[0].message))
        cross_sum = res.sum()
        expected = 1461.8428378742638
        self.assertAlmostEqual(cross_sum, expected)

    def test_masked_nearest(self):
        data = np.ones((50, 10))
        data[:, 5:] = 2
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        mask = np.ones((50, 10))
        mask[:, :5] = 0
        masked_data = np.ma.array(data, mask=mask)
        res = kd_tree.resample_nearest(swath_def, masked_data.ravel(),
                                       self.area_def, 50000, segments=1)
        expected_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_nearest_mask.dat'),
                                    sep=' ').reshape((800, 800))
        expected_data = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_nearest_data.dat'),
                                    sep=' ').reshape((800, 800))
        self.assertTrue(np.array_equal(expected_mask, res.mask))
        self.assertTrue(np.array_equal(expected_data, res.data))

    def test_masked_nearest_1d(self):
        data = np.ones((800, 800))
        data[:400, :] = 2
        lons = np.fromfunction(lambda x: 3 + x / 100., (500,))
        lats = np.fromfunction(lambda x: 75 - x / 10., (500,))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        mask = np.ones((800, 800))
        mask[400:, :] = 0
        masked_data = np.ma.array(data, mask=mask)
        res = kd_tree.resample_nearest(self.area_def, masked_data.ravel(),
                                       swath_def, 50000, segments=1)
        self.assertEqual(res.mask.sum(), 112)

    def test_masked_gauss(self):
        data = np.ones((50, 10))
        data[:, 5:] = 2
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        mask = np.ones((50, 10))
        mask[:, :5] = 0
        masked_data = np.ma.array(data, mask=mask)
        res = kd_tree.resample_gauss(swath_def, masked_data.ravel(),
                                     self.area_def, 50000, 25000, segments=1)
        expected_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_mask.dat'),
                                    sep=' ').reshape((800, 800))
        expected_data = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_data.dat'),
                                    sep=' ').reshape((800, 800))
        expected = expected_data.sum()
        cross_sum = res.data.sum()

        self.assertTrue(np.array_equal(expected_mask, res.mask))
        self.assertAlmostEqual(cross_sum, expected, places=3)

    def test_masked_fill_float(self):
        data = np.ones((50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, fill_value=None, segments=1)
        expected_fill_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_fill_value.dat'),
                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask
        self.assertTrue(np.array_equal(fill_mask, expected_fill_mask))

    def test_masked_fill_int(self):
        data = np.ones((50, 10)).astype('int')
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def, data.ravel(),
                                       self.area_def, 50000, fill_value=None, segments=1)
        expected_fill_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_fill_value.dat'),
                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask
        self.assertTrue(np.array_equal(fill_mask, expected_fill_mask))

    def test_masked_full(self):
        data = np.ones((50, 10))
        data[:, 5:] = 2
        mask = np.ones((50, 10))
        mask[:, :5] = 0
        masked_data = np.ma.array(data, mask=mask)
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def,
                                       masked_data.ravel(
                                       ), self.area_def, 50000,
                                       fill_value=None, segments=1)
        expected_fill_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_full_fill.dat'),
                                         sep=' ').reshape((800, 800))
        fill_mask = res.mask

        self.assertTrue(np.array_equal(fill_mask, expected_fill_mask))

    def test_masked_full_multi(self):
        data = np.ones((50, 10))
        data[:, 5:] = 2
        mask1 = np.ones((50, 10))
        mask1[:, :5] = 0
        mask2 = np.ones((50, 10))
        mask2[:, 5:] = 0
        mask3 = np.ones((50, 10))
        mask3[:25, :] = 0
        data_multi = np.column_stack(
            (data.ravel(), data.ravel(), data.ravel()))
        mask_multi = np.column_stack(
            (mask1.ravel(), mask2.ravel(), mask3.ravel()))
        masked_data = np.ma.array(data_multi, mask=mask_multi)
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        res = kd_tree.resample_nearest(swath_def,
                                       masked_data, self.area_def, 50000,
                                       fill_value=None, segments=1)
        expected_fill_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_full_fill_multi.dat'),
                                         sep=' ').reshape((800, 800, 3))
        fill_mask = res.mask
        cross_sum = res.sum()
        expected = 357140.0
        self.assertAlmostEqual(cross_sum, expected)
        self.assertTrue(np.array_equal(fill_mask, expected_fill_mask))

    def test_dtype(self):
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        grid_def = geometry.GridDefinition(lons, lats)
        lons = np.asarray(lons, dtype='f4')
        lats = np.asarray(lats, dtype='f4')
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        valid_input_index, valid_output_index, index_array, distance_array = \
            kd_tree.get_neighbour_info(swath_def,
                                       grid_def,
                                       50000, neighbours=1, segments=1)

    def test_nearest_from_sample(self):
        data = np.fromfunction(lambda y, x: y * x, (50, 10))
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        valid_input_index, valid_output_index, index_array, distance_array = \
            kd_tree.get_neighbour_info(swath_def,
                                       self.area_def,
                                       50000, neighbours=1, segments=1)
        res = kd_tree.get_sample_from_neighbour_info('nn', (800, 800), data.ravel(),
                                                     valid_input_index, valid_output_index,
                                                     index_array)
        cross_sum = res.sum()
        expected = 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_from_sample_np_dtypes(self):
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        valid_input_index, valid_output_index, index_array, distance_array = \
            kd_tree.get_neighbour_info(swath_def,
                                       self.area_def,
                                       50000, neighbours=1, segments=1)

        for dtype in [np.uint16, np.float32]:
            with self.subTest(dtype):
                data = np.fromfunction(lambda y, x: y * x, (50, 10)).astype(dtype)
                fill_value = dtype(0.0)
                res = \
                    kd_tree.get_sample_from_neighbour_info('nn', (800, 800),
                                                           data.ravel(),
                                                           valid_input_index,
                                                           valid_output_index,
                                                           index_array,
                                                           fill_value=fill_value)
                cross_sum = res.sum()
                expected = 15874591.0
                self.assertEqual(cross_sum, expected)

    def test_custom_multi_from_sample(self):
        def wf1(dist):
            return 1 - dist / 100000.0

        def wf2(dist):
            return 1

        def wf3(dist):
            return np.cos(dist) ** 2

        data = np.fromfunction(lambda y, x: (y + x) * 10 ** -6, (5000, 100))
        lons = np.fromfunction(
            lambda y, x: 3 + (10.0 / 100) * x, (5000, 100))
        lats = np.fromfunction(
            lambda y, x: 75 - (50.0 / 5000) * y, (5000, 100))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        data_multi = np.column_stack((data.ravel(), data.ravel(),
                                      data.ravel()))

        with catch_warnings(UserWarning) as w:
            valid_input_index, valid_output_index, index_array, distance_array = \
                kd_tree.get_neighbour_info(swath_def,
                                           self.area_def,
                                           50000, segments=1)
            self.assertFalse(len(w) != 1)
            self.assertFalse(('Possible more' not in str(w[0].message)))

        res = kd_tree.get_sample_from_neighbour_info('custom', (800, 800),
                                                     data_multi,
                                                     valid_input_index, valid_output_index,
                                                     index_array, distance_array,
                                                     weight_funcs=[wf1, wf2, wf3])

        cross_sum = res.sum()

        expected = 1461.8428378742638
        self.assertAlmostEqual(cross_sum, expected)
        res = kd_tree.get_sample_from_neighbour_info('custom', (800, 800),
                                                     data_multi,
                                                     valid_input_index, valid_output_index,
                                                     index_array, distance_array,
                                                     weight_funcs=[wf1, wf2, wf3])

        # Look for error where input data has been manipulated
        cross_sum = res.sum()
        expected = 1461.8428378742638
        self.assertAlmostEqual(cross_sum, expected)

    def test_masked_multi_from_sample(self):
        data = np.ones((50, 10))
        data[:, 5:] = 2
        mask1 = np.ones((50, 10))
        mask1[:, :5] = 0
        mask2 = np.ones((50, 10))
        mask2[:, 5:] = 0
        mask3 = np.ones((50, 10))
        mask3[:25, :] = 0
        data_multi = np.column_stack(
            (data.ravel(), data.ravel(), data.ravel()))
        mask_multi = np.column_stack(
            (mask1.ravel(), mask2.ravel(), mask3.ravel()))
        masked_data = np.ma.array(data_multi, mask=mask_multi)
        lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        valid_input_index, valid_output_index, index_array, distance_array = \
            kd_tree.get_neighbour_info(swath_def,
                                       self.area_def,
                                       50000, neighbours=1, segments=1)
        res = kd_tree.get_sample_from_neighbour_info('nn', (800, 800),
                                                     masked_data,
                                                     valid_input_index,
                                                     valid_output_index, index_array,
                                                     fill_value=None)
        expected_fill_mask = np.fromfile(os.path.join(TEST_FILES_PATH, 'mask_test_full_fill_multi.dat'),
                                         sep=' ').reshape((800, 800, 3))
        fill_mask = res.mask
        self.assertTrue(np.array_equal(fill_mask, expected_fill_mask))


class TestXArrayResamplerNN(unittest.TestCase):
    """Test the XArrayResamplerNN class."""

    @classmethod
    def setUpClass(cls):
        import dask.array as da
        import xarray as xr
        cls.area_def = geometry.AreaDefinition('areaD',
                                               'Europe (3km, HRV, VTC)',
                                               'areaD',
                                               {'a': '6378144.0',
                                                'b': '6356759.0',
                                                'lat_0': '50.00',
                                                'lat_ts': '50.00',
                                                'lon_0': '8.00',
                                                'proj': 'stere'},
                                               800,
                                               800,
                                               [-1370912.72,
                                                -909968.64000000001,
                                                1029087.28,
                                                1490031.3600000001])

        dfa = da.from_array  # shortcut
        cls.chunks = chunks = 5
        cls.tgrid = geometry.CoordinateDefinition(
            lons=dfa(np.array([
                [11.5, 12.562036, 12.9],
                [11.5, 12.562036, 12.9],
                [11.5, 12.562036, 12.9],
                [11.5, 12.562036, 12.9],
            ]), chunks=chunks),
            lats=dfa(np.array([
                [55.715613, 55.715613, 55.715613],
                [55.715613, 55.715613, 55.715613],
                [55.715613, np.nan, 55.715613],
                [55.715613, 55.715613, 55.715613],
            ]), chunks=chunks))

        cls.tdata_1d = xr.DataArray(
            dfa(np.array([1., 2., 3.]), chunks=chunks), dims=('my_dim1',))
        cls.tlons_1d = xr.DataArray(
            dfa(np.array([11.280789, 12.649354, 12.080402]), chunks=chunks),
            dims=('my_dim1',))
        cls.tlats_1d = xr.DataArray(
            dfa(np.array([56.011037, 55.629675, 55.641535]), chunks=chunks),
            dims=('my_dim1',))
        cls.tswath_1d = geometry.SwathDefinition(lons=cls.tlons_1d,
                                                 lats=cls.tlats_1d)

        cls.data_2d = xr.DataArray(
            da.from_array(np.fromfunction(lambda y, x: y * x, (50, 10)),
                          chunks=5),
            dims=('my_dim_y', 'my_dim_x'))
        cls.data_3d = xr.DataArray(
            da.from_array(np.fromfunction(lambda y, x, b: y * x * b, (50, 10, 3)),
                          chunks=5),
            dims=('my_dim_y', 'my_dim_x', 'bands'),
            coords={'bands': ['r', 'g', 'b']})
        cls.lons_2d = xr.DataArray(
            da.from_array(np.fromfunction(lambda y, x: 3 + x, (50, 10)),
                          chunks=5),
            dims=('my_dim_y', 'my_dim_x'))
        cls.lats_2d = xr.DataArray(
            da.from_array(np.fromfunction(lambda y, x: 75 - y, (50, 10)),
                          chunks=5),
            dims=('my_dim_y', 'my_dim_x'))
        cls.swath_def_2d = geometry.SwathDefinition(lons=cls.lons_2d,
                                                    lats=cls.lats_2d)
        cls.src_area_2d = geometry.AreaDefinition(
            'areaD_src', 'Europe (3km, HRV, VTC)', 'areaD',
            {'a': '6378144.0', 'b': '6356759.0', 'lat_0': '52.00',
             'lat_ts': '52.00', 'lon_0': '5.00', 'proj': 'stere'}, 50, 10,
            [-1370912.72, -909968.64000000001, 1029087.28,
             1490031.3600000001])

    def test_nearest_swath_1d_mask_to_grid_1n(self):
        """Test 1D swath definition to 2D grid definition; 1 neighbor."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        resampler = XArrayResamplerNN(self.tswath_1d, self.tgrid,
                                      radius_of_influence=100000,
                                      neighbours=1)
        data = self.tdata_1d
        ninfo = resampler.get_neighbour_info(mask=data.isnull())
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        res = resampler.get_sample_from_neighbour_info(data)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        actual = res.values
        expected = np.array([
            [1., 2., 2.],
            [1., 2., 2.],
            [1., np.nan, 2.],
            [1., 2., 2.],
        ])
        np.testing.assert_allclose(actual, expected)

    def test_nearest_type_preserve(self):
        """Test 1D swath definition to 2D grid definition; 1 neighbor."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        resampler = XArrayResamplerNN(self.tswath_1d, self.tgrid,
                                      radius_of_influence=100000,
                                      neighbours=1)
        data = self.tdata_1d
        data = xr.DataArray(da.from_array(np.array([1, 2, 3]),
                                          chunks=5),
                            dims=('my_dim1',))
        ninfo = resampler.get_neighbour_info()
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        res = resampler.get_sample_from_neighbour_info(data, fill_value=255)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        actual = res.values
        expected = np.array([
            [1, 2, 2],
            [1, 2, 2],
            [1, 255, 2],
            [1, 2, 2],
        ])
        np.testing.assert_equal(actual, expected)

    def test_nearest_swath_2d_mask_to_area_1n(self):
        """Test 2D swath definition to 2D area definition; 1 neighbor."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        swath_def = self.swath_def_2d
        data = self.data_2d
        resampler = XArrayResamplerNN(swath_def, self.area_def,
                                      radius_of_influence=50000,
                                      neighbours=1)
        ninfo = resampler.get_neighbour_info(mask=data.isnull())
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        res = resampler.get_sample_from_neighbour_info(data)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        res = res.values
        cross_sum = np.nansum(res)
        expected = 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_area_2d_to_area_1n(self):
        """Test 2D area definition to 2D area definition; 1 neighbor."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        from pyresample.test.utils import assert_maximum_dask_computes
        data = self.data_2d
        resampler = XArrayResamplerNN(self.src_area_2d, self.area_def,
                                      radius_of_influence=50000,
                                      neighbours=1)
        with assert_maximum_dask_computes(0):
            ninfo = resampler.get_neighbour_info()
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        with pytest.raises(ValueError):
            resampler.get_sample_from_neighbour_info(data)

        # rename data dimensions to match the expected area dimensions
        data = data.rename({'my_dim_y': 'y', 'my_dim_x': 'x'})
        with assert_maximum_dask_computes(0):
            res = resampler.get_sample_from_neighbour_info(data)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        res = res.values
        cross_sum = np.nansum(res)
        expected = 27706753.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_area_2d_to_area_1n_no_roi(self):
        """Test 2D area definition to 2D area definition; 1 neighbor, no radius of influence."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        data = self.data_2d
        resampler = XArrayResamplerNN(self.src_area_2d, self.area_def,
                                      neighbours=1)
        ninfo = resampler.get_neighbour_info()
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        with pytest.raises(ValueError):
            resampler.get_sample_from_neighbour_info(data)

        # rename data dimensions to match the expected area dimensions
        data = data.rename({'my_dim_y': 'y', 'my_dim_x': 'x'})
        res = resampler.get_sample_from_neighbour_info(data)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        res = res.values
        cross_sum = np.nansum(res)
        expected = 87281406.0
        self.assertEqual(cross_sum, expected)

        # pretend the resolutions can't be determined
        with mock.patch.object(self.src_area_2d, 'geocentric_resolution') as sgr, \
                mock.patch.object(self.area_def, 'geocentric_resolution') as dgr:
            sgr.side_effect = RuntimeError
            dgr.side_effect = RuntimeError
            resampler = XArrayResamplerNN(self.src_area_2d, self.area_def,
                                          neighbours=1)
            resampler.get_neighbour_info()
            res = resampler.get_sample_from_neighbour_info(data)
            self.assertIsInstance(res, xr.DataArray)
            self.assertIsInstance(res.data, da.Array)
            res = res.values
            cross_sum = np.nansum(res)
            expected = 1855928.0
            self.assertEqual(cross_sum, expected)

    def test_nearest_area_2d_to_area_1n_3d_data(self):
        """Test 2D area definition to 2D area definition; 1 neighbor, 3d data."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        data = self.data_3d
        resampler = XArrayResamplerNN(self.src_area_2d, self.area_def,
                                      radius_of_influence=50000,
                                      neighbours=1)
        ninfo = resampler.get_neighbour_info()
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        with pytest.raises(ValueError):
            resampler.get_sample_from_neighbour_info(data)

        # rename data dimensions to match the expected area dimensions
        data = data.rename({'my_dim_y': 'y', 'my_dim_x': 'x'})
        res = resampler.get_sample_from_neighbour_info(data)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertCountEqual(res.coords['bands'], ['r', 'g', 'b'])
        res = res.values
        cross_sum = np.nansum(res)
        expected = 83120259.0
        self.assertEqual(cross_sum, expected)

    @unittest.skipIf(True, "Multiple neighbors not supported yet")
    def test_nearest_swath_1d_mask_to_grid_8n(self):
        """Test 1D swath definition to 2D grid definition; 8 neighbors."""
        import dask.array as da
        import xarray as xr

        from pyresample.kd_tree import XArrayResamplerNN
        resampler = XArrayResamplerNN(self.tswath_1d, self.tgrid,
                                      radius_of_influence=100000,
                                      neighbours=8)
        data = self.tdata_1d
        ninfo = resampler.get_neighbour_info(mask=data.isnull())
        for val in ninfo[:3]:
            # vii, ia, voi
            self.assertIsInstance(val, da.Array)
        res = resampler.get_sample_from_neighbour_info(data)
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        # actual = res.values
        # expected = TODO
        # np.testing.assert_allclose(actual, expected)
