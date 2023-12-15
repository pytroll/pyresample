#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2021 Pyresample Developers
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
"""Test image interface."""

import os
import unittest

import numpy

from pyresample import geometry, image, utils
from pyresample.test.utils import TEST_FILES_PATH


class Test(unittest.TestCase):
    """Test image interface."""

    area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)',
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

    msg_area = geometry.AreaDefinition('msg_full',
                                       'Full globe MSG image 0 degrees',
                                       'msg_full',
                                       {'a': '6378169.0',
                                        'b': '6356584.0',
                                        'h': '35785831.0',
                                        'lon_0': '0',
                                        'proj': 'geos'},
                                       3712,
                                       3712,
                                       [-5568742.4000000004,
                                        -5568742.4000000004,
                                        5568742.4000000004,
                                        5568742.4000000004])

    msg_area_resize = geometry.AreaDefinition('msg_full',
                                              'Full globe MSG image 0 degrees',
                                              'msg_full',
                                              {'a': '6378169.0',
                                               'b': '6356584.0',
                                               'h': '35785831.0',
                                               'lon_0': '0',
                                               'proj': 'geos'},
                                              928,
                                              928,
                                              [-5568742.4000000004,
                                               -5568742.4000000004,
                                               5568742.4000000004,
                                               5568742.4000000004])

    def test_image(self):
        data = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        msg_con = image.ImageContainerQuick(data, self.msg_area, segments=1)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(cross_sum, expected)

    def test_image_segments(self):
        data = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        msg_con = image.ImageContainerQuick(data, self.msg_area, segments=8)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(cross_sum, expected)

    def test_return_type(self):
        data = numpy.ones((3712, 3712)).astype('int')
        msg_con = image.ImageContainerQuick(data, self.msg_area, segments=1)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        self.assertTrue(data.dtype is res.dtype)

    def test_masked_image(self):
        data = numpy.zeros((3712, 3712))
        mask = numpy.zeros((3712, 3712))
        mask[:, 1865:] = 1
        data_masked = numpy.ma.array(data, mask=mask)
        msg_con = image.ImageContainerQuick(
            data_masked, self.msg_area, segments=1)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        resampled_mask = res.mask.astype('int')
        expected = numpy.fromfile(os.path.join(TEST_FILES_PATH, 'mask_grid.dat'),
                                  sep=' ').reshape((800, 800))
        self.assertTrue(numpy.array_equal(resampled_mask, expected))

    def test_masked_image_fill(self):
        data = numpy.zeros((3712, 3712))
        mask = numpy.zeros((3712, 3712))
        mask[:, 1865:] = 1
        data_masked = numpy.ma.array(data, mask=mask)
        msg_con = image.ImageContainerQuick(data_masked, self.msg_area,
                                            fill_value=None, segments=1)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        resampled_mask = res.mask.astype('int')
        expected = numpy.fromfile(os.path.join(TEST_FILES_PATH, 'mask_grid.dat'),
                                  sep=' ').reshape((800, 800))
        self.assertTrue(numpy.array_equal(resampled_mask, expected))

    def test_nearest_neighbour(self):
        data = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        msg_con = image.ImageContainerNearest(
            data, self.msg_area, 50000, segments=1)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 399936.70287099993
        self.assertAlmostEqual(cross_sum, expected)

    def test_nearest_resize(self):
        data = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        msg_con = image.ImageContainerNearest(
            data, self.msg_area, 50000, segments=1)
        area_con = msg_con.resample(self.msg_area_resize)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 2212023.0175830
        self.assertAlmostEqual(cross_sum, expected)

    def test_nearest_neighbour_multi(self):
        data1 = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        data2 = numpy.fromfunction(
            lambda y, x: y * x * 10 ** -6, (3712, 3712)) * 2
        data = numpy.dstack((data1, data2))
        msg_con = image.ImageContainerNearest(
            data, self.msg_area, 50000, segments=1)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum1 = res[:, :, 0].sum()
        expected1 = 399936.70287099993
        self.assertAlmostEqual(cross_sum1, expected1)

        cross_sum2 = res[:, :, 1].sum()
        expected2 = 399936.70287099993 * 2
        self.assertAlmostEqual(cross_sum2, expected2)

    def test_nearest_neighbour_multi_preproc(self):
        data1 = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        data2 = numpy.fromfunction(
            lambda y, x: y * x * 10 ** -6, (3712, 3712)) * 2
        data = numpy.dstack((data1, data2))
        msg_con = image.ImageContainer(data, self.msg_area)
        # area_con = msg_con.resample_area_nearest_neighbour(self.area_def,
        # 50000)
        row_indices, col_indices = \
            utils.generate_nearest_neighbour_linesample_arrays(self.msg_area,
                                                               self.area_def,
                                                               50000)
        res = msg_con.get_array_from_linesample(row_indices, col_indices)
        cross_sum1 = res[:, :, 0].sum()
        expected1 = 399936.70287099993
        self.assertAlmostEqual(cross_sum1, expected1)

        cross_sum2 = res[:, :, 1].sum()
        expected2 = 399936.70287099993 * 2
        self.assertAlmostEqual(cross_sum2, expected2)

    def test_nearest_swath(self):
        data = numpy.fromfunction(lambda y, x: y * x, (50, 10))
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        swath_con = image.ImageContainerNearest(
            data, swath_def, 50000, segments=1)
        area_con = swath_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_nearest_swath_segments(self):
        data = numpy.fromfunction(lambda y, x: y * x, (50, 10))
        data = numpy.dstack(3 * (data,))
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        swath_con = image.ImageContainerNearest(
            data, swath_def, 50000, segments=2)
        area_con = swath_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 3 * 15874591.0
        self.assertEqual(cross_sum, expected)

    def test_bilinear(self):
        data = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (928, 928))
        msg_con = image.ImageContainerBilinear(data, self.msg_area_resize,
                                               50000, segments=1,
                                               neighbours=8)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 24712.589910252744
        self.assertAlmostEqual(cross_sum, expected)

    def test_bilinear_multi(self):
        data1 = numpy.fromfunction(lambda y, x: y * x * 10 ** -6, (928, 928))
        data2 = numpy.fromfunction(lambda y, x: y * x * 10 ** -6,
                                   (928, 928)) * 2
        data = numpy.dstack((data1, data2))
        msg_con = image.ImageContainerBilinear(data, self.msg_area_resize,
                                               50000, segments=1,
                                               neighbours=8)
        area_con = msg_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum1 = res[:, :, 0].sum()
        expected1 = 24712.589910252744
        self.assertAlmostEqual(cross_sum1, expected1)
        cross_sum2 = res[:, :, 1].sum()
        expected2 = 24712.589910252744 * 2
        self.assertAlmostEqual(cross_sum2, expected2)

    def test_bilinear_swath(self):
        data = numpy.fromfunction(lambda y, x: y * x, (50, 10))
        lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
        lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        swath_con = image.ImageContainerBilinear(data, swath_def, 500000,
                                                 segments=1, neighbours=8)
        area_con = swath_con.resample(self.area_def)
        res = area_con.image_data
        cross_sum = res.sum()
        expected = 16852120.789503865
        self.assertAlmostEqual(cross_sum, expected, places=5)
