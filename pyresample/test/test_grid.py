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
"""Test grid interface."""
import unittest

import numpy as np

from pyresample import geometry, grid, utils


class Test(unittest.TestCase):
    """Test grid interface."""

    area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
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

    area_def2 = geometry.AreaDefinition('areaD2', 'Europe (3km, HRV, VTC)', 'areaD2',
                                        {'a': '6378144.0',
                                         'b': '6356759.0',
                                         'lat_0': '50.00',
                                         'lat_ts': '50.00',
                                         'lon_0': '8.00',
                                         'proj': 'stere'},
                                        5,
                                        5,
                                        [-1370912.72,
                                            -909968.64000000001,
                                            1029087.28,
                                            1490031.3600000001])

    msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
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
                                           5568742.4000000004]
                                       )

    def test_linesample(self):
        data = np.fromfunction(lambda y, x: y * x, (40, 40))
        rows = np.array([[1, 2], [3, 4]])
        cols = np.array([[25, 26], [27, 28]])
        res = grid.get_image_from_linesample(rows, cols, data)
        expected = np.array([[25., 52.], [81., 112.]])
        self.assertTrue(np.array_equal(res, expected), 'Linesample failed')

    def test_linesample_multi(self):
        data1 = np.fromfunction(lambda y, x: y * x, (40, 40))
        data2 = np.fromfunction(lambda y, x: 2 * y * x, (40, 40))
        data3 = np.fromfunction(lambda y, x: 3 * y * x, (40, 40))
        data = np.zeros((40, 40, 3))
        data[:, :, 0] = data1
        data[:, :, 1] = data2
        data[:, :, 2] = data3
        rows = np.array([[1, 2], [3, 4]])
        cols = np.array([[25, 26], [27, 28]])
        res = grid.get_image_from_linesample(rows, cols, data)
        expected = np.array([[[25., 50., 75.],
                              [52., 104., 156.]],
                             [[81., 162., 243.],
                              [112., 224., 336.]]])
        self.assertTrue(np.array_equal(res, expected), 'Linesample failed')

    def test_from_latlon(self):
        data = np.fromfunction(lambda y, x: y * x, (800, 800))
        lons = np.fromfunction(lambda y, x: x, (10, 10))
        lats = np.fromfunction(lambda y, x: 50 - (5.0 / 10) * y, (10, 10))
        source_def = self.area_def
        res = grid.get_image_from_lonlats(lons, lats, source_def, data)
        expected = np.array([[129276., 141032., 153370., 165804., 178334., 190575.,
                              202864., 214768., 226176., 238080.],
                             [133056., 146016., 158808., 171696., 184320., 196992.,
                              209712., 222480., 234840., 247715.],
                             [137026., 150150., 163370., 177215., 190629., 203756.,
                              217464., 230256., 243048., 256373.],
                             [140660., 154496., 168714., 182484., 196542., 210650.,
                              224257., 238464., 251712., 265512.],
                             [144480., 158484., 173148., 187912., 202776., 217358.,
                              231990., 246240., 259920., 274170.],
                             [147968., 163261., 178398., 193635., 208616., 223647.,
                              238728., 253859., 268584., 283898.],
                             [151638., 167121., 182704., 198990., 214775., 230280.,
                              246442., 261617., 276792., 292574.],
                             [154980., 171186., 187860., 204016., 220542., 237120.,
                              253125., 269806., 285456., 301732.],
                             [158500., 175536., 192038., 209280., 226626., 243697.,
                              260820., 277564., 293664., 310408.],
                             [161696., 179470., 197100., 214834., 232320., 250236.,
                              267448., 285090., 302328., 320229.]])
        self.assertTrue(
            np.array_equal(res, expected), 'Sampling from lat lon failed')

    def test_proj_coords(self):
        res = self.area_def2.get_proj_coords()
        cross_sum = res[0].sum() + res[1].sum()
        expected = 2977965.9999999963
        self.assertAlmostEqual(
            cross_sum, expected, msg='Calculation of proj coords failed')

    def test_latlons(self):
        res = self.area_def2.get_lonlats()
        cross_sum = res[0].sum() + res[1].sum()
        expected = 1440.8280578215431
        self.assertAlmostEqual(
            cross_sum, expected, msg='Calculation of lat lons failed')

    def test_latlons_mp(self):
        res = self.area_def2.get_lonlats(nprocs=2)
        cross_sum = res[0].sum() + res[1].sum()
        expected = 1440.8280578215431
        self.assertAlmostEqual(
            cross_sum, expected, msg='Calculation of lat lons failed')

    def test_resampled_image(self):
        data = np.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        target_def = self.area_def
        source_def = self.msg_area
        res = grid.get_resampled_image(
            target_def, source_def, data, segments=1)
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(
            cross_sum, expected, msg='Resampling of image failed')

    def test_resampled_image_masked(self):
        # Generate test image with masked elements
        data = np.ma.ones(self.msg_area.shape)
        data.mask = np.zeros(data.shape)
        data.mask[253:400, 1970:2211] = 1

        # Resample image using multiple segments
        target_def = self.area_def
        source_def = self.msg_area
        res = grid.get_resampled_image(
            target_def, source_def, data, segments=4, fill_value=None)

        # Make sure the mask has been preserved
        self.assertGreater(res.mask.sum(), 0,
                           msg='Resampling did not preserve the mask')

    def test_generate_linesample(self):
        data = np.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        row_indices, col_indices = utils.generate_quick_linesample_arrays(self.msg_area,
                                                                          self.area_def)
        res = data[row_indices, col_indices]
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(
            cross_sum, expected, msg='Generate linesample failed')
        self.assertFalse(row_indices.dtype != np.uint16 or col_indices.dtype != np.uint16,
                         'Generate linesample failed. Downcast to uint16 expected')

    def test_resampled_image_mp(self):
        data = np.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        target_def = self.area_def
        source_def = self.msg_area
        res = grid.get_resampled_image(
            target_def, source_def, data, nprocs=2, segments=1)
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(
            cross_sum, expected, msg='Resampling of image mp failed')

    def test_single_lonlat(self):
        lon, lat = self.area_def.get_lonlat(400, 400)
        self.assertAlmostEqual(
            lon, 5.5028467120975835, msg='Resampling of single lon failed')
        self.assertAlmostEqual(
            lat, 52.566998432390619, msg='Resampling of single lat failed')

    def test_proj4_string(self):
        """Test 'proj_str' property of AreaDefinition."""
        proj4_string = self.area_def.proj_str
        # different versions of PROJ/pyproj simplify parameters in
        # different ways. Just check for the minimum expected.
        param_strings = (
            "+a=6378144",
            "+lon_0=8",
            "+lat_0=50",
            "+proj=stere",
        )
        for param_str in param_strings:
            assert param_str in proj4_string
