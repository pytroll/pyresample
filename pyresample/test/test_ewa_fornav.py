#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2019
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
"""Test EWA fornav module."""
import logging
import unittest

import numpy as np

LOG = logging.getLogger(__name__)


class TestFornav(unittest.TestCase):
    """Test fornav resampling functions."""

    def test_fornav_swath_larger(self):
        """Test that a swath larger than the output grid fills the entire grid."""
        from pyresample.ewa import _fornav
        swath_shape = (1600, 3200)
        data_type = np.float32
        # Create a fake row and cols array
        rows = np.empty(swath_shape, dtype=np.float32)
        rows[:] = np.linspace(-500, 2500, 1600)[:, None]
        cols = np.empty(swath_shape, dtype=np.float32)
        cols[:] = np.linspace(-2500, 1500, 3200)
        rows_per_scan = 16
        # Create a fake data swath
        data = np.ones(swath_shape, dtype=data_type)
        out = np.empty((1000, 1000), dtype=data_type)

        grid_points_covered = _fornav.fornav_wrapper(cols, rows, (data,), (out,),
                                                     np.nan, np.nan, rows_per_scan)
        one_grid_points_covered = grid_points_covered[0]
        # The swath was larger than the grid, all of the grid should have
        # been covered by swath pixels
        self.assertEqual(one_grid_points_covered, out.size,
                         msg="Not all grid pixels were filled")
        # The swath was all 1s so there shouldn't be any non-1 values in the
        # output except outside the swath
        self.assertTrue(((out == 1) | np.isnan(out)).all(),
                        msg="Unexpected interpolation values were returned")

    def test_fornav_swath_wide_input(self):
        """Test that a swath with large input pixels on the left edge of the output."""
        from pyresample.ewa import _fornav
        swath_shape = (400, 800)
        data_type = np.float32
        # Create a fake row and cols array
        rows = np.empty(swath_shape, dtype=np.float32)
        rows[:] = np.linspace(-500, 500, 400)[:, None]
        cols = np.empty(swath_shape, dtype=np.float32)
        cols[:] = np.linspace(-500, 500, 800) + 0.5
        rows_per_scan = 16
        # Create a fake data swath
        data = np.ones(swath_shape, dtype=data_type)
        out = np.empty((800, 1000), dtype=data_type)

        grid_points_covered = _fornav.fornav_wrapper(cols, rows, (data,), (out,),
                                                     np.nan, np.nan, rows_per_scan)
        one_grid_points_covered = grid_points_covered[0]
        # the upper-left 500x500 square should be filled with 1s at least
        assert 500 * 500 <= one_grid_points_covered <= 505 * 505
        np.testing.assert_allclose(out[:500, :500], 1)

    def test_fornav_swath_smaller(self):
        """Test that a swath smaller than the output grid is entirely used."""
        from pyresample.ewa import _fornav
        swath_shape = (1600, 3200)
        data_type = np.float32
        # Create a fake row and cols array
        rows = np.empty(swath_shape, dtype=np.float32)
        rows[:] = np.linspace(500, 800, 1600)[:, None]
        cols = np.empty(swath_shape, dtype=np.float32)
        cols[:] = np.linspace(200, 600, 3200)
        rows_per_scan = 16
        # Create a fake data swath
        data = np.ones(swath_shape, dtype=data_type)
        out = np.empty((1000, 1000), dtype=data_type)

        grid_points_covered = _fornav.fornav_wrapper(cols, rows, (data,), (out,),
                                                     np.nan, np.nan, rows_per_scan)
        one_grid_points_covered = grid_points_covered[0]
        # The swath was smaller than the grid, make sure its whole area
        # was covered (percentage of grid rows/cols to overall size)
        self.assertAlmostEqual(one_grid_points_covered / float(out.size), 0.12, 2,
                               msg="Not all input swath pixels were used")
        # The swath was all 1s so there shouldn't be any non-1 values in the
        # output except outside the swath
        self.assertTrue(((out == 1) | np.isnan(out)).all(),
                        msg="Unexpected interpolation values were returned")

    def test_fornav_swath_smaller_int8(self):
        """Test that a swath smaller than the output grid is entirely used."""
        from pyresample.ewa import _fornav
        swath_shape = (1600, 3200)
        data_type = np.int8
        # Create a fake row and cols array
        rows = np.empty(swath_shape, dtype=np.float32)
        rows[:] = np.linspace(500, 800, 1600)[:, None]
        cols = np.empty(swath_shape, dtype=np.float32)
        cols[:] = np.linspace(200, 600, 3200)
        rows_per_scan = 16
        # Create a fake data swath
        data = np.ones(swath_shape, dtype=data_type)
        out = np.empty((1000, 1000), dtype=data_type)

        grid_points_covered = _fornav.fornav_wrapper(cols, rows, (data,), (out,),
                                                     -128, -128, rows_per_scan)
        one_grid_points_covered = grid_points_covered[0]
        # The swath was smaller than the grid, make sure its whole area
        # was covered (percentage of grid rows/cols to overall size)
        self.assertAlmostEqual(one_grid_points_covered / float(out.size), 0.12, 2,
                               msg="Not all input swath pixels were used")
        # The swath was all 1s so there shouldn't be any non-1 values in the
        # output except outside the swath
        # import ipdb; ipdb.set_trace()
        self.assertTrue(((out == 1) | (out == -128)).all(),
                        msg="Unexpected interpolation values were returned")

    def test_fornav_swath_one_scan_geo_nans(self):
        """Test that a swath treated as one large scan with NaNs in geolocation still succeeds."""
        from pyresample.ewa import _fornav
        swath_shape = (1600, 3200)
        data_type = np.float32
        # Create a fake row and cols array
        rows = np.empty(swath_shape, dtype=np.float32)
        rows[:] = np.linspace(500, 800, 1600)[:, None]
        cols = np.empty(swath_shape, dtype=np.float32)
        cols[:] = np.linspace(200, 600, 3200)
        rows[:10, :] = np.nan
        cols[:10, :] = np.nan
        rows_per_scan = rows.shape[0]
        # Create a fake data swath
        data = np.ones(swath_shape, dtype=data_type)
        out = np.empty((1000, 1000), dtype=data_type)

        grid_points_covered = _fornav.fornav_wrapper(cols, rows, (data,), (out,),
                                                     np.nan, np.nan, rows_per_scan)
        one_grid_points_covered = grid_points_covered[0]
        # The swath was smaller than the grid, make sure its whole area
        # was covered (percentage of grid rows/cols to overall size)
        self.assertAlmostEqual(one_grid_points_covered / float(out.size), 0.12, 2,
                               msg="Not all input swath pixels were used")
        # The swath was all 1s so there shouldn't be any non-1 values in the
        # output except outside the swath
        self.assertTrue(((out == 1) | np.isnan(out)).all(),
                        msg="Unexpected interpolation values were returned")


class TestFornavWrapper(unittest.TestCase):
    """Test the function wrapping the lower-level fornav code."""

    def test_fornav_swath_larger_float32(self):
        """Test that a swath larger than the output grid fills the entire grid."""
        from pyresample.ewa import fornav
        swath_shape = (1600, 3200)
        data_type = np.float32
        # Create a fake row and cols array
        rows = np.empty(swath_shape, dtype=np.float32)
        rows[:] = np.linspace(-500, 2500, 1600)[:, None]
        cols = np.empty(swath_shape, dtype=np.float32)
        cols[:] = np.linspace(-2500, 1500, 3200)
        # Create a fake data swath
        data = np.ones(swath_shape, dtype=data_type)
        out = np.empty((1000, 1000), dtype=data_type)
        # area can be None because `out` is specified
        area = None

        grid_points_covered, out_res = fornav(cols, rows, area, data,
                                              rows_per_scan=16, out=out)
        self.assertIs(out, out_res)
        # The swath was larger than the grid, all of the grid should have
        # been covered by swath pixels
        self.assertEqual(grid_points_covered, out.size,
                         msg="Not all grid pixels were filled")
        # The swath was all 1s so there shouldn't be any non-1 values in the
        # output except outside the swath
        self.assertTrue(((out == 1) | np.isnan(out)).all(),
                        msg="Unexpected interpolation values were returned")
