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
import pytest

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


def _partial_coverage_swath(dtype):
    """Create an all-ones swath that covers only part of a 1000x1000 grid.

    The right side of the grid is left uncovered so the output is guaranteed
    to contain fill pixels as well as data pixels.
    """
    swath_shape = (1600, 3200)
    rows = np.empty(swath_shape, dtype=np.float32)
    rows[:] = np.linspace(-500, 2500, swath_shape[0])[:, None]
    cols = np.empty(swath_shape, dtype=np.float32)
    cols[:] = np.linspace(-2500, 500, swath_shape[1])
    data = np.ones(swath_shape, dtype=dtype)
    out = np.empty((1000, 1000), dtype=dtype)
    return cols, rows, data, out


@pytest.mark.parametrize(
    ("dtype", "fill", "expected_fill"),
    [
        (np.int8, 0, 0),
        (np.int8, None, 127),
        (np.float32, 0.0, 0.0),
        (np.float32, None, np.nan),
        (np.float64, -999.0, -999.0),
        (np.float64, np.nan, np.nan),
    ],
)
def test_fornav_fill_used_in_output(dtype, fill, expected_fill):
    """Test that the user-supplied or default fill value is written to uncovered pixels (#689)."""
    from pyresample.ewa import fornav
    cols, rows, data, out = _partial_coverage_swath(dtype)

    grid_points_covered, out_res = fornav(cols, rows, None, data,
                                          rows_per_scan=16, fill=fill, out=out)

    assert out is out_res
    assert grid_points_covered > 0
    # The swath was all 1s so every grid cell is either data or the fill value
    is_fill = np.isnan(out) if np.isnan(expected_fill) else out == expected_fill
    assert ((out == 1) | is_fill).all(), "Output should only contain data and fill values"
    assert is_fill.any(), "Uncovered pixels should contain the fill value"


@pytest.mark.parametrize(
    ("dtype", "input_fill", "output_fill"),
    [
        (np.float32, -999.0, np.nan),
        (np.float64, -999.0, np.nan),
        (np.int8, -100, -128),
    ],
)
def test_fornav_input_fill_differs_from_output_fill(dtype, input_fill, output_fill):
    """Test that input fill pixels are excluded when input and output fill differ."""
    from pyresample.ewa import _fornav
    swath_shape = (400, 800)
    rows_per_scan = 16
    rows = np.empty(swath_shape, dtype=np.float32)
    rows[:] = np.linspace(-50, 350, swath_shape[0])[:, None]
    cols = np.empty(swath_shape, dtype=np.float32)
    cols[:] = np.linspace(-100, 500, swath_shape[1])
    data = np.ones(swath_shape, dtype=dtype)
    data[100:120, 200:260] = input_fill

    out = np.empty((300, 300), dtype=dtype)
    _fornav.fornav_wrapper(cols, rows, (data,), (out,), input_fill, output_fill, rows_per_scan)

    # The input fill pixels must not be averaged into the output: every grid
    # cell is either the swath value (1) or the output fill
    is_fill = np.isnan(out) if np.isnan(output_fill) else out == output_fill
    assert ((out == 1) | is_fill).all(), "Input fill was used as valid data"
    assert is_fill.any(), "Expected some output fill cells"


@pytest.mark.parametrize("coord_dtype", [">f4", ">f8"])
def test_fornav_big_endian_coords(coord_dtype):
    """Test that big-endian cols/rows resample identically to native-order ones (#772)."""
    from pyresample.ewa import fornav
    big_dtype = np.dtype(coord_dtype)
    native_dtype = big_dtype.newbyteorder("=")
    swath_shape = (400, 800)
    rows = np.empty(swath_shape, dtype=native_dtype)
    rows[:] = np.linspace(-50, 350, swath_shape[0])[:, None]
    cols = np.empty(swath_shape, dtype=native_dtype)
    cols[:] = np.linspace(-100, 500, swath_shape[1])
    # Non-constant data so output equality pins the coordinate mapping,
    # not just grid coverage
    data = np.arange(swath_shape[0] * swath_shape[1], dtype=np.float32).reshape(swath_shape)
    out_shape = (300, 300)

    points_native, out_native = fornav(cols, rows, None, data,
                                        rows_per_scan=16,
                                        out=np.empty(out_shape, dtype=np.float32))
    points_big_cols, out_big_cols = fornav(cols.astype(big_dtype), rows, None, data,
                                            rows_per_scan=16,
                                            out=np.empty(out_shape, dtype=np.float32))
    points_big_rows, out_big_rows = fornav(cols, rows.astype(big_dtype), None, data,
                                            rows_per_scan=16,
                                            out=np.empty(out_shape, dtype=np.float32))
    points_big_both, out_big_both = fornav(cols.astype(big_dtype), rows.astype(big_dtype),
                                            None, data, rows_per_scan=16,
                                            out=np.empty(out_shape, dtype=np.float32))

    assert points_big_cols == points_native
    assert points_big_rows == points_native
    assert points_big_both == points_native
    np.testing.assert_array_equal(out_big_cols, out_native)
    np.testing.assert_array_equal(out_big_rows, out_native)
    np.testing.assert_array_equal(out_big_both, out_native)


def test_fornav_noncontiguous_coords():
    """Test that non-contiguous native-order cols/rows resample like contiguous ones (#772)."""
    from pyresample.ewa import fornav
    swath_shape = (400, 800)
    rows = np.empty(swath_shape, dtype=np.float64)
    rows[:] = np.linspace(-50, 350, swath_shape[0])[:, None]
    cols = np.empty(swath_shape, dtype=np.float64)
    cols[:] = np.linspace(-100, 500, swath_shape[1])
    # Non-constant data so output equality pins the coordinate mapping,
    # not just grid coverage
    data = np.arange(swath_shape[0] * swath_shape[1], dtype=np.float32).reshape(swath_shape)
    out_shape = (300, 300)

    points_native, out_native = fornav(cols, rows, None, data,
                                        rows_per_scan=16,
                                        out=np.empty(out_shape, dtype=np.float32))
    points_fortran, out_fortran = fornav(np.asfortranarray(cols), np.asfortranarray(rows),
                                          None, data, rows_per_scan=16,
                                          out=np.empty(out_shape, dtype=np.float32))

    assert points_fortran == points_native
    np.testing.assert_array_equal(out_fortran, out_native)


def test_native_contiguous_array_preserves_dtype_and_identity():
    """Test that the fornav coordinate normalization is a no-op for native C-contiguous arrays."""
    from pyresample.ewa.ewa import _native_contiguous_array
    for dtype in (np.float32, np.float64):
        arr = np.arange(6, dtype=dtype).reshape(2, 3)
        assert _native_contiguous_array(arr) is arr
        big_endian = arr.astype(arr.dtype.newbyteorder(">"))
        converted = _native_contiguous_array(big_endian)
        assert converted.dtype == arr.dtype
        assert converted.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(converted, arr)
        assert _native_contiguous_array(np.asfortranarray(arr)).flags["C_CONTIGUOUS"]
