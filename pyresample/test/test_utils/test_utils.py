#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015-2021 Pyresample developers
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
"""Test various utility functions."""

import os
import unittest
from timeit import timeit

import numpy as np
import pyproj
import pytest

from pyresample.test.utils import TEST_FILES_PATH, create_test_latitude, create_test_longitude
from pyresample.utils.row_appendable_array import RowAppendableArray


class TestPreprocessing(unittest.TestCase):
    """Tests for index generating functions."""

    def test_nearest_neighbor_area_area(self):
        from pyresample import geometry, utils
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)

        extents2 = [-1000, -1000, 1000. * 4000, 1000. * 4000]
        area_def2 = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                            proj_dict, 600, 700, extents2)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, area_def2, 12000.)

    def test_nearest_neighbor_area_grid(self):
        from pyresample import geometry, utils
        lon_arr = create_test_longitude(-94.9, -90.0, (50, 100), dtype=np.float64)
        lat_arr = create_test_latitude(25.1, 30.0, (50, 100), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)

        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, grid, 12000.)

    def test_nearest_neighbor_grid_area(self):
        from pyresample import geometry, utils
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 2500., 1000. * 2000.]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 40, 50, extents)

        lon_arr = create_test_longitude(-100.0, -60.0, (550, 500), dtype=np.float64)
        lat_arr = create_test_latitude(20.0, 45.0, (550, 500), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(grid, area_def, 12000.)

    def test_nearest_neighbor_grid_grid(self):
        from pyresample import geometry, utils
        lon_arr = create_test_longitude(-95.0, -85.0, (40, 50), dtype=np.float64)
        lat_arr = create_test_latitude(25.0, 35.0, (40, 50), dtype=np.float64)
        grid_dst = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)

        lon_arr = create_test_longitude(-100.0, -80.0, (400, 500), dtype=np.float64)
        lat_arr = create_test_latitude(20.0, 40.0, (400, 500), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(grid, grid_dst, 12000.)


def test_wrap_longitudes():
    # test that we indeed wrap to [-180:+180[
    from pyresample import utils
    step = 60
    lons = np.arange(-360, 360 + step, step)
    assert (lons.min() < -180) and (lons.max() >= 180) and (+180 in lons)
    wlons = utils.wrap_longitudes(lons)
    assert not ((wlons.min() < -180) or (wlons.max() >= 180) or (+180 in wlons))


def test_wrap_and_check():
    from pyresample import utils

    lons1 = np.arange(-135., +135, 50.)
    lats = np.ones_like(lons1) * 70.
    new_lons, new_lats = utils.check_and_wrap(lons1, lats)
    assert lats is new_lats
    np.testing.assert_allclose(lons1, new_lons)

    lons2 = np.where(lons1 < 0, lons1 + 360, lons1)
    new_lons, new_lats = utils.check_and_wrap(lons2, lats)
    assert lats is new_lats
    # after wrapping lons2 should look like lons1
    np.testing.assert_allclose(lons1, new_lons)

    lats2 = lats + 25.
    with pytest.raises(ValueError):
        utils.check_and_wrap(lons1, lats2)


@pytest.mark.skipif(pyproj.__proj_version__ == "9.3.0", reason="Bug in PROJ causes inequality in EPSG comparison")
def test_def2yaml_converter():
    import tempfile

    from pyresample import convert_def_to_yaml, parse_area_file
    def_file = os.path.join(TEST_FILES_PATH, 'areas.cfg')
    filehandle, yaml_file = tempfile.mkstemp()
    os.close(filehandle)
    try:
        convert_def_to_yaml(def_file, yaml_file)
        areas_new = set(parse_area_file(yaml_file))
        areas = parse_area_file(def_file)
        areas_old = set(areas)
        areas_new = {area.area_id: area for area in areas_new}
        areas_old = {area.area_id: area for area in areas_old}
        assert areas_new == areas_old
    finally:
        os.remove(yaml_file)


def test_check_slice_orientation():
    """Test that slicing fix is doing what it should."""
    from pyresample.utils import check_slice_orientation

    # Forward slicing should not be changed
    start, stop, step = 0, 10, None
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res is slice_in

    # Reverse slicing should not be changed if the step is negative
    start, stop, step = 10, 0, -1
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res is slice_in

    # Reverse slicing should be fixed if step is positive
    start, stop, step = 10, 0, 2
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res == slice(start, stop, -step)

    # Reverse slicing should be fixed if step is None
    start, stop, step = 10, 0, None
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res == slice(start, stop, -1)


class TestRowAppendableArray(unittest.TestCase):
    """Test appending numpy arrays to possible pre-allocated buffer."""

    def test_append_1d_arrays_and_trim_remaining_buffer(self):
        appendable = RowAppendableArray(7)
        appendable.append_row(np.zeros(3))
        appendable.append_row(np.ones(3))
        self.assertTrue(np.array_equal(appendable.to_array(), np.array([0, 0, 0, 1, 1, 1])))

    def test_append_rows_of_nd_arrays_and_trim_remaining_buffer(self):
        appendable = RowAppendableArray(7)
        appendable.append_row(np.zeros((3, 2)))
        appendable.append_row(np.ones((3, 2)))
        self.assertTrue(np.array_equal(appendable.to_array(), np.vstack([np.zeros((3, 2)), np.ones((3, 2))])))

    def test_append_more_1d_arrays_than_expected(self):
        appendable = RowAppendableArray(5)
        appendable.append_row(np.zeros(3))
        appendable.append_row(np.ones(3))
        self.assertTrue(np.array_equal(appendable.to_array(), np.array([0, 0, 0, 1, 1, 1])))

    def test_append_more_rows_of_nd_arrays_than_expected(self):
        appendable = RowAppendableArray(2)
        appendable.append_row(np.zeros((3, 2)))
        appendable.append_row(np.ones((3, 2)))
        self.assertTrue(np.array_equal(appendable.to_array(), np.vstack([np.zeros((3, 2)), np.ones((3, 2))])))

    def test_append_1d_arrays_pre_allocated_appendable_array(self):
        appendable = RowAppendableArray(6)
        appendable.append_row(np.zeros(3))
        appendable.append_row(np.ones(3))
        self.assertTrue(np.array_equal(appendable.to_array(), np.array([0, 0, 0, 1, 1, 1])))

    def test_append_rows_of_nd_arrays_to_pre_allocated_appendable_array(self):
        appendable = RowAppendableArray(6)
        appendable.append_row(np.zeros((3, 2)))
        appendable.append_row(np.ones((3, 2)))
        self.assertTrue(np.array_equal(appendable.to_array(), np.vstack([np.zeros((3, 2)), np.ones((3, 2))])))

    def test_pre_allocation_can_double_appending_performance(self):
        unallocated = RowAppendableArray(0)
        pre_allocated = RowAppendableArray(10000)

        unallocated_performance = timeit(lambda: unallocated.append_row(np.array([42])), number=10000)
        pre_allocated_performance = timeit(lambda: pre_allocated.append_row(np.array([42])), number=10000)
        self.assertGreater(unallocated_performance / pre_allocated_performance, 2)
