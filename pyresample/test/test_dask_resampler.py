#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""Tests for the DaskResampler."""

import unittest
import dask.array as da
import numpy as np
from pyresample.resampler import DaskResampler
from pyresample.geometry import AreaDefinition, SwathDefinition, IncompatibleAreas
import pytest


def dummy_resampler(data, source_area, destination_area):
    """Resample by filling an array with the sum of the data."""
    return np.full(destination_area.shape, data.sum())


class TestDaskResampler(unittest.TestCase):
    """Test case for the DaskResampler class."""

    def setUp(self):
        """Set up the test case."""
        self.input_data = da.arange(100*100).reshape((100, 100)).rechunk(30).astype(float)
        self.src_area = AreaDefinition('dst', 'dst area', None,
                                       {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                       100, 100,
                                       (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        self.src_swath = SwathDefinition(*self.src_area.get_lonlats(chunks=self.input_data.chunks))
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))
        self.dr = DaskResampler(self.src_area, self.dst_area, dummy_resampler)

    def test_resampling_generates_a_dask_array(self):
        """Test that resampling generates a dask array."""
        res = self.dr.resample(self.input_data)
        self.assertIsInstance(res, da.Array)

    def test_resampling_has_the_size_of_the_target_area(self):
        """Test that resampling generates an array of the right size."""
        res = self.dr.resample(self.input_data)
        assert res.shape == self.dst_area.shape

    def test_resampling_keeps_the_chunk_size(self):
        """Test that resampling keeps the chunk size from the input."""
        res = self.dr.resample(self.input_data)
        assert res.chunksize == self.input_data.chunksize

    def test_resampling_result_has_no_nans_when_fully_covered(self):
        """Test that resampling does not produce nans with full coverage."""
        res = self.dr.resample(self.input_data)
        assert np.isfinite(res).all()

    def test_resampling_result_name_is_unique(self):
        """Test that resampling generates unique dask array names."""
        res1 = self.dr.resample(self.input_data)
        input_data = da.ones((100, 100))
        res2 = self.dr.resample(input_data)
        assert res1.name != res2.name
        assert res1.name.startswith('dummy_resampler')

    def test_resampling_reduces_input_data(self):
        """Test that resampling reduces the input data."""
        res = self.dr.resample(self.input_data)
        assert res.max() < 49995000  # sum of all self.input_data

    def test_gradient_resampler(self):
        """Test the gradient resampler."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        res = dr.resample(self.input_data)
        assert np.nanmin(res - 8000) > 0

    def test_gradient_resampler_3d(self):
        """Test the gradient resampler with 3d data."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        input_data = self.input_data[np.newaxis, :, :]
        res = dr.resample(input_data)
        assert res.ndim == 3
        assert res.shape[0] == 1
        assert np.nanmin(res - 8000) > 0

    def test_gradient_resampler_3d_chunked(self):
        """Test gradient resampler in 3d with chunked data."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        input_data = self.input_data[np.newaxis, :, :].rechunk(20)
        res = dr.resample(input_data)
        assert res.ndim == 3
        assert res.shape[0] == 1
        assert np.nanmin(res - 8000) > 0


class TestDaskResamplerSlices(unittest.TestCase):
    """Test the get_slice function."""

    def setUp(self):
        """Set up the test case."""
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

    def test_source_area_covers_dest_area(self):
        """Test source area covers dest area."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))

        x_slice, y_slice = DaskResampler.get_slices(src_area, self.dst_area)
        assert x_slice.start > 0 and x_slice.stop < 100
        assert y_slice.start > 0 and y_slice.stop < 100

    def test_source_area_does_not_cover_dest_area_entirely(self):
        """Test source area does not cover dest area entirely."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 4440000.0, -5550000.0, -6660000.0))

        x_slice, y_slice = DaskResampler.get_slices(src_area, self.dst_area)
        assert x_slice.start > 0 and x_slice.stop < 100
        assert y_slice.start > 0 and y_slice.stop >= 100

    def test_source_area_does_not_cover_dest_area_at_all(self):
        """Test source area does not cover dest area at all."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  80, 100,
                                  (5550000.0, 3330000.0, -5550000.0, -5550000.0))

        with pytest.raises(IncompatibleAreas):
            DaskResampler.get_slices(src_area, self.dst_area)

    def test_dest_area_is_outside_source_area_domain(self):
        """Test dest area is outside the source area domain (nan coordinates)."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        dst_area = AreaDefinition('merc', 'merc', None,
                                  {'proj': 'merc', 'lon_0': 120.0,
                                   'lat_0': 0,
                                   'ellps': 'bessel'},
                                  102, 102,
                                  (-100000, -100000,
                                   100000, 100000))
        with pytest.raises(IncompatibleAreas):
            DaskResampler.get_slices(src_area, dst_area)

    def test_barely_touching_chunks_intersection(self):
        """Test that barely touching chunks generate slices on intersection."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        dst_area = AreaDefinition('moll', 'moll', None,
                                  {
                                      'ellps': 'WGS84',
                                      'lon_0': '0',
                                      'proj': 'moll',
                                      'units': 'm'
                                  },
                                  102, 102,
                                  (-18040095.6961, 4369712.0686,
                                   18040095.6961, 9020047.8481))
        x_slice, y_slice = DaskResampler.get_slices(src_area, dst_area)
        assert x_slice.start > 0 and x_slice.stop < 100
        assert y_slice.start > 0 and y_slice.stop >= 100
