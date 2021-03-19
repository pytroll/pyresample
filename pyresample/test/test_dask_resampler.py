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
from pyresample.geometry import AreaDefinition, SwathDefinition


def dummy_resampler(data, source_area, destination_area):
    """Resample by filling an array with the sum of the data."""
    return np.full(destination_area.shape, data.sum())


def chunk_copy_resampler(data, source_area, destination_area):
    """Resample by doing a copy of the input data."""
    after_y = destination_area.shape[0] - source_area.shape[0]
    after_x = destination_area.shape[1] - source_area.shape[1]
    return np.pad(data, ((0, after_y), (0, after_x)), 'constant', constant_values=np.nan)


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

    def test_resampling_follows_chunks(self):
        """Test that resampling follows the chunking."""
        dr = DaskResampler(self.src_area, self.src_area, chunk_copy_resampler)
        res = dr.resample(self.input_data)
        assert np.nanmax(np.abs(res - self.input_data)) < 210

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
