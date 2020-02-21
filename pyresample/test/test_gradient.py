#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019

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
"""Tests for the gradien search resampling."""

import unittest
from unittest import mock
from pyresample.geometry import AreaDefinition, SwathDefinition
import numpy as np
import dask.array as da
import xarray as xr

R = 6371000


class TestGradientResampler(unittest.TestCase):
    """Test case for the gradient resampling."""

    def setUp(self):
        """Set up the test case."""
        from pyresample.gradient import GradientSearchResampler
        self.src_area = AreaDefinition('dst', 'dst area', None,
                                       {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                       100, 100,
                                       (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        self.src_swath = SwathDefinition(*self.src_area.get_lonlats())
        self.dst_area = AreaDefinition('nrMET3km', 'nrMET3km', None,
                                       {'proj': 'eqc', 'lon_0': 0.0, 'a': R},
                                       360, 180,
                                       (-np.pi * R, -np.pi / 2 * R, np.pi * R, np.pi / 2 * R))

        self.resampler = GradientSearchResampler(self.src_area, self.dst_area)
        self.swath_resampler = GradientSearchResampler(self.src_swath, self.dst_area)

    @mock.patch('pyresample.gradient.parallel_gradient_search')
    def test_coords_initialization(self, pgsoic):
        """Check that the coordinates get initialized correctly."""
        data = xr.DataArray(da.ones((100, 100), dtype=np.float64), dims=['y', 'x'])
        pgsoic.return_value = da.ones(self.dst_area.shape)
        self.resampler.compute(data, meth='bil')
        cdst_x = self.resampler.dst_x.compute()
        cdst_y = self.resampler.dst_y.compute()
        assert(np.isinf(cdst_x[0, 0]))
        assert(np.isinf(cdst_y[0, 0]))
        assert(cdst_y[90, 180] == -55285.59156767167)
        assert(cdst_x[90, 180] == 55656.13605425304)
        assert(self.resampler.use_input_coords)
        pgsoic.assert_called_once_with(data.data[:, :],
                                       self.resampler.src_x,
                                       self.resampler.src_y,
                                       self.resampler.dst_x,
                                       self.resampler.dst_y,
                                       meth='bil')

    def test_resample_area_to_area_2d(self):
        """Resample area to area, 2d."""
        data = xr.DataArray(da.ones((100, 100), dtype=np.float64), dims=['y', 'x'])
        res = self.resampler.compute(data, meth='bil').compute(scheduler='single-threaded')
        assert(res.shape == self.dst_area.shape)
        assert(not np.all(np.isnan(res)))

    def test_resample_area_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, 100, 100), dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis], dims=['bands', 'y', 'x'])
        res = self.resampler.compute(data, meth='bil').compute(scheduler='single-threaded')
        assert(res.shape == (3, ) + self.dst_area.shape)
        assert(not np.all(np.isnan(res)))

    def test_resample_swath_to_area_2d(self):
        """Resample swath to area, 2d."""
        data = xr.DataArray(da.ones((100, 100), dtype=np.float64), dims=['y', 'x'])
        res = self.swath_resampler.compute(data, meth='bil').compute(scheduler='single-threaded')
        assert(res.shape == self.dst_area.shape)
        assert(not np.all(np.isnan(res)))

    def test_resample_swath_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, 100, 100), dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis], dims=['bands', 'y', 'x'])
        res = self.swath_resampler.compute(data, meth='bil').compute(scheduler='single-threaded')
        assert(res.shape == (3, ) + self.dst_area.shape)
        assert(not np.all(np.isnan(res)))


class TestBlockFunctions(unittest.TestCase):
    """Test case for help functions."""

    def test_reshape_arrays(self):
        """Test the chunk stacking."""
        from pyresample.gradient import reshape_arrays_in_stacked_chunks
        data = da.ones((100, 100), chunks=(25, 50))
        res = reshape_arrays_in_stacked_chunks((data, ), data.chunks)[0]
        assert(res.shape == (25, 50, 8))

    def test_reshape_array_3d(self):
        """Test the chunk stacking on 3d arrays."""
        from pyresample.gradient import reshape_to_stacked_3d
        data = da.arange(432).reshape((3, 12, 12)).rechunk((3, 4, 6))
        res = reshape_to_stacked_3d(data)
        assert(res.shape == (3, 4, 6, 6))

    def test_split(self):
        """Test splitting the arrays."""
        from pyresample.gradient import split
        data = da.arange(180).reshape((3, 6, 10)).rechunk((3, 3, 5))
        res = split(data, 2, 1)
        assert(len(res) == 2)
        assert(res[0].shape == (3, 3, 10))

        res = split(data, 2, -1)
        assert(len(res) == 2)
        assert(res[0].shape == (3, 6, 5))

    @mock.patch('pyresample.gradient.da.blockwise')
    def test_parallel_resampling_no_blockwise(self, blockwise):
        """Test the parallel resampling until the blockwise call."""
        from pyresample.gradient import parallel_gradient_search as pgs
        data = da.ones((100, 100), chunks=(25, 50))
        src_x = da.ones((100, 100), chunks=(25, 50))
        src_y = da.ones((100, 100), chunks=(25, 50))
        dst_x = da.ones((180, 360), chunks=(90, 90))
        dst_y = da.ones((180, 360), chunks=(90, 90))

        def fake_blockwise(*args, **kwargs):
            return args[2]

        blockwise.side_effect = fake_blockwise
        res = pgs(data, src_x, src_y, dst_x, dst_y)
        assert(res.shape == (25, 50))

    @mock.patch('pyresample.gradient._gradient_resample_data')
    def test_parallel_search_no_blocks(self, grd):
        """Test the parallel resampling until the blocked calls."""
        from pyresample.gradient import parallel_gradient_search as pgs
        data = da.ones((100, 100), chunks=(25, 50))
        src_x = da.ones((100, 100), chunks=(25, 50))
        src_y = da.ones((100, 100), chunks=(25, 50))
        dst_x = da.ones((180, 360), chunks=(90, 90))
        dst_y = da.ones((180, 360), chunks=(90, 90))

        def fake_gradient_resample_data(*args, **kwargs):
            assert(kwargs['method'] == 'bilinear')
            return args[7][np.newaxis, :, :, np.newaxis]
        grd.side_effect = fake_gradient_resample_data
        res = pgs(data, src_x, src_y, dst_x, dst_y)
        res = res.compute(scheduler='single-threaded')
        assert(res.shape == (180, 360))

    @mock.patch('pyresample.gradient.one_step_gradient_search')
    def test_parallel_search_no_cython(self, osgs):
        """Test the parallel resampling until the cython calls."""
        from pyresample.gradient import parallel_gradient_search as pgs
        data = da.ones((100, 100), chunks=(25, 50))
        src_x = da.ones((100, 100), chunks=(25, 50))
        src_y = da.ones((100, 100), chunks=(25, 50))
        dst_x = da.ones((180, 360), chunks=(90, 90))
        dst_y = da.ones((180, 360), chunks=(90, 90))

        def fake_gradient_resample_data(*args, **kwargs):
            assert(kwargs['method'] == 'bilinear')
            assert(args[0].shape == (1, 25, 50))
            for arg in args[1:7]:
                assert(arg.shape == (25, 50))
            for arg in args[7:]:
                assert(arg.shape == (90, 90))
            return args[7][np.newaxis, :, :]
        osgs.side_effect = fake_gradient_resample_data
        res = pgs(data, src_x, src_y, dst_x, dst_y)
        res = res.compute(scheduler='single-threaded')
        assert(res.shape == (180, 360))

        data = da.ones((3, 100, 100), chunks=(3, 25, 50))

        def fake_gradient_resample_data(*args, **kwargs):
            assert(kwargs['method'] == 'bilinear')
            assert(args[0].shape == (3, 25, 50))
            for arg in args[1:7]:
                assert(arg.shape == (25, 50))
            for arg in args[7:]:
                assert(arg.shape == (90, 90))
            return args[7][np.newaxis, :, :]
