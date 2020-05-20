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
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

        self.resampler = GradientSearchResampler(self.src_area, self.dst_area)
        self.swath_resampler = GradientSearchResampler(self.src_swath,
                                                       self.dst_area)

    def test_get_projection_coordinates_area_to_area(self):
        """Check that the coordinates are initialized, for area -> area."""
        assert self.resampler.prj is None
        self.resampler._get_projection_coordinates((10, 10))
        cdst_x = self.resampler.dst_x.compute()
        cdst_y = self.resampler.dst_y.compute()
        assert np.min(cdst_x) == -2022632.1675016289
        assert np.max(cdst_x) == 2196052.591296284
        assert np.min(cdst_y) == 3517933.413092212
        assert np.max(cdst_y) == 5387038.893400168
        assert self.resampler.use_input_coords
        assert self.resampler.prj is not None

    def test_get_projection_coordinates_swath_to_area(self):
        """Check that the coordinates are initialized, for swath -> area."""
        assert self.swath_resampler.prj is None
        self.swath_resampler._get_projection_coordinates((10, 10))
        cdst_x = self.swath_resampler.dst_x.compute()
        cdst_y = self.swath_resampler.dst_y.compute()
        assert np.min(cdst_x) == -2697103.29912692
        assert np.max(cdst_x) == 1358739.8381279823
        assert np.min(cdst_y) == -5550969.708939591
        assert np.max(cdst_y) == -1495126.5716846888
        assert self.swath_resampler.use_input_coords is False
        assert self.swath_resampler.prj is not None

    def test_get_gradients(self):
        """Test that coordinate gradients are computed correctly."""
        self.resampler._get_projection_coordinates((10, 10))
        assert self.resampler.src_gradient_xl is None
        self.resampler._get_gradients()
        assert self.resampler.src_gradient_xl.compute().max() == 0.0
        assert self.resampler.src_gradient_xp.compute().max() == -111000.0
        assert self.resampler.src_gradient_yl.compute().max() == 111000.0
        assert self.resampler.src_gradient_yp.compute().max() == 0.0

    def test_get_chunk_mappings(self):
        """Test that chunk overlap, and source and target slices are correct."""
        chunks = (10, 10)
        num_chunks = np.product(chunks)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        assert self.resampler.coverage_status is None
        self.resampler.get_chunk_mappings()
        # 8 source chunks overlap the target area
        covered_src_chunks = np.array([38, 39, 48, 49, 58, 59, 68, 69])
        res = np.where(self.resampler.coverage_status)[0]
        assert np.all(res == covered_src_chunks)
        # All *num_chunks* should have values in the lists
        assert len(self.resampler.coverage_status) == num_chunks
        assert len(self.resampler.src_slices) == num_chunks
        assert len(self.resampler.dst_slices) == num_chunks
        assert len(self.resampler.dst_mosaic_locations) == num_chunks
        # There's only one output chunk, and the covered source chunks
        # should have destination locations of (0, 0)
        res = np.array(self.resampler.dst_mosaic_locations)[covered_src_chunks]
        assert all([all(loc == (0, 0)) for loc in list(res)])

    def test_get_src_poly_area(self):
        """Test defining source chunk polygon for AreaDefinition."""
        chunks = (10, 10)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        poly = self.resampler._get_src_poly(0, 40, 0, 40)
        assert poly.area == 12365358458842.43

    def test_get_src_poly_swath(self):
        """Test defining source chunk polygon for SwathDefinition."""
        chunks = (10, 10)
        self.swath_resampler._get_projection_coordinates(chunks)
        self.swath_resampler._get_gradients()
        # Swath area defs can't be sliced, so False is returned
        poly = self.swath_resampler._get_src_poly(0, 40, 0, 40)
        assert poly is False

    @mock.patch('pyresample.gradient.get_polygon')
    def test_get_dst_poly(self, get_polygon):
        """Test defining destination chunk polygon."""
        chunks = (10, 10)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        # First call should make a call to get_polygon()
        self.resampler._get_dst_poly('idx1', 0, 10, 0, 10)
        assert get_polygon.call_count == 1
        assert 'idx1' in self.resampler.dst_polys
        # The second call to the same index should come from cache
        self.resampler._get_dst_poly('idx1', 0, 10, 0, 10)
        assert get_polygon.call_count == 1

        # Swath defs raise AttributeError, and False is returned
        get_polygon.side_effect = AttributeError
        self.resampler._get_dst_poly('idx2', 0, 10, 0, 10)
        assert self.resampler.dst_polys['idx2'] is False

    def test_filter_data(self):
        """Test filtering chunks that do not overlap."""
        chunks = (10, 10)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        self.resampler.get_chunk_mappings()

        # Basic filtering.  There should be 8 dask arrays that each
        # have a shape of (10, 10)
        res = self.resampler._filter_data(self.resampler.src_x)
        valid = [itm for itm in res if itm is not None]
        assert len(valid) == 8
        shapes = [arr.shape for arr in valid]
        for shp in shapes:
            assert shp == (10, 10)

        # Destination x/y coordinate array filtering.  Again, 8 dask
        # arrays each with shape (102, 102)
        res = self.resampler._filter_data(self.resampler.dst_x, is_src=False)
        valid = [itm for itm in res if itm is not None]
        assert len(valid) == 8
        shapes = [arr.shape for arr in valid]
        for shp in shapes:
            assert shp == (102, 102)

        # Add a dimension to the given dataset
        data = da.random.random(self.src_area.shape)
        res = self.resampler._filter_data(data, add_dim=True)
        valid = [itm for itm in res if itm is not None]
        assert len(valid) == 8
        shapes = [arr.shape for arr in valid]
        for shp in shapes:
            assert shp == (1, 10, 10)

    def test_resample_area_to_area_2d(self):
        """Resample area to area, 2d."""
        data = xr.DataArray(da.ones(self.src_area.shape, dtype=np.float64),
                            dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert np.allclose(res, 1)

    def test_resample_area_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.src_area.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        res = self.resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        assert np.allclose(res[0, :, :], 1.0)
        assert np.allclose(res[1, :, :], 2.0)
        assert np.allclose(res[2, :, :], 3.0)

    def test_resample_swath_to_area_2d(self):
        """Resample swath to area, 2d."""
        data = xr.DataArray(da.ones(self.src_swath.shape, dtype=np.float64),
                            dims=['y', 'x'])
        res = self.swath_resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert not np.all(np.isnan(res))


    def test_resample_swath_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.src_swath.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        res = self.swath_resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        assert np.nanmin(res[0, :, :]) == 1.0
        assert np.nanmax(res[0, :, :]) == 1.0
        assert np.nanmin(res[1, :, :]) == 2.0
        assert np.nanmax(res[1, :, :]) == 2.0
        assert np.nanmin(res[2, :, :]) == 3.0
        assert np.nanmax(res[2, :, :]) == 3.0

