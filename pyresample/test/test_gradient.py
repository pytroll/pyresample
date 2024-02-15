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
import warnings
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from pyresample.area_config import create_area_def
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.gradient import ResampleBlocksGradientSearchResampler


class TestOGradientResampler:
    """Test case for the gradient resampling."""

    def setup_method(self):
        """Set up the test case."""
        from pyresample.gradient import StackingGradientSearchResampler
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
        self.dst_swath = SwathDefinition(*self.dst_area.get_lonlats())

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*which is still EXPERIMENTAL.*", category=UserWarning)
            self.resampler = StackingGradientSearchResampler(self.src_area, self.dst_area)
            self.swath_resampler = StackingGradientSearchResampler(self.src_swath, self.dst_area)
            self.area_to_swath_resampler = StackingGradientSearchResampler(self.src_area, self.dst_swath)

    def test_get_projection_coordinates_area_to_area(self):
        """Check that the coordinates are initialized, for area -> area."""
        assert self.resampler.prj is None
        self.resampler._get_projection_coordinates((10, 10))
        cdst_x = self.resampler.dst_x.compute()
        cdst_y = self.resampler.dst_y.compute()
        assert np.allclose(np.min(cdst_x), -2022632.1675016289)
        assert np.allclose(np.max(cdst_x), 2196052.591296284)
        assert np.allclose(np.min(cdst_y), 3517933.413092212)
        assert np.allclose(np.max(cdst_y), 5387038.893400168)
        assert self.resampler.use_input_coords
        assert self.resampler.prj is not None

    def test_get_projection_coordinates_swath_to_area(self):
        """Check that the coordinates are initialized, for swath -> area."""
        assert self.swath_resampler.prj is None
        self.swath_resampler._get_projection_coordinates((10, 10))
        cdst_x = self.swath_resampler.dst_x.compute()
        cdst_y = self.swath_resampler.dst_y.compute()
        assert np.allclose(np.min(cdst_x), -2697103.29912692)
        assert np.allclose(np.max(cdst_x), 1358739.8381279823)
        assert np.allclose(np.min(cdst_y), -5550969.708939591)
        assert np.allclose(np.max(cdst_y), -1495126.5716846888)
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
        num_chunks = np.prod(chunks)
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
        assert np.allclose(poly.area, 12365358458842.43)

    def test_get_src_poly_swath(self):
        """Test defining source chunk polygon for SwathDefinition."""
        chunks = (10, 10)
        self.swath_resampler._get_projection_coordinates(chunks)
        self.swath_resampler._get_gradients()
        # SwathDefinition can't be sliced, so False is returned
        poly = self.swath_resampler._get_src_poly(0, 40, 0, 40)
        assert poly is False

    @mock.patch('pyresample.gradient.get_polygon')
    def test_get_dst_poly_area(self, get_polygon):
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

    def test_get_dst_poly_swath(self):
        """Test defining dst chunk polygon for SwathDefinition."""
        chunks = (10, 10)
        self.area_to_swath_resampler._get_projection_coordinates(chunks)
        self.area_to_swath_resampler._get_gradients()
        # SwathDefinition can't be sliced, so False is returned
        self.area_to_swath_resampler._get_dst_poly('idx2', 0, 10, 0, 10)
        assert self.area_to_swath_resampler.dst_polys['idx2'] is False

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

        # 1D and 3+D should raise NotImplementedError
        data = da.random.random((3,))
        try:
            res = self.resampler._filter_data(data, add_dim=True)
            raise IndexError
        except NotImplementedError:
            pass
        data = da.random.random((3, 3, 3, 3))
        try:
            res = self.resampler._filter_data(data, add_dim=True)
            raise IndexError
        except NotImplementedError:
            pass

    def test_resample_area_to_area_2d(self):
        """Resample area to area, 2d."""
        data = xr.DataArray(da.ones(self.src_area.shape, dtype=np.float64),
                            dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert np.allclose(res, 1)

    def test_resample_area_to_area_2d_fill_value(self):
        """Resample area to area, 2d, use fill value."""
        data = xr.DataArray(da.full(self.src_area.shape, np.nan,
                                    dtype=np.float64), dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bil',
            fill_value=2.0).compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert np.allclose(res, 2.0)

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

    def test_resample_area_to_area_3d_single_channel(self):
        """Resample area to area, 3d with only a single band."""
        data = xr.DataArray(da.ones((1, ) + self.src_area.shape,
                                    dtype=np.float64),
                            dims=['bands', 'y', 'x'])
        res = self.resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == (1, ) + self.dst_area.shape
        assert np.allclose(res[0, :, :], 1.0)

    def test_resample_swath_to_area_2d(self):
        """Resample swath to area, 2d."""
        data = xr.DataArray(da.ones(self.src_swath.shape, dtype=np.float64),
                            dims=['y', 'x'])
        with np.errstate(invalid="ignore"):  # 'inf' space pixels cause runtime warnings
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
        with np.errstate(invalid="ignore"):  # 'inf' space pixels cause runtime warnings
            res = self.swath_resampler.compute(
                data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        for i in range(res.shape[0]):
            arr = np.ravel(res[i, :, :])
            assert np.allclose(arr[np.isfinite(arr)], float(i + 1))


class TestRBGradientSearchResamplerArea2Area:
    """Test RBGradientSearchResampler for the Area to Area case."""

    def setup_method(self):
        """Set up the test case."""
        self.src_area = AreaDefinition('src', 'src area', None,
                                       {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                       100, 100,
                                       (5550000.0, 5550000.0, -5550000.0, -5550000.0))

        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

        self.resampler = ResampleBlocksGradientSearchResampler(self.src_area, self.dst_area)

    def test_precompute_generates_indices(self):
        self.resampler.precompute()
        assert self.resampler.indices_xy.shape == (2, ) + self.dst_area.shape

    def test_resampler_accepts_only_dataarrays_if_not_2d(self):
        data = da.ones(self.src_area.shape + (1,), dtype=np.float64, chunks=40)
        self.resampler.precompute()
        with pytest.raises(TypeError):
            self.resampler.compute(data, method='bilinear').compute(scheduler='single-threaded')

    def test_resampler_returns_the_right_shape(self):
        data = xr.DataArray(da.ones((3, ) + self.src_area.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear')
        assert res.shape == (3, ) + self.dst_area.shape

    def test_resampler_returns_a_dataarray(self):
        data = self.create_2d_src_data()
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert isinstance(res, xr.DataArray)

    def test_resampler_returns_a_dataarray_with_correct_area_attribute(self):
        data = self.create_2d_src_data()
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert res.attrs["area"] == self.dst_area

    def test_resampler_returns_a_dataarray_with_input_attributes(self):
        attrs = {"sky": "blue", "grass": "green"}
        data = self.create_2d_src_data(attrs)
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        new_attrs = res.attrs.copy()
        new_attrs.pop("area")
        assert new_attrs == attrs

    def create_2d_src_data(self, attrs=None):
        data = xr.DataArray(da.ones(self.src_area.shape, dtype=np.float64),
                            dims=['y', 'x'], attrs=attrs)
        return data

    def test_resampler_returns_a_dataarray_with_input_dims(self):
        attrs = {"sky": "blue", "grass": "green"}
        data = self.create_2d_src_data(attrs)
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')

        assert res.dims == data.dims

    def test_resampler_returns_a_dataarray_with_input_coords(self):
        attrs = {"sky": "blue", "grass": "green"}
        data = self.create_3d_rgb_src_data(attrs)
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')

        assert all(res.coords["bands"] == data.coords["bands"])

    def create_3d_rgb_src_data(self, attrs=None):
        data = xr.DataArray(da.ones((3,) + self.src_area.shape, dtype=np.float64),
                            dims=["bands", 'y', 'x'], attrs=attrs,
                            coords={"bands": ["R", "G", "B"]})
        return data

    def test_resampler_returns_a_dataarray_with_correct_xy_coords(self):
        data = self.create_3d_rgb_src_data()
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')

        assert all(res.coords["bands"] == data.coords["bands"])
        assert "x" in res.coords
        assert "y" in res.coords

    def test_resampler_can_take_random_dim_order(self):
        data = self.create_3d_rgb_src_data().transpose("x", "bands", "y")
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')

        assert res.dims == data.dims

    def test_resampler_accepts_only_bilinear_or_nn(self):
        data = da.ones(self.src_area.shape, dtype=np.float64, chunks=40)
        self.resampler.precompute()
        with pytest.raises(ValueError):
            self.resampler.compute(data, method='bilinear_neighbour').compute(scheduler='single-threaded')

    def test_resample_area_to_area_2d(self):
        """Resample area to area, 2d."""
        data = self.create_2d_src_data()
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        np.testing.assert_allclose(res, 1)

    def test_resample_area_to_area_2d_fill_value(self):
        """Resample area to area, 2d, use fill value."""
        data = self.create_2d_src_data()
        dst_area = AreaDefinition('outside', 'outside', None,
                                  {'proj': 'stere', 'lon_0': 180.0,
                                   'lat_0': 90.0, 'lat_ts': 60.0,
                                   'ellps': 'bessel'},
                                  102, 102,
                                  (-2717181.7304994687, -5571048.14031214,
                                   1378818.2695005313, -1475048.1403121399))
        self.resampler.target_geo_def = dst_area
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear',
            fill_value=2.0).compute(scheduler='single-threaded')
        assert res.shape == dst_area.shape
        np.testing.assert_allclose(res, 2.0)

    def test_resample_area_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.src_area.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        assert np.allclose(res[0, :, :], 1.0)
        assert np.allclose(res[1, :, :], 2.0)
        assert np.allclose(res[2, :, :], 3.0)

    def test_resample_area_to_area_does_not_flip_the_result(self):
        """Resample area to area, check that x and y aren't flipped."""
        data = xr.DataArray(da.arange(np.prod(self.src_area.shape), dtype=np.float64).reshape(self.src_area.shape),
                            dims=['y', 'x'])
        dst_area = create_area_def("epsg3035", "EPSG:3035", 10, 10,
                                   (2426378.0132, 1528101.2618,
                                    6293974.6215, 5446513.5222))

        self.resampler.target_geo_def = dst_area
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear',
            fill_value=2.0).compute(scheduler='single-threaded').values
        corners = (res[0, 0], res[0, -1], res[-1, -1], res[-1, 0])
        expected_corners = (9660.560479802409, 9548.105823664819, 8183.903753, 8285.489720486787)
        np.testing.assert_allclose(corners, expected_corners)
        assert res.shape == dst_area.shape

    def test_resample_area_to_area_bilinear(self):
        """Resample area to area and check result values for bilinear."""
        data = xr.DataArray(da.arange(np.prod(self.src_area.shape), dtype=np.float64).reshape(self.src_area.shape),
                            dims=['y', 'x'])
        dst_area = create_area_def("epsg3035", "EPSG:3035", 5, 5,
                                   (2426378.0132, 1528101.2618,
                                    6293974.6215, 5446513.5222))

        expected_resampled_data = [[9657.73659888, 9736.06994061, 9744.63765978, 9684.31222874, 9556.53097857],
                                   [9473.47091892, 9551.45304151, 9559.75772548, 9499.27398623, 9371.5132557],
                                   [9207.81468378, 9285.56859415, 9293.95955182, 9233.88183094, 9106.9076482],
                                   [8861.02653887, 8938.6220088, 8947.46145892, 8888.42994376, 8763.14315424],
                                   [8434.30749791, 8511.74525856, 8521.39324998, 8464.10968423, 8341.53220395]]

        self.resampler.target_geo_def = dst_area
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='bilinear',
            fill_value=2.0).compute(scheduler='single-threaded').values
        np.testing.assert_allclose(res, expected_resampled_data)
        assert res.shape == dst_area.shape

    def test_resample_area_to_area_nn(self):
        """Resample area to area and check result values for nn."""
        data = xr.DataArray(da.arange(np.prod(self.src_area.shape), dtype=np.float64).reshape(self.src_area.shape),
                            dims=['y', 'x'])
        dst_area = create_area_def("epsg3035", "EPSG:3035", 5, 5,
                                   (2426378.0132, 1528101.2618,
                                    6293974.6215, 5446513.5222))

        expected_resampled_data = [[9658., 9752., 9746., 9640., 9534.],
                                   [9457., 9551., 9545., 9539., 9333.],
                                   [9257., 9250., 9344., 9238., 9132.],
                                   [8856., 8949., 8943., 8936., 8730.],
                                   [8455., 8548., 8542., 8435., 8329.]]

        self.resampler.target_geo_def = dst_area
        self.resampler.precompute()
        res = self.resampler.compute(
            data, method='nn',
            fill_value=2.0).compute(scheduler='single-threaded').values
        np.testing.assert_allclose(res, expected_resampled_data)
        assert res.shape == dst_area.shape


class TestRBGradientSearchResamplerArea2Swath:
    """Test RBGradientSearchResampler for the Swath to Area case."""

    def setup_method(self):
        """Set up the test case."""
        chunks = 20

        self.src_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

        self.dst_area = AreaDefinition(
            'omerc_otf',
            'On-the-fly omerc area',
            None,
            {'alpha': '8.99811271718795',
             'ellps': 'sphere',
             'gamma': '0',
             'k': '1',
             'lat_0': '0',
             'lonc': '13.8096029486222',
             'proj': 'omerc',
             'units': 'm'},
            50, 100,
            (-1461111.3603, 3440088.0459, 1534864.0322, 9598335.0457)
        )

        self.lons, self.lats = self.dst_area.get_lonlats(chunks=chunks)
        xrlons = xr.DataArray(self.lons.persist())
        xrlats = xr.DataArray(self.lats.persist())
        self.dst_swath = SwathDefinition(xrlons, xrlats)

    def test_resampling_to_swath_is_not_implemented(self):
        """Test that resampling to swath is not working yet."""
        from pyresample.gradient import ResampleBlocksGradientSearchResampler

        with pytest.raises(NotImplementedError):
            ResampleBlocksGradientSearchResampler(self.src_area,
                                                  self.dst_swath)


class TestEnsureDataArray(unittest.TestCase):
    """Test the ensure_data_array decorator."""

    def test_decorator_converts_2d_array_to_dataarrays_if_needed(self):
        """Test that the decorator converts numpy or dask 2d arrays to dataarrays."""
        from pyresample.gradient import ensure_data_array
        data = da.ones((10, 10), dtype=np.float64, chunks=40)

        def fake_compute(arg1, data):
            assert isinstance(data, xr.DataArray)
            return data

        decorated = ensure_data_array(fake_compute)
        decorated('bla', data)

    def test_decorator_raises_exception_when_3d_array_is_passed(self):
        """Test that the decorator raises and exception when a 3d array is passed."""
        from pyresample.gradient import ensure_data_array
        data = da.ones((10, 10, 10), dtype=np.float64, chunks=40)

        def fake_compute(arg1, data):
            assert isinstance(data, xr.DataArray)
            return data

        decorated = ensure_data_array(fake_compute)
        with pytest.raises(TypeError):
            decorated('bla', data)

    def test_decorator_transposes_input_array(self):
        """Test that the decorator transposes dimensions to put y and x last."""
        from pyresample.gradient import ensure_data_array
        data = xr.DataArray(da.ones((10, 10, 10), dtype=np.float64, chunks=40),
                            dims=["x", "bands", "y"])

        def fake_compute(arg1, data):
            assert data.dims == ("bands", "y", "x")
            return data

        decorated = ensure_data_array(fake_compute)
        decorated('bla', data)


def test_check_overlap():
    """Test overlap check returning correct results."""
    from shapely.geometry import Polygon

    from pyresample.gradient import check_overlap

    # If either of the polygons is False, True is returned
    assert check_overlap(False, 3) is True
    assert check_overlap('eggs', False) is True
    assert check_overlap(False, False) is True

    # If either the polygons is None, False is returned
    assert check_overlap(None, 'bacon') is False
    assert check_overlap('spam', None) is False
    assert check_overlap(None, None) is False

    # If the polygons overlap, True is returned
    poly1 = Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))
    poly2 = Polygon(((-1, -1), (-1, 1), (1, 1), (1, -1)))
    assert check_overlap(poly1, poly2) is True

    # If the polygons do not overlap, False is returned
    poly2 = Polygon(((5, 5), (6, 5), (6, 6), (5, 6)))
    assert check_overlap(poly1, poly2) is False


def test_get_border_lonlats_geos():
    """Test that correct methods are called in get_border_lonlats() with geos inputs."""
    from pyresample.gradient import get_border_lonlats
    geo_def = AreaDefinition("", "", "",
                             "+proj=geos +h=1234567", 2, 2, [1, 2, 3, 4])
    with mock.patch("pyresample.gradient.get_geostationary_bounding_box_in_lonlats") as get_geostationary_bounding_box:
        get_geostationary_bounding_box.return_value = 1, 2
        res = get_border_lonlats(geo_def)
    assert res == (1, 2)
    get_geostationary_bounding_box.assert_called_with(geo_def, 3600)


def test_get_border_lonlats():
    """Test that correct methods are called in get_border_lonlats()."""
    from pyresample.boundary import SimpleBoundary
    from pyresample.gradient import get_border_lonlats
    lon_sides = SimpleBoundary(side1=np.array([1]), side2=np.array([2]),
                               side3=np.array([3]), side4=np.array([4]))
    lat_sides = SimpleBoundary(side1=np.array([1]), side2=np.array([2]),
                               side3=np.array([3]), side4=np.array([4]))
    geo_def = AreaDefinition("", "", "",
                             "+proj=lcc +lat_1=25 +lat_2=25", 2, 2, [1, 2, 3, 4])
    with mock.patch.object(geo_def, "get_boundary_lonlats") as get_boundary_lonlats:
        get_boundary_lonlats.return_value = lon_sides, lat_sides
        lon_b, lat_b = get_border_lonlats(geo_def)
    assert np.all(lon_b == np.array([1, 2, 3, 4]))
    assert np.all(lat_b == np.array([1, 2, 3, 4]))


@mock.patch('pyresample.gradient.Polygon')
@mock.patch('pyresample.gradient.get_border_lonlats')
def test_get_polygon(get_border_lonlats, Polygon):
    """Test polygon creation."""
    from pyresample.gradient import get_polygon

    # Valid polygon
    get_border_lonlats.return_value = (1, 2)
    geo_def = mock.MagicMock()
    prj = mock.MagicMock()
    x_borders = [0, 0, 1, 1]
    y_borders = [0, 1, 1, 0]
    boundary = [(0, 0), (0, 1), (1, 1), (1, 0)]
    prj.return_value = (x_borders, y_borders)
    poly = mock.MagicMock(area=2.0)
    Polygon.return_value = poly
    res = get_polygon(prj, geo_def)
    get_border_lonlats.assert_called_with(geo_def)
    prj.assert_called_with(1, 2)
    Polygon.assert_called_with(boundary)
    assert res is poly

    # Some border points are invalid, those should have been removed
    x_borders = [np.inf, 0, 0, 0, 1, np.nan, 2]
    y_borders = [-1, 0, np.nan, 1, 1, np.nan, -1]
    boundary = [(0, 0), (0, 1), (1, 1), (2, -1)]
    prj.return_value = (x_borders, y_borders)
    res = get_polygon(prj, geo_def)
    Polygon.assert_called_with(boundary)
    assert res is poly

    # Polygon area is NaN
    poly.area = np.nan
    res = get_polygon(prj, geo_def)
    assert res is None

    # Polygon area is 0.0
    poly.area = 0.0
    res = get_polygon(prj, geo_def)
    assert res is None


@mock.patch('pyresample.gradient.one_step_gradient_search')
def test_gradient_resample_data(one_step_gradient_search):
    """Test that one_step_gradient_search() is called with proper array shapes."""
    from pyresample.gradient import _gradient_resample_data

    ndim_3 = np.zeros((3, 3, 4))
    ndim_2a = np.zeros((3, 4))
    ndim_2b = np.zeros((8, 10))

    # One of the source arrays has wrong shape
    with pytest.raises(ValueError):
        _ = _gradient_resample_data(ndim_3, ndim_2a, ndim_2b, ndim_2a, ndim_2a,
                                    ndim_2a, ndim_2a, ndim_2b, ndim_2b)

    one_step_gradient_search.assert_not_called()

    # Data array has wrong shape
    with pytest.raises(ValueError):
        _ = _gradient_resample_data(ndim_2a, ndim_2a, ndim_2a, ndim_2a, ndim_2a,
                                    ndim_2a, ndim_2a, ndim_2b, ndim_2b)

    one_step_gradient_search.assert_not_called()

    # The destination x and y arrays have different shapes
    with pytest.raises(ValueError):
        _ = _gradient_resample_data(ndim_3, ndim_2a, ndim_2a, ndim_2a, ndim_2a,
                                    ndim_2a, ndim_2a, ndim_2b, ndim_2a)

    one_step_gradient_search.assert_not_called()

    # Correct shapes are given
    _ = _gradient_resample_data(ndim_3, ndim_2a, ndim_2a, ndim_2a, ndim_2a,
                                ndim_2a, ndim_2a, ndim_2b, ndim_2b)
    one_step_gradient_search.assert_called_once()


@mock.patch('pyresample.gradient.dask.delayed')
@mock.patch('pyresample.gradient._concatenate_chunks')
@mock.patch('pyresample.gradient.da')
def test_parallel_gradient_search(dask_da, _concatenate_chunks, delayed):
    """Test calling parallel_gradient_search()."""
    from pyresample.gradient import parallel_gradient_search

    def mock_cc(chunks):
        """Return the input."""
        return chunks

    _concatenate_chunks.side_effect = mock_cc

    # Mismatch in number of bands raises ValueError
    data = [np.zeros((1, 5, 5)), np.zeros((2, 5, 5))]
    try:
        parallel_gradient_search(data, None, None, None, None,
                                 None, None, None, None, None, None)
        raise
    except ValueError:
        pass

    data = [np.zeros((1, 5, 4)), np.ones((1, 5, 4)), None, None]
    src_x, src_y = [1, 2, 3, 4], [4, 5, 6, 4]
    # dst_x is used to check the target area shape, so needs "valid"
    # data.  The last values shouldn't matter as data[-2:] are None
    # and should be skipped.
    dst_x = [np.zeros((5, 5)), np.zeros((5, 5)), 'foo', 'bar']
    dst_y = [1, 2, 3, 4]
    src_gradient_xl, src_gradient_xp = [1, 2, None, None], [1, 2, None, None]
    src_gradient_yl, src_gradient_yp = [1, 2, None, None], [1, 2, None, None]
    # Destination slices are used only for padding, so the first two
    # None values shouldn't raise errors
    dst_slices = [None, None, [1, 2, 1, 3], [1, 3, 1, 4]]
    # The first two chunks have the same target location, same for the two last
    dst_mosaic_locations = [(0, 0), (0, 0), (0, 1), (0, 1)]

    res = parallel_gradient_search(data, src_x, src_y, dst_x, dst_y,
                                   src_gradient_xl, src_gradient_xp,
                                   src_gradient_yl, src_gradient_yp,
                                   dst_mosaic_locations, dst_slices,
                                   method='foo')
    assert len(res[(0, 0)]) == 2
    # The second padding shouldn't be in the chunks[(0, 1)] list
    assert len(res[(0, 1)]) == 1
    _concatenate_chunks.assert_called_with(res)
    # Two padding arrays
    assert dask_da.full.call_count == 2
    assert mock.call((1, 1, 2), np.nan) in dask_da.full.mock_calls
    assert mock.call((1, 2, 3), np.nan) in dask_da.full.mock_calls
    # Two resample calls
    assert dask_da.from_delayed.call_count == 2
    # The _gradient_resample_data() function has been delayed twice
    assert '_gradient_resample_data' in str(delayed.mock_calls[0])
    assert '_gradient_resample_data' in str(delayed.mock_calls[2])
    assert str(mock.call()(data[0],
                           src_x[0], src_y[0],
                           src_gradient_xl[0], src_gradient_xp[0],
                           src_gradient_yl[0], src_gradient_yp[0],
                           dst_x[0], dst_y[0],
                           method='foo')) == str(delayed.mock_calls[1])
    assert str(mock.call()(data[1],
                           src_x[1], src_y[1],
                           src_gradient_xl[1], src_gradient_xp[1],
                           src_gradient_yl[1], src_gradient_yp[1],
                           dst_x[1], dst_y[1],
                           method='foo')) == str(delayed.mock_calls[3])


def test_concatenate_chunks():
    """Test chunk concatenation for correct results."""
    from pyresample.gradient import _concatenate_chunks

    # 1-band image
    chunks = {(0, 0): [np.ones((1, 5, 4)), np.zeros((1, 5, 4))],
              (1, 0): [np.zeros((1, 5, 2))],
              (1, 1): [np.full((1, 3, 2), 0.5)],
              (0, 1): [np.full((1, 3, 4), -1)]}
    res = _concatenate_chunks(chunks).compute(scheduler='single-threaded')
    assert np.all(res[0, :5, :4] == 1.0)
    assert np.all(res[0, :5, 4:] == 0.0)
    assert np.all(res[0, 5:, :4] == -1.0)
    assert np.all(res[0, 5:, 4:] == 0.5)
    assert res.shape == (1, 8, 6)

    # 3-band image
    chunks = {(0, 0): [np.ones((3, 5, 4)), np.zeros((3, 5, 4))],
              (1, 0): [np.zeros((3, 5, 2))],
              (1, 1): [np.full((3, 3, 2), 0.5)],
              (0, 1): [np.full((3, 3, 4), -1)]}
    res = _concatenate_chunks(chunks).compute(scheduler='single-threaded')
    assert np.all(res[:, :5, :4] == 1.0)
    assert np.all(res[:, :5, 4:] == 0.0)
    assert np.all(res[:, 5:, :4] == -1.0)
    assert np.all(res[:, 5:, 4:] == 0.5)
    assert res.shape == (3, 8, 6)


@mock.patch('pyresample.gradient.da')
def test_concatenate_chunks_stack_calls(dask_da):
    """Test that stacking is called the correct times in chunk concatenation."""
    from pyresample.gradient import _concatenate_chunks

    chunks = {(0, 0): [np.ones((1, 5, 4)), np.zeros((1, 5, 4))],
              (1, 0): [np.zeros((1, 5, 2))],
              (1, 1): [np.full((1, 3, 2), 0.5)],
              (0, 1): [np.full((1, 3, 4), -1)]}
    _ = _concatenate_chunks(chunks)
    dask_da.stack.assert_called_once_with(chunks[(0, 0)], axis=-1)
    dask_da.nanmax.assert_called_once()
    assert 'axis=2' in str(dask_da.concatenate.mock_calls[-1])


class TestGradientCython():
    """Test the core gradient features."""

    def setup_method(self):
        """Set up the test case."""
        self.src_x, self.src_y = np.meshgrid(range(10), range(10))

        self.xl, self.xp = np.gradient(self.src_x)
        self.yl, self.yp = np.gradient(self.src_y)

        self.dst_x = np.array([[1.5, 1.99999, 2.7],
                               [1.6, 2.0, 2.8],
                               [1.7, 2.3, 2.9]]) + 5

        self.dst_y = np.array([[1.1, 1.3, 1.5],
                               [2.1, 2.3, 2.5],
                               [2.8, 2.9, 3.0]]) + 5

    def test_index_search_works(self):
        """Test that index search works."""
        from pyresample.gradient._gradient_search import one_step_gradient_indices
        res_x, res_y = one_step_gradient_indices(self.src_x.astype(float),
                                                 self.src_y.astype(float),
                                                 self.xl, self.xp, self.yl, self.yp,
                                                 self.dst_x, self.dst_y)
        np.testing.assert_allclose(res_x, self.dst_x)
        np.testing.assert_allclose(res_y, self.dst_y)

    def test_index_search_with_data_requested_outside_bottom_right_boundary(self):
        """Test index search with data requested outside bottom right boundary."""
        from pyresample.gradient._gradient_search import one_step_gradient_indices
        self.dst_x += 2
        self.dst_y += 2
        expected_x = np.full([3, 3], np.nan)
        expected_x[0, 0] = 8.5
        expected_x[0, 1] = 8.99999
        expected_y = np.full([3, 3], np.nan)
        expected_y[0, 0] = 8.1
        expected_y[0, 1] = 8.3
        res_x, res_y = one_step_gradient_indices(self.src_x.astype(float),
                                                 self.src_y.astype(float),
                                                 self.xl, self.xp, self.yl, self.yp,
                                                 self.dst_x, self.dst_y)
        np.testing.assert_allclose(res_x, expected_x)
        np.testing.assert_allclose(res_y, expected_y)

    def test_index_search_with_data_requested_outside_top_left_boundary(self):
        """Test index search with data requested outside top left boundary."""
        from pyresample.gradient._gradient_search import one_step_gradient_indices
        self.dst_x -= 7
        self.dst_y -= 7
        expected_x = np.copy(self.dst_x)
        expected_x[:, 0] = np.nan
        expected_x[0, :] = np.nan
        expected_y = np.copy(self.dst_y)
        expected_y[0, :] = np.nan
        expected_y[:, 0] = np.nan
        res_x, res_y = one_step_gradient_indices(self.src_x.astype(float),
                                                 self.src_y.astype(float),
                                                 self.xl, self.xp, self.yl, self.yp,
                                                 self.dst_x, self.dst_y)
        np.testing.assert_allclose(res_x, expected_x)
        np.testing.assert_allclose(res_y, expected_y)
