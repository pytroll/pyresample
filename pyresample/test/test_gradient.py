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
from pyresample.gradient import ResampleBlocksGradientSearchResampler, create_gradient_search_resampler


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


class TestRBGradientSearchResamplerSwath2Area:
    """Test RBGradientSearchResampler for the Area to Swath case."""

    def setup_method(self):
        """Set up the test case."""
        lons, lats = np.meshgrid(np.linspace(0, 20, 100), np.linspace(45, 66, 100))
        self.src_swath = SwathDefinition(lons, lats, crs="WGS84")
        lons, lats = self.src_swath.get_lonlats(chunks=10)
        lons = xr.DataArray(lons, dims=["y", "x"])
        lats = xr.DataArray(lats, dims=["y", "x"])
        self.src_swath_dask = SwathDefinition(lons, lats)
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

    @pytest.mark.parametrize("input_dtype", (np.float32, np.float64))
    def test_resample_swath_to_area_2d(self, input_dtype):
        """Resample swath to area, 2d."""
        swath_resampler = ResampleBlocksGradientSearchResampler(self.src_swath_dask, self.dst_area)

        data = xr.DataArray(da.ones(self.src_swath.shape, dtype=input_dtype),
                            dims=['y', 'x'])
        with np.errstate(invalid="ignore"):  # 'inf' space pixels cause runtime warnings
            swath_resampler.precompute()
            res_xr = swath_resampler.compute(data, method='bilinear')
            res_np = res_xr.compute(scheduler='single-threaded')

        assert res_xr.dtype == data.dtype
        assert res_np.dtype == data.dtype
        assert res_xr.shape == self.dst_area.shape
        assert res_np.shape == self.dst_area.shape
        assert type(res_xr) is type(data)
        assert type(res_xr.data) is type(data.data)
        assert not np.all(np.isnan(res_np))

    @pytest.mark.parametrize("input_dtype", (np.float32, np.float64))
    def test_resample_swath_to_area_3d(self, input_dtype):
        """Resample area to area, 3d."""
        swath_resampler = ResampleBlocksGradientSearchResampler(self.src_swath_dask, self.dst_area)

        data = xr.DataArray(da.ones((3, ) + self.src_swath.shape,
                                    dtype=input_dtype) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        with np.errstate(invalid="ignore"):  # 'inf' space pixels cause runtime warnings
            swath_resampler.precompute()
            res_xr = swath_resampler.compute(data, method='bilinear')
            res_np = res_xr.compute(scheduler='single-threaded')

        assert res_xr.dtype == data.dtype
        assert res_np.dtype == data.dtype
        assert res_xr.shape == (3, ) + self.dst_area.shape
        assert res_np.shape == (3, ) + self.dst_area.shape
        assert type(res_xr) is type(data)
        assert type(res_xr.data) is type(data.data)
        for i in range(res_np.shape[0]):
            arr = np.ravel(res_np[i, :, :])
            assert np.allclose(arr[np.isfinite(arr)], float(i + 1))


class TestRBGradientSearchResamplerArea2Swath:
    """Test RBGradientSearchResampler for the Area to Swath case."""

    def setup_method(self):
        """Set up the test case."""
        lons, lats = np.meshgrid(np.linspace(0, 20, 100), np.linspace(45, 66, 100))
        self.dst_swath = SwathDefinition(lons, lats, crs="WGS84")
        lons, lats = self.dst_swath.get_lonlats(chunks=10)
        lons = xr.DataArray(lons, dims=["y", "x"])
        lats = xr.DataArray(lats, dims=["y", "x"])
        self.dst_swath_dask = SwathDefinition(lons, lats)

        self.src_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

    @pytest.mark.parametrize("input_dtype", (np.float32, np.float64))
    def test_resample_area_to_swath_2d(self, input_dtype):
        """Resample swath to area, 2d."""
        swath_resampler = create_gradient_search_resampler(self.src_area, self.dst_swath_dask)

        data = xr.DataArray(da.ones(self.src_area.shape, dtype=input_dtype),
                            dims=['y', 'x'])
        with np.errstate(invalid="ignore"):  # 'inf' space pixels cause runtime warnings
            swath_resampler.precompute()
            res_xr = swath_resampler.compute(data, method='bilinear')
            res_np = res_xr.compute(scheduler='single-threaded')

        assert res_xr.dtype == data.dtype
        assert res_np.dtype == data.dtype
        assert res_xr.shape == self.dst_swath.shape
        assert res_np.shape == self.dst_swath.shape
        assert type(res_xr) is type(data)
        assert type(res_xr.data) is type(data.data)
        assert not np.all(np.isnan(res_np))

    @pytest.mark.parametrize("input_dtype", (np.float32, np.float64))
    def test_resample_area_to_swath_3d(self, input_dtype):
        """Resample area to area, 3d."""
        swath_resampler = ResampleBlocksGradientSearchResampler(self.src_area, self.dst_swath_dask)

        data = xr.DataArray(da.ones((3, ) + self.src_area.shape,
                                    dtype=input_dtype) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        with np.errstate(invalid="ignore"):  # 'inf' space pixels cause runtime warnings
            swath_resampler.precompute()
            res_xr = swath_resampler.compute(data, method='bilinear')
            res_np = res_xr.compute(scheduler='single-threaded')

        assert res_xr.dtype == data.dtype
        assert res_np.dtype == data.dtype
        assert res_xr.shape == (3, ) + self.dst_swath.shape
        assert res_np.shape == (3, ) + self.dst_swath.shape
        assert type(res_xr) is type(data)
        assert type(res_xr.data) is type(data.data)
        for i in range(res_np.shape[0]):
            arr = np.ravel(res_np[i, :, :])
            assert np.allclose(arr[np.isfinite(arr)], float(i + 1))


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
