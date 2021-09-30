#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
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
"""Tests for the 'nearest' resampler."""

from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from pyresample import geometry
from pyresample.future.resamplers import NearestNeighborResampler


@pytest.fixture(scope="session")
def area_def_stere_800x800_target():
    """Create an AreaDefinition with a polar-stereographic projection (800, 800)."""
    return geometry.AreaDefinition(
        'areaD', 'Europe (3km, HRV, VTC)', 'areaD',
        {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '50.00',
            'lat_ts': '50.00',
            'lon_0': '8.00',
            'proj': 'stere'
        },
        800, 800,
        [-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001]
    )


@pytest.fixture(scope="session")
def coord_def_2d_dask():
    """Create a 2D CoordinateDefinition of dask arrays (4, 3)."""
    chunks = 5
    lons = da.from_array(np.array([
        [11.5, 12.562036, 12.9],
        [11.5, 12.562036, 12.9],
        [11.5, 12.562036, 12.9],
        [11.5, 12.562036, 12.9],
    ]), chunks=chunks)
    lats = da.from_array(np.array([
        [55.715613, 55.715613, 55.715613],
        [55.715613, 55.715613, 55.715613],
        [55.715613, np.nan, 55.715613],
        [55.715613, 55.715613, 55.715613],
    ]), chunks=chunks)
    return geometry.CoordinateDefinition(lons=lons, lats=lats)


@pytest.fixture(scope="session")
def swath_def_1d_xarray_dask():
    """Create a 1D SwathDefinition of DataArrays(dask) (3,)."""
    chunks = 5
    tlons_1d = xr.DataArray(
        da.from_array(np.array([11.280789, 12.649354, 12.080402]), chunks=chunks),
        dims=('my_dim1',))
    tlats_1d = xr.DataArray(
        da.from_array(np.array([56.011037, 55.629675, 55.641535]), chunks=chunks),
        dims=('my_dim1',))
    return geometry.SwathDefinition(lons=tlons_1d, lats=tlats_1d)


@pytest.fixture(scope="session")
def data_1d_xarray_dask():
    """Create a sample 1D data DataArray(dask) (3,)."""
    return xr.DataArray(
        da.from_array(np.array([1., 2., 3.]), chunks=5), dims=('my_dim1',))


@pytest.fixture(scope="session")
def data_2d_xarray_dask():
    """Create a sample 2D data DataArray(dask) (50, 10)."""
    return xr.DataArray(
        da.from_array(np.fromfunction(lambda y, x: y * x, (50, 10)),
                      chunks=5),
        dims=('my_dim_y', 'my_dim_x'))


@pytest.fixture(scope="session")
def data_3d_xarray_dask():
    """Create a sample 3D data DataArray(dask) (50, 10, 3)."""
    return xr.DataArray(
        da.from_array(np.fromfunction(lambda y, x, b: y * x * b, (50, 10, 3)),
                      chunks=5),
        dims=('my_dim_y', 'my_dim_x', 'bands'),
        coords={'bands': ['r', 'g', 'b']})


@pytest.fixture(scope="session")
def swath_def_2d_xarray_dask():
    """Create a 2D SwathDefinition of DataArrays(dask) (50, 10)."""
    lons_2d = xr.DataArray(
        da.from_array(np.fromfunction(lambda y, x: 3 + x, (50, 10)),
                      chunks=5),
        dims=('my_dim_y', 'my_dim_x'))
    lats_2d = xr.DataArray(
        da.from_array(np.fromfunction(lambda y, x: 75 - y, (50, 10)),
                      chunks=5),
        dims=('my_dim_y', 'my_dim_x'))
    return geometry.SwathDefinition(lons=lons_2d, lats=lats_2d)


@pytest.fixture(scope="session")
def area_def_stere_50x10_source():
    """Create an AreaDefinition with a polar-stereographic projection (10, 50)."""
    return geometry.AreaDefinition(
        'areaD', 'Europe (3km, HRV, VTC)', 'areaD',
        {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '52.00',
            'lat_ts': '52.00',
            'lon_0': '5.00',
            'proj': 'stere'
        },
        50, 10,
        [-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001]
    )


class TestNearestNeighborResampler:
    """Test the NearestNeighborResampler class."""

    def test_nearest_swath_1d_mask_to_grid_1n(self, swath_def_1d_xarray_dask, data_1d_xarray_dask, coord_def_2d_dask):
        """Test 1D swath definition to 2D grid definition; 1 neighbor."""
        resampler = NearestNeighborResampler(swath_def_1d_xarray_dask, coord_def_2d_dask)
        res = resampler.resample(data_1d_xarray_dask,
                                 mask_area=data_1d_xarray_dask.isnull(),
                                 radius_of_influence=100000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        actual = res.values
        expected = np.array([
            [1., 2., 2.],
            [1., 2., 2.],
            [1., np.nan, 2.],
            [1., 2., 2.],
        ])
        np.testing.assert_allclose(actual, expected)

    def test_nearest_type_preserve(self, swath_def_1d_xarray_dask, coord_def_2d_dask):
        """Test 1D swath definition to 2D grid definition; 1 neighbor."""
        data = xr.DataArray(da.from_array(np.array([1, 2, 3]),
                                          chunks=5),
                            dims=('my_dim1',))

        resampler = NearestNeighborResampler(swath_def_1d_xarray_dask, coord_def_2d_dask)
        res = resampler.resample(data, fill_value=255,
                                 radius_of_influence=100000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        actual = res.values
        expected = np.array([
            [1, 2, 2],
            [1, 2, 2],
            [1, 255, 2],
            [1, 2, 2],
        ])
        np.testing.assert_equal(actual, expected)

    def test_nearest_swath_2d_mask_to_area_1n(self, swath_def_2d_xarray_dask, data_2d_xarray_dask,
                                              area_def_stere_800x800_target):
        """Test 2D swath definition to 2D area definition; 1 neighbor."""
        resampler = NearestNeighborResampler(
            swath_def_2d_xarray_dask, area_def_stere_800x800_target)
        res = resampler.resample(data_2d_xarray_dask, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        res = res.values
        cross_sum = np.nansum(res)
        expected = 15874591.0
        assert cross_sum == expected

    def test_nearest_area_2d_to_area_1n(self, area_def_stere_50x10_source, data_2d_xarray_dask,
                                        area_def_stere_800x800_target):
        """Test 2D area definition to 2D area definition; 1 neighbor."""
        from pyresample.test.utils import assert_maximum_dask_computes
        resampler = NearestNeighborResampler(
            area_def_stere_50x10_source, area_def_stere_800x800_target)
        with assert_maximum_dask_computes(0):
            resampler.precompute(radius_of_influence=50000)
        pytest.raises(AssertionError, resampler.resample, data_2d_xarray_dask, radius_of_influence=50000)

        # rename data dimensions to match the expected area dimensions
        data = data_2d_xarray_dask.rename({'my_dim_y': 'y', 'my_dim_x': 'x'})
        with assert_maximum_dask_computes(0):
            res = resampler.resample(data, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        res = res.values
        cross_sum = np.nansum(res)
        expected = 27706753.0
        assert cross_sum == expected

    def test_nearest_area_2d_to_area_1n_no_roi(self, area_def_stere_50x10_source, data_2d_xarray_dask,
                                               area_def_stere_800x800_target):
        """Test 2D area definition to 2D area definition; 1 neighbor, no radius of influence."""
        resampler = NearestNeighborResampler(
            area_def_stere_50x10_source, area_def_stere_800x800_target)
        resampler.precompute()
        pytest.raises(AssertionError, resampler.resample, data_2d_xarray_dask)

        # rename data dimensions to match the expected area dimensions
        data = data_2d_xarray_dask.rename({'my_dim_y': 'y', 'my_dim_x': 'x'})
        res = resampler.resample(data)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        res = res.values
        cross_sum = np.nansum(res)
        expected = 87281406.0
        assert cross_sum == expected

        # pretend the resolutions can't be determined
        with mock.patch.object(area_def_stere_50x10_source, 'geocentric_resolution') as sgr, \
                mock.patch.object(area_def_stere_800x800_target, 'geocentric_resolution') as dgr:
            sgr.side_effect = RuntimeError
            dgr.side_effect = RuntimeError
            resampler = NearestNeighborResampler(
                area_def_stere_50x10_source, area_def_stere_800x800_target)
            res = resampler.resample(data)
            assert isinstance(res, xr.DataArray)
            assert isinstance(res.data, da.Array)
            res = res.values
            cross_sum = np.nansum(res)
            expected = 1855928.0
            assert cross_sum == expected

    def test_nearest_area_2d_to_area_1n_3d_data(self, area_def_stere_50x10_source, data_3d_xarray_dask,
                                                area_def_stere_800x800_target):
        """Test 2D area definition to 2D area definition; 1 neighbor, 3d data."""
        resampler = NearestNeighborResampler(
            area_def_stere_50x10_source, area_def_stere_800x800_target)
        resampler.precompute(radius_of_influence=50000)
        pytest.raises(AssertionError, resampler.resample, data_3d_xarray_dask, radius_of_influence=50000)

        # rename data dimensions to match the expected area dimensions
        data = data_3d_xarray_dask.rename({'my_dim_y': 'y', 'my_dim_x': 'x'})
        res = resampler.resample(data, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert list(res.coords['bands']) == ['r', 'g', 'b']
        res = res.values
        cross_sum = np.nansum(res)
        expected = 83120259.0
        assert cross_sum == expected

    @pytest.mark.skipif(True, reason="Multiple neighbors not supported yet")
    def test_nearest_swath_1d_mask_to_grid_8n(self, swath_def_1d_xarray_dask, data_1d_xarray_dask, coord_def_2d_dask):
        """Test 1D swath definition to 2D grid definition; 8 neighbors."""
        resampler = NearestNeighborResampler(
            swath_def_1d_xarray_dask, coord_def_2d_dask)
        resampler.precompute(mask=data_1d_xarray_dask.isnull(),
                             radius_of_influence=100000, neighbors=8)
        res = resampler.resample(data_1d_xarray_dask)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        # actual = res.values
        # expected =
        # np.testing.assert_allclose(actual, expected)
