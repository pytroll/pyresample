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

from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pytest_lazy_fixtures import lf

from pyresample.future.geometry import AreaDefinition, SwathDefinition
from pyresample.future.resamplers import KDTreeNearestXarrayResampler
from pyresample.test.utils import assert_maximum_dask_computes, assert_warnings_contain, catch_warnings
from pyresample.utils.errors import PerformanceWarning


def _check_common_metadata(data_arr: Any, target_is_area: bool = False) -> None:
    if not isinstance(data_arr, xr.DataArray):
        return
    if not target_is_area:
        return

    for coord_name in ("y", "x"):
        assert coord_name in data_arr.coords
        c_arr = data_arr.coords[coord_name]
        assert c_arr.attrs.get("units") in ("meter", "degrees_north", "degrees_east")
    assert "y" in data_arr.coords
    assert "x" in data_arr.coords


class TestNearestNeighborResampler:
    """Test the KDTreeNearestXarrayResampler class."""

    def test_nearest_swath_1d_mask_to_grid_1n(
            self,
            swath_def_1d_xarray_dask, data_1d_float32_xarray_dask, coord_def_2d_float32_dask):
        """Test 1D swath definition to 2D grid definition; 1 neighbor."""
        resampler = KDTreeNearestXarrayResampler(swath_def_1d_xarray_dask, coord_def_2d_float32_dask)
        res = resampler.resample(data_1d_float32_xarray_dask,
                                 mask_area=data_1d_float32_xarray_dask.isnull(),
                                 radius_of_influence=100000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        _check_common_metadata(res, isinstance(coord_def_2d_float32_dask, AreaDefinition))
        actual = res.values
        expected = np.array([
            [1., 2., 2.],
            [1., 2., 2.],
            [1., np.nan, 2.],
            [1., 2., 2.],
        ])
        np.testing.assert_allclose(actual, expected)

    def test_nearest_type_preserve(self, swath_def_1d_xarray_dask, coord_def_2d_float32_dask):
        """Test 1D swath definition to 2D grid definition; 1 neighbor."""
        data = xr.DataArray(da.from_array(np.array([1, 2, 3]),
                                          chunks=5),
                            dims=('my_dim1',))

        resampler = KDTreeNearestXarrayResampler(swath_def_1d_xarray_dask, coord_def_2d_float32_dask)
        res = resampler.resample(data, fill_value=255,
                                 radius_of_influence=100000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        _check_common_metadata(res, isinstance(coord_def_2d_float32_dask, AreaDefinition))
        actual = res.values
        expected = np.array([
            [1, 2, 2],
            [1, 2, 2],
            [1, 255, 2],
            [1, 2, 2],
        ])
        np.testing.assert_equal(actual, expected)

    def test_nearest_swath_2d_mask_to_area_1n(self, swath_def_2d_xarray_dask, data_2d_float32_xarray_dask,
                                              area_def_stere_target):
        """Test 2D swath definition to 2D area definition; 1 neighbor."""
        resampler = KDTreeNearestXarrayResampler(
            swath_def_2d_xarray_dask, area_def_stere_target)
        res = resampler.resample(data_2d_float32_xarray_dask, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        _check_common_metadata(res, isinstance(area_def_stere_target, AreaDefinition))
        res = res.values
        cross_sum = float(np.nansum(res))
        expected = 167913.0
        assert cross_sum == expected
        assert res.shape == resampler.target_geo_def.shape

    def test_nearest_area_2d_to_area_1n(self, area_def_stere_source, data_2d_float32_xarray_dask,
                                        area_def_stere_target):
        """Test 2D area definition to 2D area definition; 1 neighbor."""
        resampler = KDTreeNearestXarrayResampler(
            area_def_stere_source, area_def_stere_target)
        with assert_maximum_dask_computes(0):
            resampler.precompute(radius_of_influence=50000)

        with assert_maximum_dask_computes(0):
            res = resampler.resample(data_2d_float32_xarray_dask, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        _check_common_metadata(res, isinstance(area_def_stere_target, AreaDefinition))
        res = res.values
        cross_sum = float(np.nansum(res))
        expected = 303048.0
        assert cross_sum == expected
        assert res.shape == resampler.target_geo_def.shape

    def test_nearest_swath_2d_to_area_1n_pm180(self, swath_def_2d_xarray_dask_antimeridian, data_2d_float32_xarray_dask,
                                               area_def_lonlat_pm180_target):
        """Test 2D swath definition to 2D area definition; 1 neighbor; output prime meridian at 180 degrees."""
        resampler = KDTreeNearestXarrayResampler(
            swath_def_2d_xarray_dask_antimeridian, area_def_lonlat_pm180_target)
        res = resampler.resample(data_2d_float32_xarray_dask, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        _check_common_metadata(res, isinstance(area_def_lonlat_pm180_target, AreaDefinition))
        res = res.values
        cross_sum = float(np.nansum(res))
        expected = 115591.0
        assert cross_sum == expected
        assert res.shape == resampler.target_geo_def.shape

    def test_nearest_area_2d_to_area_1n_no_roi(self, area_def_stere_source, data_2d_float32_xarray_dask,
                                               area_def_stere_target):
        """Test 2D area definition to 2D area definition; 1 neighbor, no radius of influence."""
        resampler = KDTreeNearestXarrayResampler(
            area_def_stere_source, area_def_stere_target)
        resampler.precompute()

        res = resampler.resample(data_2d_float32_xarray_dask)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        _check_common_metadata(res, isinstance(area_def_stere_target, AreaDefinition))
        res = res.values
        cross_sum = float(np.nansum(res))
        expected = 952386.0
        assert cross_sum == expected
        assert res.shape == resampler.target_geo_def.shape

    def test_nearest_area_2d_to_area_1n_no_roi_no_geocentric(
            self,
            area_def_stere_source,
            data_2d_float32_xarray_dask,
            area_def_stere_target):
        # pretend the resolutions can't be determined
        with mock.patch.object(area_def_stere_source, 'geocentric_resolution') as sgr, \
                mock.patch.object(area_def_stere_target, 'geocentric_resolution') as dgr:
            sgr.side_effect = RuntimeError
            dgr.side_effect = RuntimeError
            resampler = KDTreeNearestXarrayResampler(
                area_def_stere_source, area_def_stere_target)
            res = resampler.resample(data_2d_float32_xarray_dask)
            assert isinstance(res, xr.DataArray)
            assert isinstance(res.data, da.Array)
            _check_common_metadata(res, isinstance(area_def_stere_target, AreaDefinition))
            res = res.values
            cross_sum = np.nansum(res)
            expected = 20666.0
            assert cross_sum == expected
            assert res.shape == resampler.target_geo_def.shape

    @pytest.mark.parametrize("input_data", [
        lf("data_2d_float32_numpy"),
        lf("data_2d_float32_dask"),
        lf("data_2d_float32_xarray_numpy"),
    ])
    def test_object_type_with_warnings(
            self,
            area_def_stere_source,
            area_def_stere_target,
            input_data):
        """Test that providing certain input data causes a warning."""
        resampler = KDTreeNearestXarrayResampler(area_def_stere_source, area_def_stere_target)
        with catch_warnings(PerformanceWarning) as w, assert_maximum_dask_computes(1):
            res = resampler.resample(input_data)
            assert type(res) is type(input_data)
        _check_common_metadata(res, isinstance(area_def_stere_target, AreaDefinition))
        is_data_arr_dask = isinstance(input_data, xr.DataArray) and isinstance(input_data.data, da.Array)
        is_dask_based = isinstance(input_data, da.Array) or is_data_arr_dask
        if is_dask_based:
            assert not w
        else:
            assert_warnings_contain(w, "will be converted")
        res = np.array(res)
        cross_sum = np.nansum(res)
        expected = 952386.0  # same as 'test_nearest_area_2d_to_area_1n_no_roi'
        assert cross_sum == expected
        assert res.shape == resampler.target_geo_def.shape

    def test_nearest_area_2d_to_area_1n_3d_data(self, area_def_stere_source, data_3d_float32_xarray_dask,
                                                area_def_stere_target):
        """Test 2D area definition to 2D area definition; 1 neighbor, 3d data."""
        resampler = KDTreeNearestXarrayResampler(
            area_def_stere_source, area_def_stere_target)
        resampler.precompute(radius_of_influence=50000)

        res = resampler.resample(data_3d_float32_xarray_dask, radius_of_influence=50000)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert list(res.coords['bands']) == ['r', 'g', 'b']
        _check_common_metadata(res, isinstance(area_def_stere_target, AreaDefinition))
        res = res.values
        cross_sum = float(np.nansum(res))
        expected = 909144.0
        assert cross_sum == expected
        assert res.shape[:2] == resampler.target_geo_def.shape


class TestInvalidUsageNearestNeighborResampler:
    """Test the resampler being given input that should raise an error.

    If a case here is removed because functionality is added to the resampler
    then a working case should be added above.

    """

    @pytest.mark.parametrize(
        "input_data",
        [
            lf("data_2d_float32_xarray_dask"),
            lf("data_3d_float32_xarray_dask"),
        ]
    )
    def test_mismatch_geo_data_dims(
            self,
            area_def_stere_source,
            area_def_stere_target,
            input_data,
    ):
        resampler = KDTreeNearestXarrayResampler(area_def_stere_source, area_def_stere_target)
        data = input_data.rename({'y': 'my_dim_y', 'x': 'my_dim_x'})
        with pytest.raises(ValueError, match='.*dimensions do not match.*'):
            resampler.resample(data)

    def test_mismatch_geo_data_dims_swath(
            self,
            swath_def_2d_xarray_dask,
            area_def_stere_target,
            data_2d_float32_xarray_dask):
        new_swath_def = SwathDefinition(
            swath_def_2d_xarray_dask.lons.rename({'y': 'my_dim_y', 'x': 'my_dim_x'}),
            swath_def_2d_xarray_dask.lats.rename({'y': 'my_dim_y', 'x': 'my_dim_x'})
        )
        resampler = KDTreeNearestXarrayResampler(new_swath_def, area_def_stere_target)
        with pytest.raises(ValueError, match='.*dimensions do not match.*'):
            resampler.resample(data_2d_float32_xarray_dask)

    @pytest.mark.parametrize(
        "src_geom",
        [
            lf("area_def_stere_source"),
            lf("swath_def_2d_xarray_dask")
        ]
    )
    @pytest.mark.parametrize(
        ("match", "call_precompute"),
        [
            (".*data.*shape.*", False),
            (".*'mask'.*shape.*", True),
        ]
    )
    def test_inconsistent_input_shapes(self, src_geom, match, call_precompute,
                                       area_def_stere_target, data_2d_float32_xarray_dask):
        """Test that geometry and data of the same size but different size still error."""
        # transpose the source geometries
        if isinstance(src_geom, AreaDefinition):
            src_geom = AreaDefinition(
                src_geom.crs,
                (src_geom.width, src_geom.height),
                src_geom.area_extent,
                attrs=src_geom.attrs.copy(),
            )
        else:
            src_geom = SwathDefinition(
                src_geom.lons.T.rename({'y': 'x', 'x': 'y'}),
                src_geom.lats.T.rename({'y': 'x', 'x': 'y'}),
            )
        resampler = KDTreeNearestXarrayResampler(src_geom, area_def_stere_target)
        with pytest.raises(ValueError, match=match):
            if call_precompute:
                resampler.precompute(mask=data_2d_float32_xarray_dask.notnull())
            else:
                resampler.resample(data_2d_float32_xarray_dask)
