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
"""Test base resampler class functionality."""
from __future__ import annotations

from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pytest_lazy_fixtures import lf

from pyresample.future.resamplers.resampler import Resampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.resampler import BaseResampler


class FakeResampler(Resampler):
    """Helper fake resampler for easier testing."""

    def __init__(self, *args, **kwargs):
        self.precompute = mock.Mock(wraps=self.precompute)
        self.resample = mock.Mock(wraps=self.resample)
        self._internal_cache = {}
        super().__init__(*args, **kwargs)

    def precompute(self, **kwargs):
        hash = self._get_hash(**kwargs)
        if hash not in self._internal_cache:
            some_result = np.empty((2, 2))
            self._internal_cache[hash] = some_result
        return self._internal_cache[hash]

    def resample(self, data, **kwargs):
        self.precompute(**kwargs)
        hash = self._get_hash(**kwargs)
        _ = self._internal_cache[hash]
        return np.empty(self.target_geo_def.shape)


@pytest.mark.parametrize(
    "src",
    [
        lf("swath_def_2d_numpy"),
        lf("swath_def_2d_dask"),
        lf("swath_def_2d_xarray_numpy"),
        lf("swath_def_2d_xarray_dask"),
    ]
)
@pytest.mark.parametrize(
    "dst",
    [
        lf("area_def_lcc_conus_1km"),
    ]
)
def test_resampler(src, dst):
    """Test basic operations of the base resampler with and without a caching."""
    rs = FakeResampler(src, dst)
    some_data = np.zeros(src.shape, dtype=np.float64)
    resample_results = rs.resample(some_data)
    rs.precompute.assert_called_once()
    assert resample_results.shape == dst.shape


@pytest.mark.parametrize(
    ("use_swaths", "copy_dst_swath"),
    [
        (False, None),
        (True, None),  # same objects are equal
        (True, "dask"),  # same dask tasks are equal
        (True, "swath_def"),  # same underlying arrays are equal
    ])
def test_base_resampler_does_nothing_when_src_and_dst_areas_are_equal(_geos_area, use_swaths, copy_dst_swath):
    """Test that the BaseResampler does nothing when the source and target areas are the same."""
    src_geom = _geos_area if not use_swaths else _xarray_swath_def_from_area(_geos_area)
    dst_geom = src_geom
    if copy_dst_swath == "dask":
        dst_geom = _xarray_swath_def_from_area(_geos_area)
    elif copy_dst_swath == "swath_def":
        dst_geom = SwathDefinition(dst_geom.lons, dst_geom.lats)

    resampler = BaseResampler(src_geom, dst_geom)
    some_data = xr.DataArray(da.zeros(src_geom.shape, dtype=np.float64), dims=('y', 'x'))
    assert resampler.resample(some_data) is some_data


@pytest.mark.parametrize(
    ("src_area", "numpy_swath"),
    [
        (False, False),
        (False, True),
        (True, False),
    ])
@pytest.mark.parametrize("dst_area", [False, True])
def test_base_resampler_unequal_geometries(_geos_area, _geos_area2, src_area, numpy_swath, dst_area):
    """Test cases where BaseResampler geometries are not considered equal."""
    src_geom = _geos_area if src_area else _xarray_swath_def_from_area(_geos_area, numpy_swath)
    dst_geom = _geos_area2 if dst_area else _xarray_swath_def_from_area(_geos_area2)
    resampler = BaseResampler(src_geom, dst_geom)
    some_data = xr.DataArray(da.zeros(src_geom.shape, dtype=np.float64), dims=('y', 'x'))
    with pytest.raises(NotImplementedError):
        resampler.resample(some_data)


def _xarray_swath_def_from_area(area_def, use_numpy=False):
    chunks = None if use_numpy else -1
    lons_da, lats_da = area_def.get_lonlats(chunks=chunks)
    lons = xr.DataArray(lons_da, dims=('y', 'x'))
    lats = xr.DataArray(lats_da, dims=('y', 'x'))
    swath_def = SwathDefinition(lons, lats)
    return swath_def


@pytest.fixture
def _geos_area():
    src_area = AreaDefinition('src', 'src area', None,
                              {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                              100, 100,
                              (5550000.0, 5550000.0, -5550000.0, -5550000.0))
    return src_area


@pytest.fixture
def _geos_area2():
    src_area = AreaDefinition('src', 'src area', None,
                              {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                              200, 200,
                              (5550000.0, 5550000.0, -5550000.0, -5550000.0))
    return src_area
