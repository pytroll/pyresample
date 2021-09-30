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
"""Shared test configuration and fixtures."""

import dask.array as da
import pytest
import xarray as xr
from pyproj import CRS

from pyresample import AreaDefinition, SwathDefinition
from pyresample.test.utils import create_test_latitude, create_test_longitude

SWATH_SHAPE = (200, 1500)
AREA_SHAPE = (1500, 2000)


def _conus_lonlats():
    lons = create_test_longitude(-105.0, -90.0, SWATH_SHAPE)
    lats = create_test_latitude(25.0, 33.0, SWATH_SHAPE)
    return lons, lats


def _conus_lonlats_dask():
    lons, lats = _conus_lonlats()
    lons = da.from_array(lons)
    lats = da.from_array(lats)
    return lons, lats


def _antimeridian_lonlats():
    lons = create_test_longitude(172.0, 190.0, SWATH_SHAPE)
    lons[lons > 180.0] = lons - 360.0
    lats = create_test_latitude(25.0, 33.0, SWATH_SHAPE)
    return lons, lats


@pytest.fixture(scope="session")
def swath_def_2d_numpy():
    """Create a SwathDefinition with numpy arrays (200, 1500)."""
    lons, lats = _conus_lonlats()
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_dask():
    """Create a SwathDefinition with dask arrays (200, 1500)."""
    lons, lats = _conus_lonlats_dask()
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_xarray_numpy():
    """Create a SwathDefinition with DataArrays(numpy) (200, 1500)."""
    lons, lats = _conus_lonlats()
    lons = xr.DataArray(lons, dims=("y", "x"))
    lats = xr.DataArray(lats, dims=("y", "x"))
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_xarray_dask():
    """Create a SwathDefinition with DataArrays(dask) (200, 1500)."""
    lons, lats = _conus_lonlats_dask()
    lons = xr.DataArray(lons, dims=("y", "x"))
    lats = xr.DataArray(lats, dims=("y", "x"))
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_numpy_antimeridian():
    """Create a SwathDefinition with numpy arrays (200, 1500) over the antimeridian.

    Longitude values go from positive values to negative values as they cross -180/180.

    """
    lons, lats = _antimeridian_lonlats()
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def area_def_lcc_conus_1km():
    """Create an AreaDefinition with an LCC projection over CONUS (1500, 2000)."""
    proj_str = "+proj=lcc +lon_0=-95 +lat_1=35.0 +lat_2=35.0 +datum=WGS84 +no_defs"
    crs = CRS.from_string(proj_str)
    area_def = AreaDefinition("area_def_lcc_conus", "", "",
                              crs, AREA_SHAPE[1], AREA_SHAPE[0],
                              (-750000, -750000, 750000, 750000))
    return area_def
