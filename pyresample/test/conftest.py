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
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

import pyresample
from pyresample import LegacyAreaDefinition, LegacySwathDefinition
from pyresample.future.geometry import AreaDefinition, CoordinateDefinition, SwathDefinition
from pyresample.test.utils import create_test_latitude, create_test_longitude

SRC_SWATH_2D_SHAPE = (50, 10)
SRC_SWATH_1D_SHAPE = (3,)
SRC_AREA_SHAPE = (50, 10)
DST_AREA_SHAPE = (80, 85)


@pytest.fixture(autouse=True)
def reset_pyresample_config(tmpdir):
    """Set pyresample config to logical defaults for tests."""
    test_config = {
        "cache_geometry_slices": False,
        "features": {
            "future_geometries": False,
        },
    }
    with pyresample.config.set(test_config):
        yield


@pytest.fixture(
    scope="session",
    params=[LegacySwathDefinition, SwathDefinition],
    ids=["LegacySwathDefinition", "SwathDefinition"])
def swath_class(request):
    """Get one of the currently active 'SwathDefinition' classes.

    Currently only includes the legacy 'SwathDefinition' class and the future
    'SwathDefinition' class in 'pyresample.future.geometry.swath'.

    """
    return request.param


@pytest.fixture(scope="session")
def create_test_swath(swath_class):
    """Get a function for creating SwathDefinitions for testing.

    Should be used as a pytest fixture and will automatically run the test
    function with the legacy SwathDefinition class and the future
    SwathDefinition class. If tests require a specific class they should
    NOT use this fixture and instead use the exact class directly.

    """
    def _create_test_swath(lons, lats, **kwargs):
        if swath_class is SwathDefinition:
            kwargs.pop("nproc", None)
        return swath_class(lons, lats, **kwargs)
    return _create_test_swath


@pytest.fixture(
    scope="session",
    params=[LegacyAreaDefinition, AreaDefinition],
    ids=["LegacyAreaDefinition", "AreaDefinition"])
def area_class(request):
    """Get one of the currently active 'AreaDefinition' classes.

    Currently only includes the legacy 'AreaDefinition' class and the future
    'AreaDefinition' class in 'pyresample.future.geometry.area'.

    """
    return request.param


@pytest.fixture(scope="session")
def create_test_area(area_class):
    """Get a function for creating AreaDefinitions for testing.

    Should be used as a pytest fixture and will automatically run the test
    function with the legacy AreaDefinition class and the future
    AreaDefinition class. If tests require a specific class they should
    NOT use this fixture and instead use the exact class directly.

    """
    def _create_test_area(crs, width, height, area_extent, **kwargs):
        """Create an AreaDefinition object for testing."""
        args = (crs, (height, width), area_extent)
        if area_class is LegacyAreaDefinition:
            args = (crs, width, height, area_extent)
            attrs = kwargs.pop("attrs", {})
            area_id = attrs.pop("name", "test_area")
            args = (area_id, "", "") + args
        area = area_class(*args, **kwargs)
        return area
    return _create_test_area


def _euro_lonlats():
    lons = create_test_longitude(3.0, 12.0, SRC_SWATH_2D_SHAPE)
    lats = create_test_latitude(75.0, 26.0, SRC_SWATH_2D_SHAPE)
    return lons, lats


def _euro_lonlats_dask():
    lons, lats = _euro_lonlats()
    lons = da.from_array(lons)
    lats = da.from_array(lats)
    return lons, lats


def _antimeridian_lonlats():
    lons = create_test_longitude(172.0, 190.0, SRC_SWATH_2D_SHAPE)
    lons[lons > 180.0] -= 360.0
    lats = create_test_latitude(25.0, 33.0, SRC_SWATH_2D_SHAPE)
    return lons, lats


@pytest.fixture(scope="session")
def swath_def_2d_numpy():
    """Create a SwathDefinition with numpy arrays (200, 1500)."""
    lons, lats = _euro_lonlats()
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_dask():
    """Create a SwathDefinition with dask arrays (200, 1500)."""
    lons, lats = _euro_lonlats_dask()
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_xarray_numpy():
    """Create a SwathDefinition with DataArrays(numpy) (200, 1500)."""
    lons, lats = _euro_lonlats()
    lons = xr.DataArray(lons, dims=("y", "x"))
    lats = xr.DataArray(lats, dims=("y", "x"))
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def swath_def_2d_xarray_dask():
    """Create a SwathDefinition with DataArrays(dask) (200, 1500)."""
    lons, lats = _euro_lonlats_dask()
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
def swath_def_2d_xarray_dask_antimeridian():
    """Create a SwathDefinition with DataArrays(dask) arrays (200, 1500) over the antimeridian.

    Longitude values go from positive values to negative values as they cross -180/180.

    """
    lons, lats = _antimeridian_lonlats()
    lons = xr.DataArray(lons, dims=("y", "x"))
    lats = xr.DataArray(lats, dims=("y", "x"))
    return SwathDefinition(lons, lats)


@pytest.fixture(scope="session")
def area_def_lcc_conus_1km():
    """Create an AreaDefinition with an LCC projection over CONUS (1500, 2000)."""
    proj_str = "+proj=lcc +lon_0=-95 +lat_1=35.0 +lat_2=35.0 +datum=WGS84 +no_defs"
    crs = CRS.from_string(proj_str)
    area_def = AreaDefinition(crs, (SRC_AREA_SHAPE[0], SRC_AREA_SHAPE[1]),
                              (-750000, -750000, 750000, 750000))
    return area_def


@pytest.fixture(scope="session")
def area_def_stere_source():
    """Create an AreaDefinition with a polar-stereographic projection (10, 50).

    This area is the same shape as input swath definitions.

    """
    return AreaDefinition(
        {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '52.00',
            'lat_ts': '52.00',
            'lon_0': '5.00',
            'proj': 'stere'
        },
        (SRC_AREA_SHAPE[0], SRC_AREA_SHAPE[1]),
        (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001),
    )


@pytest.fixture(scope="session")
def area_def_stere_target():
    """Create an AreaDefinition with a polar-stereographic projection (800, 850)."""
    return AreaDefinition(
        {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '50.00',
            'lat_ts': '50.00',
            'lon_0': '8.00',
            'proj': 'stere'
        },
        (DST_AREA_SHAPE[0], DST_AREA_SHAPE[1]),
        (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001)
    )


@pytest.fixture(scope="session")
def area_def_lonlat_pm180_target():
    """Create an AreaDefinition with a geographic lon/lat projection with prime meridian at 180 (800, 850)."""
    return AreaDefinition(
        {
            'proj': 'longlat',
            'pm': '180.0',
            'datum': 'WGS84',
            'no_defs': None,
        },
        (DST_AREA_SHAPE[0], DST_AREA_SHAPE[1]),
        (-20.0, 20.0, 20.0, 35.0)
    )


@pytest.fixture(scope="session")
def coord_def_2d_float32_dask():
    """Create a 2D CoordinateDefinition of dask arrays (4, 3)."""
    chunks = 5
    lons = da.from_array(np.array([
        [11.5, 12.562036, 12.9],
        [11.5, 12.562036, 12.9],
        [11.5, 12.562036, 12.9],
        [11.5, 12.562036, 12.9],
    ], dtype=np.float32), chunks=chunks)
    lats = da.from_array(np.array([
        [55.715613, 55.715613, 55.715613],
        [55.715613, 55.715613, 55.715613],
        [55.715613, np.nan, 55.715613],
        [55.715613, 55.715613, 55.715613],
    ], dtype=np.float32), chunks=chunks)
    return CoordinateDefinition(lons=lons, lats=lats)


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
    return SwathDefinition(lons=tlons_1d, lats=tlats_1d)


# Input data arrays

@pytest.fixture(scope="session")
def data_1d_float32_xarray_dask():
    """Create a sample 1D data DataArray(dask) (3,)."""
    return xr.DataArray(
        da.from_array(np.array([1., 2., 3.], dtype=np.float32), chunks=5), dims=('my_dim1',))


@pytest.fixture(scope="session")
def data_2d_float32_numpy():
    """Create a sample 2D data numpy array (50, 10)."""
    return np.fromfunction(lambda y, x: y * x, SRC_SWATH_2D_SHAPE, dtype=np.float32)


@pytest.fixture(scope="session")
def data_2d_float32_dask(data_2d_float32_numpy):
    """Create a sample 2D data numpy array (50, 10)."""
    return da.from_array(data_2d_float32_numpy, chunks=5)


@pytest.fixture(scope="session")
def data_2d_float32_xarray_numpy(data_2d_float32_numpy):
    """Create a sample 2D data DataArray(numpy) (50, 10)."""
    return xr.DataArray(data_2d_float32_numpy, dims=('y', 'x'))


@pytest.fixture(scope="session")
def data_2d_float32_xarray_dask(data_2d_float32_dask):
    """Create a sample 2D data DataArray(dask) (50, 10)."""
    return xr.DataArray(data_2d_float32_dask, dims=('y', 'x'))


@pytest.fixture(scope="session")
def data_3d_float32_xarray_dask():
    """Create a sample 3D data DataArray(dask) (50, 10, 3)."""
    return xr.DataArray(
        da.from_array(np.fromfunction(lambda y, x, b: y * x * b, (50, 10, 3), dtype=np.float32),
                      chunks=5),
        dims=('y', 'x', 'bands'),
        coords={'bands': ['r', 'g', 'b']})
