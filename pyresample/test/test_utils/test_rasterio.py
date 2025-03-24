# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Pyresample developers
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
"""Tests for rasterio helpers."""

import uuid
from unittest import mock

import numpy as np
import pytest
from pyproj import CRS

import pyresample
from pyresample.test.utils import assert_future_geometry


def tmptiff(width=100, height=100, transform=None, crs=None, dtype=np.uint8):
    """Create a temporary in-memory TIFF file of all ones."""
    import rasterio
    array = np.ones((width, height)).astype(dtype)
    fname = '/vsimem/%s' % uuid.uuid4()
    with rasterio.open(fname, 'w', driver='GTiff', count=1, transform=transform,
                       width=width, height=height, crs=crs, dtype=dtype) as dst:
        dst.write(array, 1)
    return fname


class TestFromRasterio:
    """Test loading geometries from rasterio datasets."""

    def test_get_area_def_from_raster(self):
        from affine import Affine
        from rasterio.crs import CRS as RCRS

        from pyresample import utils
        x_size = 791
        y_size = 718
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        crs = RCRS(init='epsg:3857')
        source = tmptiff(x_size, y_size, transform, crs=crs)
        area_id = 'area_id'
        proj_id = 'proj_id'
        description = 'name'
        area_def = utils.rasterio.get_area_def_from_raster(
            source, area_id=area_id, name=description, proj_id=proj_id)
        assert area_def.area_id == area_id
        assert area_def.proj_id == proj_id
        assert area_def.description == description
        assert area_def.width == x_size
        assert area_def.height == y_size
        assert crs == area_def.crs
        assert area_def.area_extent == (
            transform.c, transform.f + transform.e * y_size,
            transform.c + transform.a * x_size, transform.f)

    def test_get_area_def_from_raster_extracts_proj_id(self):
        from rasterio.crs import CRS as RCRS

        from pyresample import utils
        crs = RCRS(init='epsg:3857')
        source = tmptiff(crs=crs)
        area_def = utils.rasterio.get_area_def_from_raster(source)
        epsg3857_names = (
            'WGS_1984_Web_Mercator_Auxiliary_Sphere',  # gdal>=3.0 + proj>=6.0
            'WGS 84 / Pseudo-Mercator',                # proj<6.0
        )
        assert area_def.proj_id in epsg3857_names

    @pytest.mark.parametrize("x_rotation", [0.0, 0.1])
    def test_get_area_def_from_raster_non_georef_value_err(self, x_rotation):
        from affine import Affine

        from pyresample import utils
        transform = Affine(300.0379266750948, x_rotation, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        with pytest.raises(ValueError):
            utils.rasterio.get_area_def_from_raster(source)

    @pytest.mark.parametrize("future_geometries", [False, True])
    def test_get_area_def_from_raster_non_georef_respects_proj_dict(
            self,
            future_geometries,
            _mock_rasterio_with_importerror
    ):
        from affine import Affine

        from pyresample import utils
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        with pyresample.config.set({"features.future_geometries": future_geometries}):
            area_def = utils.rasterio.get_area_def_from_raster(source, projection="EPSG:3857")
        assert_future_geometry(area_def, future_geometries)
        assert area_def.crs == CRS(3857)


@pytest.fixture(params=[False, True])
def _mock_rasterio_with_importerror(request):
    """Mock rasterio importing so it isn't available and GDAL is used."""
    if not request.param:
        yield None
        return
    try:
        from osgeo import gdal
    except ImportError:
        # GDAL isn't available at all
        pytest.skip("'gdal' not available for testing")

    with mock.patch("pyresample.utils.rasterio._import_raster_libs") as imp_func:
        imp_func.return_value = (None, gdal)
        yield imp_func
