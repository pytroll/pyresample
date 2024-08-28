#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015-2021 Pyresample developers
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
"""Test various utility functions."""

import os
import unittest
import uuid
from timeit import timeit
from unittest import mock

import numpy as np
import pyproj
import pytest
from pyproj import CRS

import pyresample
from pyresample.test.utils import TEST_FILES_PATH, assert_future_geometry, create_test_latitude, create_test_longitude
from pyresample.utils import load_cf_area
from pyresample.utils.row_appendable_array import RowAppendableArray


def tmptiff(width=100, height=100, transform=None, crs=None, dtype=np.uint8):
    """Create a temporary in-memory TIFF file of all ones."""
    import rasterio
    array = np.ones((width, height)).astype(dtype)
    fname = '/vsimem/%s' % uuid.uuid4()
    with rasterio.open(fname, 'w', driver='GTiff', count=1, transform=transform,
                       width=width, height=height, crs=crs, dtype=dtype) as dst:
        dst.write(array, 1)
    return fname


class TestPreprocessing(unittest.TestCase):
    """Tests for index generating functions."""

    def test_nearest_neighbor_area_area(self):
        from pyresample import geometry, utils
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)

        extents2 = [-1000, -1000, 1000. * 4000, 1000. * 4000]
        area_def2 = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                            proj_dict, 600, 700, extents2)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, area_def2, 12000.)

    def test_nearest_neighbor_area_grid(self):
        from pyresample import geometry, utils
        lon_arr = create_test_longitude(-94.9, -90.0, (50, 100), dtype=np.float64)
        lat_arr = create_test_latitude(25.1, 30.0, (50, 100), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)

        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, grid, 12000.)

    def test_nearest_neighbor_grid_area(self):
        from pyresample import geometry, utils
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 2500., 1000. * 2000.]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 40, 50, extents)

        lon_arr = create_test_longitude(-100.0, -60.0, (550, 500), dtype=np.float64)
        lat_arr = create_test_latitude(20.0, 45.0, (550, 500), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(grid, area_def, 12000.)

    def test_nearest_neighbor_grid_grid(self):
        from pyresample import geometry, utils
        lon_arr = create_test_longitude(-95.0, -85.0, (40, 50), dtype=np.float64)
        lat_arr = create_test_latitude(25.0, 35.0, (40, 50), dtype=np.float64)
        grid_dst = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)

        lon_arr = create_test_longitude(-100.0, -80.0, (400, 500), dtype=np.float64)
        lat_arr = create_test_latitude(20.0, 40.0, (400, 500), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(grid, grid_dst, 12000.)


class TestMisc(unittest.TestCase):
    """Test miscellaneous utilities."""

    def test_wrap_longitudes(self):
        # test that we indeed wrap to [-180:+180[
        from pyresample import utils
        step = 60
        lons = np.arange(-360, 360 + step, step)
        self.assertTrue(
            (lons.min() < -180) and (lons.max() >= 180) and (+180 in lons))
        wlons = utils.wrap_longitudes(lons)
        self.assertFalse(
            (wlons.min() < -180) or (wlons.max() >= 180) or (+180 in wlons))

    def test_wrap_and_check(self):
        from pyresample import utils

        lons1 = np.arange(-135., +135, 50.)
        lats = np.ones_like(lons1) * 70.
        new_lons, new_lats = utils.check_and_wrap(lons1, lats)
        self.assertIs(lats, new_lats)
        self.assertTrue(np.isclose(lons1, new_lons).all())

        lons2 = np.where(lons1 < 0, lons1 + 360, lons1)
        new_lons, new_lats = utils.check_and_wrap(lons2, lats)
        self.assertIs(lats, new_lats)
        # after wrapping lons2 should look like lons1
        self.assertTrue(np.isclose(lons1, new_lons).all())

        lats2 = lats + 25.
        self.assertRaises(ValueError, utils.check_and_wrap, lons1, lats2)

    def test_unicode_proj4_string(self):
        """Test that unicode is accepted for area creation."""
        from pyresample import get_area_def
        get_area_def(u"eurol", u"eurol", u"bla",
                     u'+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45',
                     1000, 1000, (-1000, -1000, 1000, 1000))

    def test_proj4_radius_parameters_provided(self):
        """Test proj4_radius_parameters with a/b."""
        from pyresample import utils
        a, b = utils.proj4.proj4_radius_parameters(
            '+proj=stere +a=6378273 +b=6356889.44891',
        )
        np.testing.assert_almost_equal(a, 6378273)
        np.testing.assert_almost_equal(b, 6356889.44891)

    def test_proj4_radius_parameters_ellps(self):
        """Test proj4_radius_parameters with ellps."""
        from pyresample import utils
        a, b = utils.proj4.proj4_radius_parameters(
            '+proj=stere +ellps=WGS84',
        )
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_radius_parameters_default(self):
        """Test proj4_radius_parameters with default parameters."""
        from pyresample import utils
        a, b = utils.proj4.proj4_radius_parameters(
            '+proj=lcc +lat_0=10 +lat_1=10',
        )
        # WGS84
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_radius_parameters_spherical(self):
        """Test proj4_radius_parameters in case of a spherical earth."""
        from pyresample import utils
        a, b = utils.proj4.proj4_radius_parameters(
            '+proj=stere +R=6378273',
        )
        np.testing.assert_almost_equal(a, 6378273.)
        np.testing.assert_almost_equal(b, 6378273.)

    def test_convert_proj_floats(self):
        from collections import OrderedDict

        import pyresample.utils as utils

        pairs = [('proj', 'lcc'), ('ellps', 'WGS84'), ('lon_0', '-95'), ('no_defs', True)]
        expected = OrderedDict([('proj', 'lcc'), ('ellps', 'WGS84'), ('lon_0', -95.0), ('no_defs', True)])
        self.assertDictEqual(utils.proj4.convert_proj_floats(pairs), expected)

        # EPSG
        pairs = [('init', 'EPSG:4326'), ('EPSG', 4326)]
        for pair in pairs:
            expected = OrderedDict([pair])
            self.assertDictEqual(utils.proj4.convert_proj_floats([pair]), expected)

    def test_proj4_str_dict_conversion(self):
        from pyresample import utils

        proj_str = "+proj=lcc +ellps=WGS84 +lon_0=-95 +lat_1=25.5 +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
        proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
        self.assertDictEqual(proj_dict, proj_dict2)
        self.assertIsInstance(proj_dict['lon_0'], (float, int))
        self.assertIsInstance(proj_dict2['lon_0'], (float, int))
        self.assertIsInstance(proj_dict['lat_1'], float)
        self.assertIsInstance(proj_dict2['lat_1'], float)

        # EPSG
        proj_str = '+init=EPSG:4326'
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
        proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
        # pyproj usually expands EPSG definitions so we can only round trip
        self.assertEqual(proj_dict, proj_dict2)

        proj_str = 'EPSG:4326'
        proj_dict_exp2 = {'proj': 'longlat', 'datum': 'WGS84', 'no_defs': None, 'type': 'crs'}
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        self.assertEqual(proj_dict, proj_dict_exp2)
        # input != output for this style of EPSG code
        # EPSG to PROJ.4 can be lossy
        # self.assertEqual(utils._proj4.proj4_dict_to_str(proj_dict), proj_str)  # round-trip

    def test_proj4_str_dict_conversion_with_valueless_parameter(self):
        from pyresample import utils

        # Value-less south parameter
        proj_str = "+ellps=WGS84 +no_defs +proj=utm +south +type=crs +units=m +zone=54"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
        proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
        self.assertDictEqual(proj_dict, proj_dict2)

    @pytest.mark.skipif(pyproj.__proj_version__ == "9.3.0", reason="Bug in PROJ causes inequality in EPSG comparison")
    def test_def2yaml_converter(self):
        import tempfile

        from pyresample import convert_def_to_yaml, parse_area_file
        def_file = os.path.join(TEST_FILES_PATH, 'areas.cfg')
        filehandle, yaml_file = tempfile.mkstemp()
        os.close(filehandle)
        try:
            convert_def_to_yaml(def_file, yaml_file)
            areas_new = set(parse_area_file(yaml_file))
            areas = parse_area_file(def_file)
            areas_old = set(areas)
            areas_new = {area.area_id: area for area in areas_new}
            areas_old = {area.area_id: area for area in areas_old}
            self.assertEqual(areas_new, areas_old)
        finally:
            os.remove(yaml_file)


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


def _prepare_cf_nh10km():
    import xarray as xr
    nx = 760
    ny = 1120
    ds = xr.Dataset({'ice_conc': (('time', 'yc', 'xc'), np.ma.masked_all((1, ny, nx)),
                                  {'grid_mapping': 'Polar_Stereographic_Grid'}),
                     'xc': ('xc', np.linspace(-3845, 3745, num=nx),
                            {'standard_name': 'projection_x_coordinate', 'units': 'km'}),
                     'yc': ('yc', np.linspace(+5845, -5345, num=ny),
                            {'standard_name': 'projection_y_coordinate', 'units': 'km'})},
                    coords={'lat': (('yc', 'xc'), np.ma.masked_all((ny, nx))),
                            'lon': (('yc', 'xc'), np.ma.masked_all((ny, nx)))},)
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lat'].attrs['standard_name'] = 'latitude'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lon'].attrs['standard_name'] = 'longitude'

    ds['Polar_Stereographic_Grid'] = 0
    ds['Polar_Stereographic_Grid'].attrs['grid_mapping_name'] = "polar_stereographic"
    ds['Polar_Stereographic_Grid'].attrs['false_easting'] = 0.
    ds['Polar_Stereographic_Grid'].attrs['false_northing'] = 0.
    ds['Polar_Stereographic_Grid'].attrs['semi_major_axis'] = 6378273.
    ds['Polar_Stereographic_Grid'].attrs['semi_minor_axis'] = 6356889.44891
    ds['Polar_Stereographic_Grid'].attrs['straight_vertical_longitude_from_pole'] = -45.
    ds['Polar_Stereographic_Grid'].attrs['latitude_of_projection_origin'] = 90.
    ds['Polar_Stereographic_Grid'].attrs['standard_parallel'] = 70.

    return ds


def _prepare_cf_goes():
    import xarray as xr

    from pyresample.geometry import AreaDefinition
    area_id = 'GOES-East'
    description = '2km at nadir'
    proj_id = 'abi_fixed_grid'
    h = 35786023
    projection = {'ellps': 'GRS80', 'h': h, 'lon_0': '-75', 'no_defs': 'None',
                  'proj': 'geos', 'sweep': 'x', 'type': 'crs',
                  'units': 'm', 'x_0': '0', 'y_0': '0'}
    width = 2500
    height = 1500
    area_extent = (-3627271.2913 / h, 1583173.6575 / h, 1382771.9287 / h, 4589199.5895 / h)
    goes_area = AreaDefinition(area_id, description, proj_id, projection,
                               width, height, area_extent)
    x = np.linspace(goes_area.area_extent[0], goes_area.area_extent[2], goes_area.shape[1])
    y = np.linspace(goes_area.area_extent[3], goes_area.area_extent[1], goes_area.shape[0])
    ds = xr.Dataset({'C13': (('y', 'x'), np.ma.masked_all((height, width)),
                             {'grid_mapping': 'GOES-East'})},
                    coords={'y': y, 'x': x})

    ds['x'].attrs['units'] = 'radians'
    ds['x'].attrs['standard_name'] = 'projection_x_coordinate'
    ds['y'].attrs['units'] = 'radians'
    ds['y'].attrs['standard_name'] = 'projection_y_coordinate'

    ds['GOES-East'] = 0
    ds['GOES-East'].attrs['grid_mapping_name'] = 'geostationary'
    ds['GOES-East'].attrs['false_easting'] = 0.0
    ds['GOES-East'].attrs['false_northing'] = 0.0
    ds['GOES-East'].attrs['semi_major_axis'] = 6378137.0
    ds['GOES-East'].attrs['semi_minor_axis'] = 6356752.31414
    ds['GOES-East'].attrs['geographic_crs_name'] = 'unknown'
    ds['GOES-East'].attrs['horizontal_datum_name'] = 'unknown'
    ds['GOES-East'].attrs['inverse_flattening'] = 298.257222096042
    ds['GOES-East'].attrs['latitude_of_projection_origin'] = 0.0
    ds['GOES-East'].attrs['long_name'] = 'GOES-East'
    ds['GOES-East'].attrs['longitude_of_prime_meridian'] = 0.0
    ds['GOES-East'].attrs['longitude_of_projection_origin'] = -75.0
    ds['GOES-East'].attrs['perspective_point_height'] = 35786023.0
    ds['GOES-East'].attrs['prime_meridian_name'] = 'Greenwich'
    ds['GOES-East'].attrs['projected_crs_name'] = 'unknown'
    ds['GOES-East'].attrs['reference_ellipsoid_name'] = 'GRS 1980'
    ds['GOES-East'].attrs['sweep_angle_axis'] = 'x'

    return ds


def _prepare_cf_llwgs84():
    import xarray as xr
    nlat = 19
    nlon = 37
    ds = xr.Dataset({'temp': (('lat', 'lon'), np.ma.masked_all((nlat, nlon)), {'grid_mapping': 'crs'})},
                    coords={'lat': np.linspace(-90., +90., num=nlat),
                            'lon': np.linspace(-180., +180., num=nlon)})
    ds['lat'].attrs['units'] = 'degreesN'
    ds['lat'].attrs['standard_name'] = 'latitude'
    ds['lon'].attrs['units'] = 'degreesE'
    ds['lon'].attrs['standard_name'] = 'longitude'

    ds['crs'] = 0
    ds['crs'].attrs['grid_mapping_name'] = "latitude_longitude"
    ds['crs'].attrs['longitude_of_prime_meridian'] = 0.
    ds['crs'].attrs['semi_major_axis'] = 6378137.
    ds['crs'].attrs['inverse_flattening'] = 298.257223563

    return ds


def _prepare_cf_llnocrs():
    import xarray as xr
    nlat = 19
    nlon = 37
    ds = xr.Dataset({'temp': (('lat', 'lon'), np.ma.masked_all((nlat, nlon)))},
                    coords={'lat': np.linspace(-90., +90., num=nlat),
                            'lon': np.linspace(-180., +180., num=nlon)})
    ds['lat'].attrs['units'] = 'degreeN'
    ds['lat'].attrs['standard_name'] = 'latitude'
    ds['lon'].attrs['units'] = 'degreeE'
    ds['lon'].attrs['standard_name'] = 'longitude'

    return ds


class TestLoadCFAreaPublic:
    """Test public API load_cf_area() for loading an AreaDefinition from netCDF/CF files."""

    def test_load_cf_no_exist(self):
        cf_file = os.path.join(TEST_FILES_PATH, 'does_not_exist.nc')
        with pytest.raises(FileNotFoundError):
            load_cf_area(cf_file)

    def test_load_cf_from_not_nc(self):
        cf_file = os.path.join(TEST_FILES_PATH, 'areas.yaml')
        with pytest.raises((ValueError, OSError)):
            load_cf_area(cf_file)

    @pytest.mark.parametrize(
        ("exc_type", "kwargs"),
        [
            (KeyError, {"variable": "doesNotExist"}),
            (ValueError, {"variable": "Polar_Stereographic_Grid"}),
            (KeyError, {"variable": "Polar_Stereographic_Grid", "y": "doesNotExist", "x": "xc"}),
            (ValueError, {"variable": "Polar_Stereographic_Grid", "y": "time", "x": "xc"}),
            (ValueError, {"variable": "lat"}),
        ]
    )
    def test_load_cf_parameters_errors(self, exc_type, kwargs):
        cf_file = _prepare_cf_nh10km()
        with pytest.raises(exc_type):
            load_cf_area(cf_file, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"variable": "Polar_Stereographic_Grid", "y": "yc", "x": "xc"},
            {"variable": "ice_conc"},
            {},
        ]
    )
    def test_load_cf_nh10km(self, kwargs):
        cf_file = _prepare_cf_nh10km()
        adef, _ = load_cf_area(cf_file, **kwargs)
        assert adef.shape == (1120, 760)
        xc = adef.projection_x_coords
        yc = adef.projection_y_coords
        assert xc[0] == -3845000.0, "Wrong x axis (index 0)"
        assert xc[1] == xc[0] + 10000.0, "Wrong x axis (index 1)"
        assert yc[0] == 5845000.0, "Wrong y axis (index 0)"
        assert yc[1] == yc[0] - 10000.0, "Wrong y axis (index 1)"

    @pytest.mark.parametrize(
        ("kwargs", "exp_var", "exp_lat", "exp_lon"),
        [
            ({"variable": "Polar_Stereographic_Grid", "y": "yc", "x": "xc"}, "Polar_Stereographic_Grid", None, None),
            ({"variable": "ice_conc"}, "ice_conc", "lat", "lon"),
            ({}, "ice_conc", "lat", "lon"),
        ]
    )
    def test_load_cf_nh10km_cfinfo(self, kwargs, exp_var, exp_lat, exp_lon):
        cf_file = _prepare_cf_nh10km()
        _, cf_info = load_cf_area(cf_file, **kwargs)
        assert cf_info['variable'] == exp_var
        assert cf_info['grid_mapping_variable'] == 'Polar_Stereographic_Grid'
        assert cf_info['type_of_grid_mapping'] == 'polar_stereographic'
        assert cf_info['lon'] == exp_lon
        assert cf_info['lat'] == exp_lat
        assert cf_info['x']['varname'] == 'xc'
        assert cf_info['x']['first'] == -3845.0
        assert cf_info['y']['varname'] == 'yc'
        assert cf_info['y']['last'] == -5345.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"variable": "C13"},
            {},
        ])
    def test_load_cf_goes(self, kwargs):
        cf_file = _prepare_cf_goes()
        adef, cf_info = load_cf_area(cf_file, **kwargs)
        assert cf_info['grid_mapping_variable'] == 'GOES-East'
        assert cf_info['type_of_grid_mapping'] == 'geostationary'
        assert cf_info['x']['varname'] == 'x'
        assert cf_info['x']['first'] == -3627271.2913
        assert cf_info['y']['varname'] == 'y'
        assert cf_info['y']['last'] == 1583173.6575

    @pytest.mark.parametrize(
        ("file_func", "kwargs", "exp_lat", "exp_lon"),
        [
            (_prepare_cf_llwgs84, {"variable": "crs", "y": "lat", "x": "lon"}, None, None),
            (_prepare_cf_llwgs84, {"variable": "temp"}, "lat", "lon"),
            (_prepare_cf_llwgs84, {}, "lat", "lon"),
            (_prepare_cf_llnocrs, {"variable": "temp"}, "lat", "lon"),
            (_prepare_cf_llnocrs, {}, "lat", "lon"),
        ]
    )
    @pytest.mark.parametrize("future_geometries", [False, True])
    def test_load_cf_latlon(self, file_func, kwargs, exp_lat, exp_lon, future_geometries):
        cf_file = file_func()
        with pyresample.config.set({"features.future_geometries": future_geometries}):
            adef, cf_info = load_cf_area(cf_file, **kwargs)
        _validate_lonlat_cf_area(adef, cf_info, exp_lon, exp_lat)
        assert_future_geometry(adef, future_geometries)


def _validate_lonlat_cf_area(adef, cf_info, exp_lon, exp_lat):
    assert adef.shape == (19, 37)
    xc = adef.projection_x_coords
    yc = adef.projection_y_coords
    assert xc[0] == -180., "Wrong x axis (index 0)"
    assert xc[1] == -180. + 10.0, "Wrong x axis (index 1)"
    assert yc[0] == -90., "Wrong y axis (index 0)"
    assert yc[1] == -90. + 10.0, "Wrong y axis (index 1)"
    assert cf_info['lon'] == exp_lon
    assert cf_info['lat'] == exp_lat
    assert cf_info['type_of_grid_mapping'] == 'latitude_longitude'
    assert cf_info['x']['varname'] == 'lon'
    assert cf_info['x']['first'] == -180.
    assert cf_info['y']['varname'] == 'lat'
    assert cf_info['y']['first'] == -90.


class TestLoadCFArea_Private(unittest.TestCase):
    """Test private routines involved in loading an AreaDefinition from netCDF/CF files."""

    def setUp(self):
        """Prepare nc_handles."""
        self.nc_handles = {}
        self.nc_handles['nh10km'] = _prepare_cf_nh10km()
        self.nc_handles['llwgs84'] = _prepare_cf_llwgs84()
        self.nc_handles['llnocrs'] = _prepare_cf_llnocrs()

    def test_cf_guess_lonlat(self):
        from pyresample.utils.cf import _guess_cf_lonlat_varname

        # nominal
        self.assertEqual(_guess_cf_lonlat_varname(self.nc_handles['nh10km'], 'ice_conc', 'lat'), 'lat',)
        self.assertEqual(_guess_cf_lonlat_varname(self.nc_handles['nh10km'], 'ice_conc', 'lon'), 'lon',)
        self.assertEqual(_guess_cf_lonlat_varname(self.nc_handles['llwgs84'], 'temp', 'lat'), 'lat')
        self.assertEqual(_guess_cf_lonlat_varname(self.nc_handles['llwgs84'], 'temp', 'lon'), 'lon')
        self.assertEqual(_guess_cf_lonlat_varname(self.nc_handles['llnocrs'], 'temp', 'lat'), 'lat')
        self.assertEqual(_guess_cf_lonlat_varname(self.nc_handles['llnocrs'], 'temp', 'lon'), 'lon')

        # error cases
        self.assertRaises(ValueError, _guess_cf_lonlat_varname, self.nc_handles['nh10km'], 'ice_conc', 'wrong',)
        self.assertRaises(KeyError, _guess_cf_lonlat_varname, self.nc_handles['nh10km'], 'doesNotExist', 'lat',)

    def test_cf_guess_axis_varname(self):
        from pyresample.utils.cf import _guess_cf_axis_varname

        # nominal
        self.assertEqual(_guess_cf_axis_varname(
            self.nc_handles['nh10km'], 'ice_conc', 'x', 'polar_stereographic'), 'xc')
        self.assertEqual(_guess_cf_axis_varname(
            self.nc_handles['nh10km'], 'ice_conc', 'y', 'polar_stereographic'), 'yc')
        self.assertEqual(_guess_cf_axis_varname(self.nc_handles['llwgs84'], 'temp', 'x', 'latitude_longitude'), 'lon')
        self.assertEqual(_guess_cf_axis_varname(self.nc_handles['llwgs84'], 'temp', 'y', 'latitude_longitude'), 'lat')

        # error cases
        self.assertRaises(ValueError, _guess_cf_axis_varname,
                          self.nc_handles['nh10km'], 'ice_conc', 'wrong', 'polar_stereographic')
        self.assertRaises(KeyError, _guess_cf_axis_varname,
                          self.nc_handles['nh10km'], 'doesNotExist', 'x', 'polar_stereographic')

    def test_cf_is_valid_coordinate_standardname(self):
        from pyresample.utils.cf import _is_valid_coordinate_standardname, _valid_cf_type_of_grid_mapping

        # nominal
        for proj_type in _valid_cf_type_of_grid_mapping:
            if proj_type == 'geostationary':
                self.assertTrue(_is_valid_coordinate_standardname('projection_x_angular_coordinate', 'x', proj_type))
                self.assertTrue(_is_valid_coordinate_standardname('projection_y_angular_coordinate', 'y', proj_type))
                self.assertTrue(_is_valid_coordinate_standardname('projection_x_coordinate', 'x', proj_type))
                self.assertTrue(_is_valid_coordinate_standardname('projection_y_coordinate', 'y', proj_type))
            elif proj_type == 'latitude_longitude':
                self.assertTrue(_is_valid_coordinate_standardname('longitude', 'x', proj_type))
                self.assertTrue(_is_valid_coordinate_standardname('latitude', 'y', proj_type))
            elif proj_type == 'rotated_latitude_longitude':
                self.assertTrue(_is_valid_coordinate_standardname('grid_longitude', 'x', proj_type))
                self.assertTrue(_is_valid_coordinate_standardname('grid_latitude', 'y', proj_type))
            else:
                self.assertTrue(_is_valid_coordinate_standardname('projection_x_coordinate', 'x', 'default'))
                self.assertTrue(_is_valid_coordinate_standardname('projection_y_coordinate', 'y', 'default'))

        # error cases
        self.assertRaises(ValueError, _is_valid_coordinate_standardname, 'projection_x_coordinate', 'x', 'wrong')
        self.assertRaises(ValueError, _is_valid_coordinate_standardname, 'projection_y_coordinate', 'y', 'also_wrong')

    def test_cf_is_valid_coordinate_variable(self):
        from pyresample.utils.cf import _is_valid_coordinate_variable

        # nominal
        self.assertTrue(_is_valid_coordinate_variable(self.nc_handles['nh10km'], 'xc', 'x', 'polar_stereographic'))
        self.assertTrue(_is_valid_coordinate_variable(self.nc_handles['nh10km'], 'yc', 'y', 'polar_stereographic'))
        self.assertTrue(_is_valid_coordinate_variable(self.nc_handles['llwgs84'], 'lon', 'x', 'latitude_longitude'))
        self.assertTrue(_is_valid_coordinate_variable(self.nc_handles['llwgs84'], 'lat', 'y', 'latitude_longitude'))
        self.assertTrue(_is_valid_coordinate_variable(self.nc_handles['llnocrs'], 'lon', 'x', 'latitude_longitude'))
        self.assertTrue(_is_valid_coordinate_variable(self.nc_handles['llnocrs'], 'lat', 'y', 'latitude_longitude'))

        # error cases
        self.assertRaises(KeyError, _is_valid_coordinate_variable,
                          self.nc_handles['nh10km'], 'doesNotExist', 'x', 'polar_stereographic')
        self.assertRaises(ValueError, _is_valid_coordinate_variable,
                          self.nc_handles['nh10km'], 'xc', 'wrong', 'polar_stereographic')
        self.assertRaises(ValueError, _is_valid_coordinate_variable,
                          self.nc_handles['nh10km'], 'xc', 'x', 'wrong')

    def test_cf_load_crs_from_cf_gridmapping(self):
        from pyresample.utils.cf import _load_crs_from_cf_gridmapping

        def validate_crs_nh10km(crs):
            crs_dict = crs.to_dict()
            self.assertEqual(crs_dict['proj'], 'stere')
            self.assertEqual(crs_dict['lat_0'], 90.)

        def validate_crs_llwgs84(crs):
            crs_dict = crs.to_dict()
            self.assertEqual(crs_dict['proj'], 'longlat')
            self.assertEqual(crs_dict['ellps'], 'WGS84')

        crs = _load_crs_from_cf_gridmapping(self.nc_handles['nh10km'], 'Polar_Stereographic_Grid')
        validate_crs_nh10km(crs)
        crs = _load_crs_from_cf_gridmapping(self.nc_handles['llwgs84'], 'crs')
        validate_crs_llwgs84(crs)


def test_check_slice_orientation():
    """Test that slicing fix is doing what it should."""
    from pyresample.utils import check_slice_orientation

    # Forward slicing should not be changed
    start, stop, step = 0, 10, None
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res is slice_in

    # Reverse slicing should not be changed if the step is negative
    start, stop, step = 10, 0, -1
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res is slice_in

    # Reverse slicing should be fixed if step is positive
    start, stop, step = 10, 0, 2
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res == slice(start, stop, -step)

    # Reverse slicing should be fixed if step is None
    start, stop, step = 10, 0, None
    slice_in = slice(start, stop, step)
    res = check_slice_orientation(slice_in)
    assert res == slice(start, stop, -1)


class TestRowAppendableArray(unittest.TestCase):
    """Test appending numpy arrays to possible pre-allocated buffer."""

    def test_append_1d_arrays_and_trim_remaining_buffer(self):
        appendable = RowAppendableArray(7)
        appendable.append_row(np.zeros(3))
        appendable.append_row(np.ones(3))
        self.assertTrue(np.array_equal(appendable.to_array(), np.array([0, 0, 0, 1, 1, 1])))

    def test_append_rows_of_nd_arrays_and_trim_remaining_buffer(self):
        appendable = RowAppendableArray(7)
        appendable.append_row(np.zeros((3, 2)))
        appendable.append_row(np.ones((3, 2)))
        self.assertTrue(np.array_equal(appendable.to_array(), np.vstack([np.zeros((3, 2)), np.ones((3, 2))])))

    def test_append_more_1d_arrays_than_expected(self):
        appendable = RowAppendableArray(5)
        appendable.append_row(np.zeros(3))
        appendable.append_row(np.ones(3))
        self.assertTrue(np.array_equal(appendable.to_array(), np.array([0, 0, 0, 1, 1, 1])))

    def test_append_more_rows_of_nd_arrays_than_expected(self):
        appendable = RowAppendableArray(2)
        appendable.append_row(np.zeros((3, 2)))
        appendable.append_row(np.ones((3, 2)))
        self.assertTrue(np.array_equal(appendable.to_array(), np.vstack([np.zeros((3, 2)), np.ones((3, 2))])))

    def test_append_1d_arrays_pre_allocated_appendable_array(self):
        appendable = RowAppendableArray(6)
        appendable.append_row(np.zeros(3))
        appendable.append_row(np.ones(3))
        self.assertTrue(np.array_equal(appendable.to_array(), np.array([0, 0, 0, 1, 1, 1])))

    def test_append_rows_of_nd_arrays_to_pre_allocated_appendable_array(self):
        appendable = RowAppendableArray(6)
        appendable.append_row(np.zeros((3, 2)))
        appendable.append_row(np.ones((3, 2)))
        self.assertTrue(np.array_equal(appendable.to_array(), np.vstack([np.zeros((3, 2)), np.ones((3, 2))])))

    def test_pre_allocation_can_double_appending_performance(self):
        unallocated = RowAppendableArray(0)
        pre_allocated = RowAppendableArray(10000)

        unallocated_performance = timeit(lambda: unallocated.append_row(np.array([42])), number=10000)
        pre_allocated_performance = timeit(lambda: pre_allocated.append_row(np.array([42])), number=10000)
        self.assertGreater(unallocated_performance / pre_allocated_performance, 2)
