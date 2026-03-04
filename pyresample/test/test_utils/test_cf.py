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
"""Tests for CF helpers."""

import os
import unittest

import numpy as np
import pytest

import pyresample
from pyresample.test.utils import TEST_FILES_PATH, assert_future_geometry
from pyresample.utils import load_cf_area


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

    def test_load_cf_axis_without_units(self):
        cf_file = _prepare_cf_nh10km()
        del cf_file['xc'].attrs['units']
        del cf_file['yc'].attrs['units']

        _, cf_info = load_cf_area(cf_file, variable='ice_conc')

        assert cf_info['x']['unit'] is None
        assert cf_info['y']['unit'] is None

    def test_load_cf_axis_with_non_string_units(self):
        cf_file = _prepare_cf_nh10km()
        cf_file['xc'].attrs['units'] = 1
        cf_file['yc'].attrs['units'] = 1

        _, cf_info = load_cf_area(cf_file, variable='ice_conc')

        assert cf_info['x']['unit'] is None
        assert cf_info['y']['unit'] is None


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
