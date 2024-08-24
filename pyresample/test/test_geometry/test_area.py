# Copyright (C) 2010-2022 Pyresample developers
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
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test AreaDefinition objects."""
import io
import sys
from glob import glob
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Proj

import pyresample
from pyresample import geo_filter, parse_area_file
from pyresample.future.geometry import AreaDefinition, SwathDefinition
from pyresample.future.geometry.area import ignore_pyproj_proj_warnings
from pyresample.future.geometry.base import get_array_hashable
from pyresample.geometry import AreaDefinition as LegacyAreaDefinition
from pyresample.test.utils import assert_future_geometry


class TestAreaHashability:
    """Test various hashing cases of AreaDefinitions."""

    def test_area_hash(self, stere_area, create_test_area):
        """Test the area hash."""
        area_def = stere_area
        assert isinstance(hash(area_def), int)

        # different dict order
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'lat_0': '50.00',
                'proj': 'stere'},
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001),
            attrs={"name": "areaD"},
        )

        assert isinstance(hash(area_def), int)

        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'lat_0': '50.00',
                'proj': 'stere'},
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001),
            attrs={"name": "New area"},
        )

        assert isinstance(hash(area_def), int)

    def test_get_array_hashable(self):
        """Test making the array hashable."""
        arr = np.array([1.2, 1.3, 1.4, 1.5])
        if sys.byteorder == 'little':
            # arr.view(np.uint8)
            reference = np.array([51, 51, 51, 51, 51, 51, 243,
                                  63, 205, 204, 204, 204, 204,
                                  204, 244, 63, 102, 102, 102, 102,
                                  102, 102, 246, 63, 0, 0,
                                  0, 0, 0, 0, 248, 63],
                                 dtype=np.uint8)
        else:
            # on le machines use arr.byteswap().view(np.uint8)
            reference = np.array([63, 243, 51, 51, 51, 51, 51,
                                  51, 63, 244, 204, 204, 204,
                                  204, 204, 205, 63, 246, 102, 102,
                                  102, 102, 102, 102, 63, 248,
                                  0, 0, 0, 0, 0, 0],
                                 dtype=np.uint8)

        np.testing.assert_allclose(reference,
                                   get_array_hashable(arr))

        xrarr = xr.DataArray(arr)
        np.testing.assert_allclose(reference,
                                   get_array_hashable(arr))

        xrarr.attrs['hash'] = 42
        assert get_array_hashable(xrarr) == xrarr.attrs['hash']


class TestAreaComparisons:
    """Test comparisons with areas and other areas or swaths."""

    def test_area_equal(self, create_test_area):
        """Test area equality."""
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        area_def2 = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        assert not (area_def != area_def2), 'area_defs are not equal as expected'

    def test_not_area_equal(self, create_test_area):
        """Test area inequality."""
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))

        msg_area = create_test_area(
            {
                'a': '6378169.0',
                'b': '6356584.0',
                'h': '35785831.0',
                'lon_0': '0',
                'proj': 'geos'
            },
            3712,
            3712,
            (-5568742.4000000004, -5568742.4000000004, 5568742.4000000004, 5568742.4000000004))
        assert not (area_def == msg_area), 'area_defs are not expected to be equal'
        assert not (area_def == "area"), 'area_defs are not expected to be equal'

    def test_swath_equal_area(self, stere_area, create_test_swath):
        """Test equality between an area and a swath definition."""
        area_def = stere_area
        swath_def = create_test_swath(*area_def.get_lonlats())
        assert not (swath_def != area_def), "swath_def and area_def should be equal"
        assert not (area_def != swath_def), "swath_def and area_def should be equal"

    def test_swath_not_equal_area(self, stere_area, create_test_swath):
        """Test inequality between an area and a swath definition."""
        area_def = stere_area
        lons = np.array([1.2, 1.3, 1.4, 1.5])
        lats = np.array([65.9, 65.86, 65.82, 65.78])
        swath_def = create_test_swath(lons, lats)

        assert not (swath_def == area_def), "swath_def and area_def should be different"
        assert not (area_def == swath_def), "swath_def and area_def should be different"


class TestGridFilter:
    """Tests for the GridFilter class."""

    def test_grid_filter_valid(self, create_test_area):
        """Test valid grid filtering."""
        lons = np.array([-170, -30, 30, 170])
        lats = np.array([20, -40, 50, -80])
        swath_def = SwathDefinition(lons, lats)
        filter_area = create_test_area(
            {
                'proj': 'eqc',
                'lon_0': 0.0,
                'lat_0': 0.0
            },
            8,
            8,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           ])
        grid_filter = geo_filter.GridFilter(filter_area, filter)
        valid_index = grid_filter.get_valid_index(swath_def)
        expected = np.array([1, 0, 0, 1])
        np.testing.assert_array_equal(valid_index, expected, err_msg='Failed to find grid filter')

    def test_grid_filter(self, create_test_area):
        """Test filtering a grid."""
        lons = np.array([-170, -30, 30, 170])
        lats = np.array([20, -40, 50, -80])
        swath_def = SwathDefinition(lons, lats)
        data = np.array([1, 2, 3, 4])
        filter_area = create_test_area(
            {
                'proj': 'eqc',
                'lon_0': 0.0,
                'lat_0': 0.0
            },
            8,
            8,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           ])
        grid_filter = geo_filter.GridFilter(filter_area, filter)
        swath_def_f, data_f = grid_filter.filter(swath_def, data)
        expected = np.array([1, 4])
        np.testing.assert_array_equal(data_f, expected, err_msg='Failed grid filtering data')
        expected_lons = np.array([-170, 170])
        expected_lats = np.array([20, -80])
        assert (np.array_equal(swath_def_f.lons[:], expected_lons) and
                np.array_equal(swath_def_f.lats[:], expected_lats)), 'Failed finding grid filtering lon lats'

    def test_grid_filter2D(self, create_test_area):
        """Test filtering a 2D grid."""
        lons = np.array([[-170, -30, 30, 170],
                         [-170, -30, 30, 170]])
        lats = np.array([[20, -40, 50, -80],
                         [25, -35, 55, -75]])
        swath_def = SwathDefinition(lons, lats)
        data1 = np.ones((2, 4))
        data2 = np.ones((2, 4)) * 2
        data3 = np.ones((2, 4)) * 3
        data = np.dstack((data1, data2, data3))
        filter_area = create_test_area(
            {
                'proj': 'eqc',
                'lon_0': 0.0,
                'lat_0': 0.0
            },
            8,
            8,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           ])
        grid_filter = geo_filter.GridFilter(filter_area, filter, nprocs=2)
        swath_def_f, data_f = grid_filter.filter(swath_def, data)
        expected = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(data_f, expected, err_msg='Failed 2D grid filtering data')
        expected_lons = np.array([-170, 170, -170, 170])
        expected_lats = np.array([20, -80, 25, -75])
        assert (np.array_equal(swath_def_f.lons[:], expected_lons) and
                np.array_equal(swath_def_f.lats[:], expected_lats)), 'Failed finding 2D grid filtering lon lats'


class TestAreaDefinition:
    """Unit tests for the AreaDefinition class."""

    def test_lonlat_precomp(self, stere_area):
        """Test the lonlat precomputation."""
        area_def = stere_area
        area_def.get_lonlats()
        lon, lat = area_def.get_lonlat(400, 400)
        np.testing.assert_allclose(lon, 5.5028467120975835, err_msg='lon retrieval from precomputed grid failed')
        np.testing.assert_allclose(lat, 52.566998432390619, err_msg='lat retrieval from precomputed grid failed')

    def test_cartesian(self, stere_area):
        """Test getting the cartesian coordinates."""
        area_def = stere_area
        cart_coords = area_def.get_cartesian_coords()
        exp = 5872039989466.8457031
        assert (cart_coords.sum() - exp) < 1e-7 * exp, 'Calculation of cartesian coordinates failed'

    def test_cartopy_crs(self, stere_area, create_test_area):
        """Test conversion from area definition to cartopy crs."""
        europe = stere_area
        seviri = create_test_area(
            {
                'proj': 'geos',
                'lon_0': 0.0,
                'a': 6378169.00,
                'b': 6356583.80,
                'h': 35785831.00,
                'units': 'm'},
            123,
            123,
            (5500000, 5500000, -5500000, -5500000))

        for area_def in [europe, seviri]:
            crs = area_def.to_cartopy_crs()

            # Bounds
            assert crs.bounds == (area_def.area_extent[0],
                                  area_def.area_extent[2],
                                  area_def.area_extent[1],
                                  area_def.area_extent[3])

            # Threshold
            thresh_exp = min(np.fabs(area_def.area_extent[2] - area_def.area_extent[0]),
                             np.fabs(area_def.area_extent[3] - area_def.area_extent[1])) / 100.
            assert crs.threshold == thresh_exp

    def test_cartopy_crs_epsg(self, create_test_area):
        """Test conversion from area def to cartopy crs with EPSG codes."""
        projections = ['+init=EPSG:6932', 'EPSG:6932']
        for projection in projections:
            area = create_test_area(
                projection,
                123, 123,
                (-40000., -40000., 40000., 40000.))
            area.to_cartopy_crs()

    def test_cartopy_crs_latlon_bounds(self, create_test_area):
        """Test that a cartopy CRS for a lon/lat area has proper bounds."""
        area_def = create_test_area(
            {'proj': 'latlong', 'lon0': 0},
            360,
            180,
            (-180, -90, 180, 90))
        latlong_crs = area_def.to_cartopy_crs()
        np.testing.assert_allclose(latlong_crs.bounds, [-180, 180, -90, 90])

    def test_to_odc_geobox_odc_missing(self, monkeypatch, stere_area):
        """Test odc-geo not installed."""
        area = stere_area

        with monkeypatch.context() as m:
            m.setattr(pyresample.geometry, "odc_geo", None)

            with pytest.raises(ModuleNotFoundError):
                area.to_odc_geobox()

    def test_to_odc_geobox(self, stere_area, create_test_area):
        """Test conversion from area definition to odc GeoBox."""
        from odc.geo.geobox import GeoBox

        europe = stere_area
        seviri = create_test_area(
            {
                'proj': 'geos',
                'lon_0': 0.0,
                'a': 6378169.00,
                'b': 6356583.80,
                'h': 35785831.00,
                'units': 'm'},
            123,
            123,
            (-5500000, -5500000, 5500000, 5500000))

        for area_def in [europe, seviri]:
            geobox = area_def.to_odc_geobox()

            assert isinstance(geobox, GeoBox)

            # Affine coefficiants
            af = geobox.affine
            assert af.a == area_def.pixel_size_x
            assert af.e == -area_def.pixel_size_y
            assert af.xoff == area_def.area_extent[0]
            assert af.yoff == area_def.area_extent[3]

    def test_dump(self, create_test_area):
        """Test exporting area defs."""
        from io import StringIO

        import yaml
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '90.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001),
            attrs={"name": "areaD"},
        )
        res = yaml.safe_load(area_def.dump())
        expected = yaml.safe_load(('areaD:\n  description: ""\n'
                                   '  projection:\n    a: 6378144.0\n    b: 6356759.0\n'
                                   '    lat_0: 90.0\n    lat_ts: 50.0\n    lon_0: 8.0\n'
                                   '    proj: stere\n  shape:\n    height: 800\n'
                                   '    width: 800\n  area_extent:\n'
                                   '    lower_left_xy: [-1370912.72, -909968.64]\n'
                                   '    upper_right_xy: [1029087.28, 1490031.36]\n'))

        assert set(res.keys()) == set(expected.keys())
        res = res['areaD']
        expected = expected['areaD']
        assert set(res.keys()) == set(expected.keys())
        assert res['description'] == expected['description']
        assert res['shape'] == expected['shape']
        assert res['area_extent']['lower_left_xy'] == expected['area_extent']['lower_left_xy']
        # pyproj versions may affect how the PROJ is formatted
        for proj_key in ['a', 'lat_0', 'lon_0', 'proj', 'lat_ts']:
            assert res['projection'][proj_key] == expected['projection'][proj_key]

        # EPSG
        projections = {
            '+init=epsg:3006': 'init: epsg:3006',
            'EPSG:3006': 'EPSG: 3006',
        }

        for projection, epsg_yaml in projections.items():
            area_def = create_test_area(
                projection,
                4667,
                4667,
                (-49739, 5954123, 1350361, 7354223),
                attrs={"name": "baws300_sweref99tm"},
            )
            res = yaml.safe_load(area_def.dump())
            yaml_string = ('baws300_sweref99tm:\n'
                           '  description: \'\'\n'
                           '  projection:\n'
                           '    {epsg}\n'
                           '  shape:\n'
                           '    height: 4667\n'
                           '    width: 4667\n'
                           '  area_extent:\n'
                           '    lower_left_xy: [-49739, 5954123]\n'
                           '    upper_right_xy: [1350361, 7354223]\n'.format(epsg=epsg_yaml))
            expected = yaml.safe_load(yaml_string)
        assert res == expected

        # testing writing to file with file-like object
        sio = StringIO()
        area_def.dump(filename=sio)
        res = yaml.safe_load(sio.getvalue())
        assert res == expected

        # test writing to file with string filename
        with patch('pyresample.geometry.open') as mock_open:
            area_def.dump(filename='area_file.yml')
            mock_open.assert_called_once_with('area_file.yml', 'a')
            mock_open.return_value.__enter__().write.assert_called_once_with(yaml_string)

    def test_dump_numpy_extents(self, create_test_area):
        """Test exporting area defs when extents are Numpy floats."""
        import yaml

        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '90.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            [np.float64(-1370912.72),
             np.float64(-909968.64000000001),
             np.float64(1029087.28),
             np.float64(1490031.3600000001)],
            attrs={"name": "areaD"},
        )
        res = yaml.safe_load(area_def.dump())
        expected = yaml.safe_load(('areaD:\n  description: Europe (3km, HRV, VTC)\n'
                                   '  projection:\n    a: 6378144.0\n    b: 6356759.0\n'
                                   '    lat_0: 90.0\n    lat_ts: 50.0\n    lon_0: 8.0\n'
                                   '    proj: stere\n  shape:\n    height: 800\n'
                                   '    width: 800\n  area_extent:\n'
                                   '    lower_left_xy: [-1370912.72, -909968.64]\n'
                                   '    upper_right_xy: [1029087.28, 1490031.36]\n'))

        assert res['areaD']['area_extent']['lower_left_xy'] == expected['areaD']['area_extent']['lower_left_xy']

    def test_parse_area_file(self, stere_area, create_test_area):
        """Test parsing the area file."""
        expected = stere_area
        yaml_str = ('areaD:\n  description: Europe (3km, HRV, VTC)\n'
                    '  projection:\n    a: 6378144.0\n    b: 6356759.0\n'
                    '    lat_0: 50.0\n    lat_ts: 50.0\n    lon_0: 8.0\n'
                    '    proj: stere\n  shape:\n    height: 800\n'
                    '    width: 800\n  area_extent:\n'
                    '    lower_left_xy: [-1370912.72, -909968.64]\n'
                    '    upper_right_xy: [1029087.28, 1490031.36]\n')
        yaml_filelike = io.StringIO(yaml_str)
        area_def = parse_area_file(yaml_filelike, 'areaD')[0]
        assert area_def == expected

        # EPSG
        projections = {
            '+init=epsg:3006': 'init: epsg:3006',
            'EPSG:3006': 'EPSG: 3006',
        }
        for projection, epsg_yaml in projections.items():
            expected = create_test_area(
                projection,
                4667,
                4667,
                (-49739, 5954123, 1350361, 7354223),
                attrs={"name": "baws300_sweref99tm"}
            )
            yaml_str = ('baws300_sweref99tm:\n'
                        '  description: BAWS, 300m resolution, sweref99tm\n'
                        '  projection:\n'
                        '    {epsg}\n'
                        '  shape:\n'
                        '    height: 4667\n'
                        '    width: 4667\n'
                        '  area_extent:\n'
                        '    lower_left_xy: [-49739, 5954123]\n'
                        '    upper_right_xy: [1350361, 7354223]'.format(epsg=epsg_yaml))
            yaml_filelike = io.StringIO(yaml_str)
            area_def = parse_area_file(yaml_filelike, 'baws300_sweref99tm')[0]
            assert area_def == expected

    def test_projection_coordinates(self, create_test_area):
        """Test getting the boundary."""
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            10,
            10,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        proj_x_boundary, proj_y_boundary = area_def.projection_x_coords, area_def.projection_y_coords
        expected_x = np.array([-1250912.72, -1010912.72, -770912.72,
                               -530912.72, -290912.72, -50912.72, 189087.28,
                               429087.28, 669087.28, 909087.28])
        expected_y = np.array([1370031.36, 1130031.36, 890031.36, 650031.36,
                               410031.36, 170031.36, -69968.64, -309968.64,
                               -549968.64, -789968.64])
        np.testing.assert_allclose(proj_x_boundary, expected_x, err_msg='Failed to find projection x coords')
        np.testing.assert_allclose(proj_y_boundary, expected_y, err_msg='Failed to find projection y coords')

    def test_area_extent_ll(self, create_test_area):
        """Test getting the lower left area extent."""
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            10,
            10,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        np.testing.assert_allclose(sum(area_def.area_extent_ll), 122.06448093539757, atol=1e-5,
                                   err_msg='Failed to get lon and lats of area extent')

    def test_latlong_area(self, create_test_area):
        """Test getting lons and lats from an area."""
        area_def = create_test_area(
            {
                'proj': 'latlong',
            },
            360,
            180,
            (-180, -90, 180, 90))
        lons, lats = area_def.get_lonlats()
        assert lons[0, 0] == -179.5
        assert lats[0, 0] == 89.5

    def test_colrow2lonlat(self, create_test_area):
        """Test colrow2lonlat."""
        # test square, symmetric areadef
        x_size = 3712
        y_size = 3712
        area_extent = [-5570248.477339261, -5567248.074173444,
                       5567248.074173444, 5570248.477339261]
        proj_dict = {'a': '6378169.00',
                     'b': '6356583.80',
                     'h': '35785831.0',
                     'lon_0': '0.0',
                     'proj': 'geos'}
        area = create_test_area(proj_dict, x_size, y_size, area_extent)

        # Imatra, Wiesbaden
        cols = np.array([2304, 2040])
        rows = np.array([186, 341])
        lons__, lats__ = area.colrow2lonlat(cols, rows)

        # test arrays
        lon_expects = np.array([28.77763033, 8.23765962])
        lat_expects = np.array([61.20120556, 50.05836402])
        np.testing.assert_allclose(lons__, lon_expects, rtol=0, atol=1e-7)
        np.testing.assert_allclose(lats__, lat_expects, rtol=0, atol=1e-7)

        # test scalars
        lon__, lat__ = area.colrow2lonlat(1567, 2375)
        lon_expect = -8.125547604568746
        lat_expect = -14.345524111874646
        np.testing.assert_allclose(lon__, lon_expect, rtol=0, atol=1e-7)
        np.testing.assert_allclose(lat__, lat_expect, rtol=0, atol=1e-7)

        # test rectangular areadef
        x_size = 2560
        y_size = 2048
        area_extent = [-3780000.0, -7644000.0, 3900000.0, -1500000.0]
        proj_dict = {
            'lat_0': 90.0,
            'lon_0': 0.0,
            'lat_ts': 60.0,
            'ellps': 'WGS84',
            'proj': 'stere'}
        area = create_test_area(proj_dict, x_size, y_size, area_extent)

        # Darmstadt, Gibraltar
        cols = np.array([1477, 1069])
        rows = np.array([938, 1513])
        lons__, lats__ = area.colrow2lonlat(cols, rows)

        # test arrays
        lon_expects = np.array([8.597949006575268, -5.404744177829209])
        lat_expects = np.array([49.79024658538765, 36.00540657185169])
        np.testing.assert_allclose(lons__, lon_expects, rtol=0, atol=1e-7)
        np.testing.assert_allclose(lats__, lat_expects, rtol=0, atol=1e-7)

        # test scalars
        # Selva di Val Gardena
        lon__, lat__ = area.colrow2lonlat(1582, 1049)
        lon_expect = 11.75721385976652
        lat_expect = 46.56384754346095
        np.testing.assert_allclose(lon__, lon_expect, rtol=0, atol=1e-7)
        np.testing.assert_allclose(lat__, lat_expect, rtol=0, atol=1e-7)

    @pytest.mark.parametrize("use_dask", [False, True])
    def test_get_proj_coords(self, laea_area, use_dask):
        """Test basic get_proj_coords usage."""
        kwargs = {} if not use_dask else {"chunks": 4096}
        xcoord, ycoord = laea_area.get_proj_coords(**kwargs)
        if use_dask:
            xcoord, ycoord = da.compute(xcoord, ycoord)

        np.testing.assert_allclose(xcoord[0, :],
                                   np.array([1002500., 1007500., 1012500.,
                                             1017500., 1022500., 1027500.,
                                             1032500., 1037500., 1042500.,
                                             1047500.]))
        np.testing.assert_allclose(ycoord[:, 0],
                                   np.array([47500., 42500., 37500., 32500.,
                                             27500., 22500., 17500., 12500.,
                                             7500., 2500.]))

    @pytest.mark.parametrize("use_dask", [False, True])
    def test_get_proj_coords_slices(self, laea_area, use_dask):
        """Test get_proj_coords with slicing."""
        kwargs = {} if not use_dask else {"chunks": 4096}
        xcoord, ycoord = laea_area.get_proj_coords(data_slice=(slice(None, None, 2),
                                                               slice(None, None, 2)),
                                                   **kwargs)
        if use_dask:
            xcoord, ycoord = da.compute(xcoord, ycoord)

        np.testing.assert_allclose(xcoord[0, :],
                                   np.array([1002500., 1012500., 1022500.,
                                             1032500., 1042500.]))
        np.testing.assert_allclose(ycoord[:, 0],
                                   np.array([47500., 37500., 27500., 17500.,
                                             7500.]))

    def test_get_proj_coords_dask_names(self, laea_area):
        """Test that generated coordinate dask task names are chunks-unique."""
        xcoord, ycoord = laea_area.get_proj_coords(chunks=4096)
        xcoord2, ycoord2 = laea_area.get_proj_coords(chunks=2048)
        assert xcoord2.name != xcoord.name
        assert ycoord2.name != ycoord.name

    @pytest.mark.parametrize("use_dask", [False, True])
    def test_roundtrip_lonlat_array_coordinates(self, create_test_area, use_dask):
        """Test roundtrip array coordinates with lon/lat and x/y."""
        x_size = 100
        y_size = 100
        area_extent = [0, -500000, 1000000, 500000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '50',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = create_test_area(proj_dict, x_size, y_size, area_extent)
        lat1, lon1 = 48.832222, 2.355556  # Paris, 13th arrondissement, France
        lat2, lon2 = 58.6, 16.2  # Norrköping, Sweden
        lon = np.array([lon1, lon2])
        lat = np.array([lat1, lat2])
        if use_dask:
            lon = da.from_array(lon)
            lat = da.from_array(lat)
        x__, y__ = area_def.get_array_coordinates_from_lonlat(lon, lat)
        res_lon, res_lat = area_def.get_lonlat_from_array_coordinates(x__, y__)
        assert isinstance(res_lon, da.Array if use_dask else np.ndarray)
        assert isinstance(res_lat, da.Array if use_dask else np.ndarray)
        np.testing.assert_allclose(res_lon, lon)
        np.testing.assert_allclose(res_lat, lat)

    def test_get_lonlats_vs_get_lonlat(self, create_test_area):
        """Test that both function yield similar results."""
        x_size = 100
        y_size = 100
        area_extent = [0, -500000, 1000000, 500000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '50',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = create_test_area(proj_dict, x_size, y_size, area_extent)
        lons, lats = area_def.get_lonlats()
        x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))
        lon, lat = area_def.get_lonlat_from_array_coordinates(x, y)
        np.testing.assert_allclose(lons, lon)
        np.testing.assert_allclose(lats, lat)

    def test_area_corners_around_south_pole(self, create_test_area):
        """Test corner values for the ease-sh area."""
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        width = 425
        height = 425
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
        area_def = create_test_area(projection, width, height, area_extent)

        expected = [(-45.0, -17.713517415148853),
                    (45.000000000000014, -17.71351741514884),
                    (135.0, -17.713517415148825),
                    (-135.00000000000003, -17.71351741514884)]
        actual = [(np.rad2deg(coord.lon), np.rad2deg(coord.lat)) for coord in area_def.corners]
        np.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        ("lon", "lat", "exp_mask", "exp_col", "exp_row"),
        [
            # Choose a point outside the area
            (33.5, -40.5, True, 0.0, 0.0),
            # A point just barely outside the left extent (near floating point precision)
            (-63.62135, 37.253807, False, 0, 5),
            # A point just barely outside the right extent (near floating point precision)
            (63.59189, 37.26574, False, 3711, 5),
        ]
    )
    def test_get_array_indices_from_lonlat_mask_actual_values(
            self, create_test_area, lon, lat, exp_mask, exp_col, exp_row):
        """Test masking behavior of get_array_indices_from_lonlat edge cases."""
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'lat_1': 25.,
                     'lat_2': 25., 'lon_0': 0.0, 'proj': 'lcc', 'units': 'm'}
        area_def = create_test_area(proj_dict, x_size, y_size, area_extent)

        x, y = area_def.get_array_indices_from_lonlat([lon], [lat])
        if exp_mask:
            assert x.mask.all()
            assert y.mask.all()
        else:
            assert not x.mask.any()
            assert not y.mask.any()
            assert x.item() == exp_col
            assert y.item() == exp_row

    @pytest.mark.parametrize(
        ("lons", "lats", "exp_cols", "exp_rows"),
        [
            # Imatra, Wiesbaden
            (np.array([28.75242, 8.24932]), np.array([61.17185, 50.08258]),
             np.array([2304, 2040]), np.array([186, 341])),
            (-8.125547604568746, -14.345524111874646,
             1567, 2375),
        ]
    )
    def test_get_array_indices_from_lonlat_geos(self, create_test_area, lons, lats, exp_cols, exp_rows):
        """Test get_array_indices_from_lonlat."""
        x_size = 3712
        y_size = 3712
        area_extent = [-5570248.477339261, -5567248.074173444, 5567248.074173444, 5570248.477339261]
        proj_dict = {'a': '6378169.00',
                     'b': '6356583.80',
                     'h': '35785831.0',
                     'lon_0': '0.0',
                     'proj': 'geos'}
        area = create_test_area(proj_dict, x_size, y_size, area_extent)

        cols__, rows__ = area.get_array_indices_from_lonlat(lons, lats)
        if hasattr(exp_cols, "shape"):
            np.testing.assert_array_equal(cols__, exp_cols)
            np.testing.assert_array_equal(rows__, exp_rows)
        else:
            assert cols__ == exp_cols
            assert rows__ == exp_rows

    def test_get_array_indices_from_lonlat(self, create_test_area):
        """Test the function get_array_indices_from_lonlat."""
        x_size = 2
        y_size = 2
        area_extent = [1000000, 0, 1050000, 50000]
        proj_dict = {"proj": 'laea',
                     'lat_0': '60',
                     'lon_0': '0',
                     'a': '6371228.0', 'units': 'm'}
        area_def = create_test_area(proj_dict, x_size, y_size, area_extent)
        p__ = Proj(proj_dict)
        lon_ul, lat_ul = p__(1000000, 50000, inverse=True)
        lon_ur, lat_ur = p__(1050000, 50000, inverse=True)
        lon_ll, lat_ll = p__(1000000, 0, inverse=True)
        lon_lr, lat_lr = p__(1050000, 0, inverse=True)

        eps_lonlat = 0.01
        eps_meters = 100
        x__, y__ = area_def.get_array_indices_from_lonlat(lon_ul + eps_lonlat, lat_ul - eps_lonlat)
        x_expect, y_expect = 0, 0
        assert x__ == x_expect
        assert y__ == y_expect
        x__, y__ = area_def.get_array_indices_from_lonlat(lon_ur - eps_lonlat, lat_ur - eps_lonlat)
        assert x__ == 1
        assert y__ == 0
        x__, y__ = area_def.get_array_indices_from_lonlat(lon_ll + eps_lonlat, lat_ll + eps_lonlat)
        assert x__ == 0
        assert y__ == 1
        x__, y__ = area_def.get_array_indices_from_lonlat(lon_lr - eps_lonlat, lat_lr + eps_lonlat)
        assert x__ == 1
        assert y__ == 1

        lon, lat = p__(1025000 - eps_meters, 25000 - eps_meters, inverse=True)
        x__, y__ = area_def.get_array_indices_from_lonlat(lon, lat)
        assert x__ == 0
        assert y__ == 1

        lon, lat = p__(1025000 + eps_meters, 25000 - eps_meters, inverse=True)
        x__, y__ = area_def.get_array_indices_from_lonlat(lon, lat)
        assert x__ == 1
        assert y__ == 1

        lon, lat = p__(1025000 - eps_meters, 25000 + eps_meters, inverse=True)
        x__, y__ = area_def.get_array_indices_from_lonlat(lon, lat)
        assert x__ == 0
        assert y__ == 0

        lon, lat = p__(1025000 + eps_meters, 25000 + eps_meters, inverse=True)
        x__, y__ = area_def.get_array_indices_from_lonlat(lon, lat)
        assert x__ == 1
        assert y__ == 0

        lon, lat = p__(999000, -10, inverse=True)
        with pytest.raises(ValueError):
            area_def.get_array_indices_from_lonlat(lon, lat)
        with pytest.raises(ValueError):
            area_def.get_array_indices_from_lonlat(0., 0.)

        # Test getting arrays back:
        lons = [lon_ll + eps_lonlat, lon_ur - eps_lonlat]
        lats = [lat_ll + eps_lonlat, lat_ur - eps_lonlat]
        x__, y__ = area_def.get_array_indices_from_lonlat(lons, lats)

        x_expects = np.array([0, 1])
        y_expects = np.array([1, 0])
        assert (x__.data == x_expects).all()
        assert (y__.data == y_expects).all()

    @pytest.mark.parametrize(
        "src_extent",
        [
            # Source and target have the same orientation
            (-5580248.477339745, -5571247.267842293, 5577248.074173927, 5580248.477339745),
            # Source is flipped in X direction
            (5577248.074173927, -5571247.267842293, -5580248.477339745, 5580248.477339745),
            # Source is flipped in Y direction
            (-5580248.477339745, 5580248.477339745, 5577248.074173927, -5571247.267842293),
            # Source is flipped in both X and Y directions
            (5577248.074173927, 5580248.477339745, -5580248.477339745, -5571247.267842293),
        ]
    )
    def test_get_slice_starts_stops(self, create_test_area, src_extent):
        """Check area slice end-points."""
        from pyresample.future.geometry._subset import _get_slice_starts_stops
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        target_area = create_test_area(proj_dict, x_size, y_size, area_extent)
        source_area = create_test_area(proj_dict, x_size, y_size, src_extent)
        res = _get_slice_starts_stops(source_area, target_area)
        assert res == (3, 3709, 3, 3709)

    def test_proj_str(self, create_test_area):
        """Test the 'proj_str' property of AreaDefinition."""
        from collections import OrderedDict

        from pyresample.test.utils import friendly_crs_equal

        # pyproj 2.0+ adds a +type=crs parameter
        proj_dict = OrderedDict()
        proj_dict['proj'] = 'stere'
        proj_dict['a'] = 6378144.0
        proj_dict['b'] = 6356759.0
        proj_dict['lat_0'] = 90.00
        proj_dict['lat_ts'] = 50.00
        proj_dict['lon_0'] = 8.00
        area = create_test_area(
            proj_dict,
            10,
            10,
            (-1370912.72, -909968.64, 1029087.28, 1490031.36))
        friendly_crs_equal(
            '+a=6378144.0 +b=6356759.0 +lat_0=90.0 +lat_ts=50.0 '
            '+lon_0=8.0 +proj=stere',
            area
        )
        # try a omerc projection and no_rot parameters
        proj_dict['proj'] = 'omerc'
        proj_dict['lat_0'] = 50.0
        proj_dict['alpha'] = proj_dict.pop('lat_ts')
        proj_dict['no_rot'] = True
        area = create_test_area(
            proj_dict,
            10,
            10,
            (-1370912.72, -909968.64, 1029087.28, 1490031.36))
        friendly_crs_equal(
            '+proj=omerc +a=6378144.0 +b=6356759.0 +lat_0=50.0 '
            '+lon_0=8.0 +alpha=50.0 +no_rot',
            area
        )

        # EPSG
        # With pyproj 2.0+ we expand EPSG to full parameter list
        full_proj = ('+datum=WGS84 +lat_0=-90 +lon_0=0 +no_defs '
                     '+proj=laea +type=crs +units=m +x_0=0 +y_0=0')
        projections = [
            ('+init=EPSG:6932', full_proj),
            ('EPSG:6932', full_proj)
        ]
        for projection, expected_proj in projections:
            area = create_test_area(
                projection,
                123,
                123,
                (-40000., -40000., 40000., 40000.))
            with ignore_pyproj_proj_warnings():
                assert area.proj_str == expected_proj

        # CRS with towgs84 in it
        # we remove towgs84 if they are all 0s
        projection = {'proj': 'laea', 'lat_0': 52, 'lon_0': 10, 'x_0': 4321000, 'y_0': 3210000,
                      'ellps': 'GRS80', 'towgs84': '0,0,0,0,0,0,0', 'units': 'm', 'no_defs': True}
        area = create_test_area(
            projection,
            123,
            123,
            (-40000., -40000., 40000., 40000.))
        with ignore_pyproj_proj_warnings():
            assert area.proj_str == ('+ellps=GRS80 +lat_0=52 +lon_0=10 +no_defs +proj=laea '
                                     '+type=crs +units=m +x_0=4321000 +y_0=3210000')
        projection = {'proj': 'laea', 'lat_0': 52, 'lon_0': 10, 'x_0': 4321000, 'y_0': 3210000,
                      'ellps': 'GRS80', 'towgs84': '0,5,0,0,0,0,0', 'units': 'm', 'no_defs': True}
        area = create_test_area(
            projection,
            123,
            123,
            (-40000., -40000., 40000., 40000.))
        with ignore_pyproj_proj_warnings():
            assert area.proj_str == ('+ellps=GRS80 +lat_0=52 +lon_0=10 +no_defs +proj=laea '
                                     '+towgs84=0.0,5.0,0.0,0.0,0.0,0.0,0.0 '
                                     '+type=crs +units=m +x_0=4321000 +y_0=3210000')

    def test_striding(self, create_test_area):
        """Test striding AreaDefinitions."""
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_def = create_test_area(proj_dict, x_size, y_size, area_extent)

        reduced_area = area_def[::4, ::4]
        np.testing.assert_allclose(reduced_area.area_extent, (area_extent[0],
                                                              area_extent[1] + 3 * area_def.pixel_size_y,
                                                              area_extent[2] - 3 * area_def.pixel_size_x,
                                                              area_extent[3]))
        assert reduced_area.shape == (928, 928)

    def test_get_lonlats_options(self, create_test_area):
        """Test that lotlat options are respected."""
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        (lon, _) = area_def.get_lonlats(dtype="f4")
        assert lon.dtype == np.dtype("f4", )

        (lon, _) = area_def.get_lonlats(dtype="f8")
        assert lon.dtype == np.dtype("f8", )

        from dask.array.core import Array as dask_array
        (lon, _) = area_def.get_lonlats(dtype="f4", chunks=4)
        assert lon.dtype == np.dtype("f4", )
        assert isinstance(lon, dask_array)

        (lon, _) = area_def.get_lonlats(dtype="f8", chunks=4)
        assert lon.dtype == np.dtype("f8", )
        assert isinstance(lon, dask_array)

    def test_area_def_geocentric_resolution(self, create_test_area):
        """Test the AreaDefinition.geocentric_resolution method."""
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        # metered projection
        area_def = create_test_area(proj_dict, 3712, 3712, area_extent)
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(10646.562531, geo_res, atol=1e-1)

        # non-square area non-space area
        area_extent = (-4570248.477339745, -3561247.267842293, 0, 3570248.477339745)
        area_def = create_test_area(proj_dict, 2000, 5000, area_extent)
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(2397.687307, geo_res, atol=1e-1)

        # lon/lat
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'proj': 'latlong'}
        area_def = create_test_area(proj_dict, 3712, 3712, [-130, 30, -120, 40])
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(298.647232, geo_res, atol=1e-1)

    def test_area_def_geocentric_resolution_close_dist(self, create_test_area):
        """Test geocentric resolution when distance range isn't big enough for histogram bins.

        The method currently uses `np.histogram_bin_edges`. Starting in numpy
        2.1.0, if the number of bins requested (10 in this default case) can't
        be created because the range of the data is too small, it will raise an
        exception. This test makes sure that geocentric_resolution doesn't
        error out when this case is encountered.

        """
        # this area is known to produce horizontal distances of ~999.9999989758
        # and trigger the error in numpy 2.1.0
        ar = create_test_area(4087, 5, 5, (-2500.0, -2500.0, 2500.0, 2500.0))
        np.testing.assert_allclose(ar.geocentric_resolution(), 999.999999, atol=1e-2)

    def test_area_def_geocentric_resolution_latlong(self, create_test_area):
        """Test the AreaDefinition.geocentric_resolution method on a latlong projection."""
        area_extent = (-110.0, 45.0, -95.0, 55.0)
        area_def = create_test_area("EPSG:4326", 3712, 3712, area_extent)
        geo_res = area_def.geocentric_resolution()
        np.testing.assert_allclose(299.411133, geo_res)

    def test_from_epsg(self, area_class):
        """Test the from_epsg class method."""
        sweref = area_class.from_epsg('3006', 2000)
        assert sweref.description == 'SWEREF99 TM'
        with ignore_pyproj_proj_warnings():
            assert sweref.proj_dict == {'ellps': 'GRS80', 'no_defs': None,
                                        'proj': 'utm', 'type': 'crs', 'units': 'm',
                                        'zone': 33}
        assert sweref.width == 453
        assert sweref.height == 794
        np.testing.assert_allclose(sweref.area_extent,
                                   (181896.3291, 6101648.0705,
                                    1086312.942376, 7689478.3056))

    def test_from_cf(self, area_class):
        """Test the from_cf class method."""
        # prepare a netCDF/CF lookalike with xarray
        nlat = 19
        nlon = 37
        ds = xr.Dataset({'temp': (('lat', 'lon'), np.ma.masked_all((nlat, nlon)))},
                        coords={'lat': np.linspace(-90., +90., num=nlat),
                                'lon': np.linspace(-180., +180., num=nlon)},)
        ds['lat'].attrs['units'] = 'degreeN'
        ds['lat'].attrs['standard_name'] = 'latitude'
        ds['lon'].attrs['units'] = 'degreeE'
        ds['lon'].attrs['standard_name'] = 'longitude'

        # call from_cf() and check the results
        adef = area_class.from_cf(ds)

        assert adef.shape == (19, 37)
        xc = adef.projection_x_coords
        yc = adef.projection_y_coords
        assert xc[0] == -180., "Wrong x axis (index 0)"
        assert xc[1] == -180. + 10.0, "Wrong x axis (index 1)"
        assert yc[0] == -90., "Wrong y axis (index 0)"
        assert yc[1] == -90. + 10.0, "Wrong y axis (index 1)"

    def test_area_def_init_projection(self, create_test_area):
        """Test AreaDefinition with different projection definitions."""
        proj_dict = {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '90.00',
            'lat_ts': '50.00',
            'lon_0': '8.00',
            'proj': 'stere'
        }
        crs = CRS(CRS.from_dict(proj_dict).to_wkt())
        # pass CRS object directly
        area_def = create_test_area(
            crs,
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        assert crs == area_def.crs
        # PROJ dictionary
        with ignore_pyproj_proj_warnings():
            proj_dict = crs.to_dict()
            proj_str = crs.to_string()

        area_def = create_test_area(
            proj_dict,
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        assert crs == area_def.crs
        # PROJ string
        area_def = create_test_area(
            proj_str,
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        assert crs == area_def.crs
        # WKT2
        area_def = create_test_area(
            crs.to_wkt(),
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        assert crs == area_def.crs
        # WKT1_ESRI
        area_def = create_test_area(
            crs.to_wkt(version='WKT1_ESRI'),
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        # WKT1 to WKT2 has some different naming of things so this fails
        # assert crs == area_def.crs

    def test_areadef_immutable(self, create_test_area):
        """Test that some properties of an area definition are immutable."""
        area_def = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            10,
            10,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        with pytest.raises(AttributeError):
            area_def.shape = (10, 10)
        with pytest.raises(AttributeError):
            area_def.proj_str = "seaweed"
        with pytest.raises(AttributeError):
            area_def.area_extent = (-1000000, -900000, 1000000, 1500000)

    def test_aggregate(self, create_test_area):
        """Test aggregation of AreaDefinitions."""
        area = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        res = area.aggregate(x=4, y=2)
        assert res.crs == area.crs
        np.testing.assert_allclose(res.area_extent, area.area_extent)
        assert res.shape[0] == area.shape[0] / 2
        assert res.shape[1] == area.shape[1] / 4


class TestAreaDefinitionMetadata:
    """Test behavior of metadata in an AreaDefinition."""

    def test_area_def_creation_metadata(self):
        """Test passing metadata to AreaDefinition."""
        my_meta = {
            "a": 1,
        }
        area_def = AreaDefinition(
            4326,
            (100, 200),
            (-1000, -500, 1500, 2000),
            attrs=my_meta,
        )
        assert area_def.attrs == my_meta

    def test_area_def_creation_no_metadata(self):
        """Test not passing metadata to AreaDefinition still results in a usable mapping."""
        area_def = AreaDefinition(
            4326,
            (100, 200),
            (-1000, -500, 1500, 2000),
        )
        assert area_def.attrs == {}

    def test_area_def_metadata_equality(self):
        """Test that metadata differences don't contribute to inequality."""
        area_def1 = AreaDefinition(
            4326,
            (100, 200),
            (-1000, -500, 1500, 2000),
            attrs={"a": 1},
        )
        area_def2 = AreaDefinition(
            4326,
            (100, 200),
            (-1000, -500, 1500, 2000),
            attrs={"a": 2},
        )
        assert area_def1.attrs != area_def2.attrs
        assert area_def1 == area_def2


class TestMakeSliceDivisible:
    """Test the _make_slice_divisible."""

    @pytest.mark.parametrize(
        ("sli", "factor"),
        [
            (slice(10, 21), 2),
            (slice(10, 23), 3),
            (slice(10, 23), 5),
        ]
    )
    def test_make_slice_divisible(self, sli, factor):
        """Test that making area shape divisible by a given factor works."""
        from pyresample.future.geometry._subset import _make_slice_divisible

        # Divisible by 2
        assert (sli.stop - sli.start) % factor != 0
        res = _make_slice_divisible(sli, 1000, factor=factor)
        assert (res.stop - res.start) % factor == 0


def assert_np_dict_allclose(dict1, dict2):
    """Check allclose on dicts."""
    assert set(dict1.keys()) == set(dict2.keys())
    for key, val in dict1.items():
        try:
            np.testing.assert_allclose(val, dict2[key])
        except TypeError:
            assert val == dict2[key]


class TestCreateAreaDef:
    """Test the 'create_area_def' utility function."""

    @staticmethod
    def _compare_area_defs(actual, expected, use_proj4=False):
        if use_proj4:
            # some EPSG codes have a lot of extra metadata that makes the CRS
            # unequal. Skip real area equality and use this as an approximation
            with ignore_pyproj_proj_warnings():
                actual_str = actual.crs.to_proj4()
                expected_str = expected.crs.to_proj4()
            assert actual_str == expected_str
            assert actual.shape == expected.shape
            np.allclose(actual.area_extent, expected.area_extent)
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        'projection',
        [
            {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m',
            '+init=EPSG:3409',
            'EPSG:3409',
        ])
    @pytest.mark.parametrize(
        'center',
        [
            [0, 0],
            'a',
            (1, 2, 3),
        ])
    @pytest.mark.parametrize('units', ['meters', 'degrees'])
    def test_create_area_def_base_combinations(self, projection, center, units, create_test_area):
        """Test create_area_def and the four sub-methods that call it in AreaDefinition."""
        from pyresample.area_config import create_area_def as cad

        area_id = 'ease_sh'
        description = 'Antarctic EASE grid'
        proj_id = 'ease_sh'
        shape = (425, 850)
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
        base_def = create_test_area(
            {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            shape[1], shape[0], area_extent,
        )

        # Tests that incorrect lists do not create an area definition, that both projection strings and
        # dicts are accepted, and that degrees and meters both create the same area definition.
        # area_list used to check that areas are all correct at the end.
        # essentials = center, radius, upper_left_extent, resolution, shape.
        if 'm' in units:
            # Meters.
            essentials = [[0, 0], [5326849.0625, 5326849.0625], (-5326849.0625, 5326849.0625),
                          (12533.7625, 25067.525), (425, 850)]
        else:
            # Degrees.
            essentials = [(0.0, -90.0), 49.4217406986, (-45.0, -17.516001139327766),
                          (0.11271481862984278, 0.22542974631297721), (425, 850)]
        # If center is valid, use it.
        if len(center) == 2:
            center = essentials[0]

        args = (area_id, projection)
        kwargs = dict(
            proj_id=proj_id,
            upper_left_extent=essentials[2],
            center=center,
            shape=essentials[4],
            resolution=essentials[3],
            radius=essentials[1],
            description=description,
            units=units,
        )

        should_fail = isinstance(center, str) or len(center) != 2
        if should_fail:
            pytest.raises(ValueError, cad, *args, **kwargs)
            return

        future_geometries = isinstance(base_def, AreaDefinition)
        with pyresample.config.set({"features.future_geometries": future_geometries}):
            area_def = cad(*args, **kwargs)
        assert_future_geometry(area_def, future_geometries)
        self._compare_area_defs(area_def, base_def, use_proj4="EPSG" in projection)

    def test_create_area_def_extra_combinations(self, create_test_area):
        """Test extra combinations of create_area_def parameters."""
        from pyresample import create_area_def as cad

        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        area_id = 'ease_sh'
        shape = (425, 850)
        upper_left_extent = (-5326849.0625, 5326849.0625)
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
        resolution = (12533.7625, 25067.525)
        radius = [5326849.0625, 5326849.0625]
        base_def = create_test_area(
            {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'},
            shape[1], shape[0], area_extent)

        # Tests that specifying units through xarrays works.
        area_def = cad(area_id, projection, shape=shape,
                       area_extent=xr.DataArray(
                           (-135.0, -17.516001139327766, 45.0, -17.516001139327766),
                           attrs={'units': 'degrees'}))
        self._compare_area_defs(area_def, base_def)

        # Tests area functions 1-A and 2-A.
        area_def = cad(area_id, projection, resolution=resolution, area_extent=area_extent)
        self._compare_area_defs(area_def, base_def)

        # Tests area function 1-B. Also test that DynamicAreaDefinition arguments don't crash AreaDefinition.
        area_def = cad(area_id, projection, shape=shape, center=[0, 0],
                       upper_left_extent=upper_left_extent, optimize_projection=None)
        self._compare_area_defs(area_def, base_def)

        # Tests area function 1-C.
        area_def = cad(area_id, projection, shape=shape, center=[0, 0],
                       radius=radius)
        self._compare_area_defs(area_def, base_def)

        # Tests area function 1-D.
        area_def = cad(area_id, projection, shape=shape,
                       radius=radius, upper_left_extent=upper_left_extent)
        self._compare_area_defs(area_def, base_def)

        # Tests all 4 user cases.
        area_def = AreaDefinition.from_extent(area_id, projection, shape, area_extent)
        self._compare_area_defs(area_def, base_def)

        area_def = AreaDefinition.from_circle(area_id, projection, [0, 0], radius,
                                              resolution=resolution)
        self._compare_area_defs(area_def, base_def)
        area_def = AreaDefinition.from_area_of_interest(area_id, projection,
                                                        shape, [0, 0],
                                                        resolution)
        self._compare_area_defs(area_def, base_def)
        area_def = AreaDefinition.from_ul_corner(area_id, projection, shape,
                                                 upper_left_extent,
                                                 resolution)
        self._compare_area_defs(area_def, base_def)

    def test_create_area_def_nonpole_center(self):
        """Test that a non-pole center can be used."""
        from pyresample import create_area_def as cad
        from pyresample.geometry import AreaDefinition
        area_def = cad('ease_sh', '+a=6371228.0 +units=m +lon_0=0 +proj=merc +lat_0=0',
                       center=(0, 0), radius=45,
                       resolution=(1, 0.9999291722135637),
                       units='degrees')
        assert isinstance(area_def, AreaDefinition)
        np.testing.assert_allclose(area_def.area_extent, (-5003950.7698, -5615432.0761, 5003950.7698, 5615432.0761))
        assert area_def.shape == (101, 90)


class TestCrop:
    """Test the area helpers."""

    def test_sub_area(self, create_test_area):
        """Sub area slicing."""
        area = create_test_area(
            {
                'a': '6378144.0',
                'b': '6356759.0',
                'lat_0': '50.00',
                'lat_ts': '50.00',
                'lon_0': '8.00',
                'proj': 'stere'
            },
            800,
            800,
            (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001))
        res = area[slice(20, 720), slice(100, 500)]
        np.testing.assert_allclose((-1070912.72, -669968.6399999999,
                                    129087.28000000003, 1430031.36),
                                   res.area_extent)
        assert res.shape == (700, 400)


def test_enclose_areas(create_test_area):
    """Test enclosing areas."""
    from pyresample.geometry import enclose_areas
    proj_dict = {'proj': 'geos', 'sweep': 'x', 'lon_0': 0, 'h': 35786023,
                 'x_0': 0, 'y_0': 0, 'ellps': 'GRS80', 'units': 'm',
                 'no_defs': None, 'type': 'crs'}
    proj_dict_alt = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0,
                     'units': 'm'}

    ar1 = create_test_area(
        proj_dict,
        10, 10,
        area_extent=[0, 20, 100, 120],
    )

    ar2 = create_test_area(
        proj_dict,
        10, 10,
        [20, 40, 120, 140],
    )

    ar3 = create_test_area(
        proj_dict,
        10, 10,
        [20, 0, 120, 100],
    )

    ar4 = create_test_area(
        proj_dict_alt,
        10, 10,
        [20, 0, 120, 100],
    )

    ar5 = create_test_area(
        proj_dict,
        100, 100,
        [-50, -50, 50, 50],
    )

    ar_joined = enclose_areas(ar1, ar2, ar3)
    np.testing.assert_allclose(ar_joined.area_extent, [0, 0, 120, 140])
    with pytest.raises(ValueError):
        enclose_areas(ar3, ar4)
    with pytest.raises(ValueError):
        enclose_areas(ar3, ar5)
    with pytest.raises(TypeError):
        enclose_areas()


class TestAreaDefGetAreaSlices:
    """Test AreaDefinition's get_area_slices."""

    def test_get_area_slices_geos_subset(self, geos_src_area, create_test_area):
        """Check area slicing."""
        area_def = geos_src_area
        # An area that is a subset of the original one
        area_to_cover = create_test_area(
            area_def.crs,
            1000, 1000,
            area_extent=(area_def.area_extent[0] + 10000,
                         area_def.area_extent[1] + 10000,
                         area_def.area_extent[2] - 10000,
                         area_def.area_extent[3] - 10000))
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        assert slice(3, 3709, None) == slice_x
        assert slice(3, 3709, None) == slice_y

    def test_get_area_slices_geos_similar(self, geos_src_area, create_test_area):
        """Test slicing with an area similar to the source data but not the same."""
        area_def = geos_src_area
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339261, -5567248.074173444, 5567248.074173444, 5570248.477339261)
        proj_dict = {'a': 6378169.5, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_to_cover = create_test_area(
            proj_dict,
            x_size,
            y_size,
            area_extent)
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        assert slice(46, 3667, None) == slice_x
        assert slice(56, 3659, None) == slice_y

    def test_get_area_slices_geos_stereographic(self, geos_src_area, create_test_area):
        """Test slicing with a geos area and polar stereographic area."""
        area_def = geos_src_area
        area_to_cover = create_test_area(
            {'a': 6378144.0, 'b': 6356759.0, 'lat_0': 50.00, 'lat_ts': 50.00, 'lon_0': 8.00, 'proj': 'stere'},
            10,
            10,
            (-1370912.72, -909968.64, 1029087.28, 1490031.36))
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        assert slice_x == slice(1610, 2343)
        assert slice_y == slice(158, 515, None)

    def test_get_area_slices_geos_flipped_xy(self, geos_src_area, create_test_area):
        """Test slicing with two geos areas but one has flipped x/y dimensions."""
        area_def = geos_src_area
        x_size = 3712
        y_size = 3712
        area_extent = (5567248.074173927, 5570248.477339745, -5570248.477339745, -5561247.267842293)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_to_cover = create_test_area(
            proj_dict,
            x_size,
            y_size,
            area_extent)
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert isinstance(slice_x.start, int)
        assert isinstance(slice_y.start, int)
        assert slice(0, x_size, None) == slice_x
        assert slice(0, y_size, None) == slice_y

    def test_get_area_slices_geos_epsg_lonlat(self, geos_src_area, create_test_area):
        """Test slicing with a geos area and EPSG lon/lat areas."""
        area_def = geos_src_area
        projections = [{"init": 'EPSG:4326'}, 'EPSG:4326']
        for projection in projections:
            area_to_cover = create_test_area(
                projection,
                8192,
                4096,
                (-180.0, -90.0, 180.0, 90.0))

            slice_x, slice_y = area_def.get_area_slices(area_to_cover)
            assert isinstance(slice_x.start, int)
            assert isinstance(slice_y.start, int)
            assert slice_x == slice(46, 3667, None)
            assert slice_y == slice(56, 3659, None)

    def test_get_area_slices_nongeos(self, create_test_area):
        """Check area slicing for non-geos projections."""
        x_size = 3712
        y_size = 3712
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'lat_1': 25.,
                     'lat_2': 25., 'lon_0': 0.0, 'proj': 'lcc', 'units': 'm'}
        area_def = create_test_area(
            proj_dict,
            x_size,
            y_size,
            area_extent)
        area_to_cover = create_test_area(
            area_def.crs,
            1000,
            1000,
            (area_def.area_extent[0] + 10000, area_def.area_extent[1] + 10000,
             area_def.area_extent[2] - 10000, area_def.area_extent[3] - 10000))
        slice_x, slice_y = area_def.get_area_slices(area_to_cover)
        assert slice(3, 3709, None) == slice_x
        assert slice(3, 3709, None) == slice_y

    def test_on_flipped_geos_area(self, create_test_area):
        """Test get_area_slices on flipped areas."""
        src_area = create_test_area(
            {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
            100, 100,
            (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        expected_slice_lines = slice(60, 91)
        expected_slice_cols = slice(90, 100)
        cropped_area = src_area[expected_slice_lines, expected_slice_cols]
        slice_cols, slice_lines = src_area.get_area_slices(cropped_area)
        assert slice_lines == expected_slice_lines
        assert slice_cols == expected_slice_cols

        expected_slice_cols = slice(30, 61)
        cropped_area = src_area[expected_slice_lines, expected_slice_cols]
        slice_cols, slice_lines = src_area.get_area_slices(cropped_area)
        assert slice_lines == expected_slice_lines
        assert slice_cols == expected_slice_cols

    def test_non_geos_can_be_cropped(self, create_test_area):
        """Test that non-geos areas can be cropped also."""
        src_area = create_test_area(dict(proj="utm", zone=33),
                                    10980, 10980,
                                    (499980.0, 6490200.0, 609780.0, 6600000.0))
        crop_area = create_test_area({'proj': 'latlong'},
                                     100, 100,
                                     (15.9689, 58.5284, 16.4346, 58.6995))
        slice_x, slice_y = src_area.get_area_slices(crop_area)
        assert slice_x == slice(5630, 8339)
        assert slice_y == slice(9261, 10980)

    def test_area_to_cover_all_nan_bounds(self, geos_src_area, create_test_area):
        """Check area slicing when the target doesn't have a valid boundary."""
        area_def = geos_src_area
        # An area that is a subset of the original one
        area_to_cover = create_test_area(
            {"proj": "moll"},
            1000, 1000,
            area_extent=(-18000000.0, -9000000.0, 18000000.0, 9000000.0))
        with pytest.raises(NotImplementedError):
            area_def.get_area_slices(area_to_cover)

    @pytest.mark.parametrize("cache_slices", [False, True])
    def test_area_slices_caching(self, create_test_area, tmp_path, cache_slices):
        """Check that area slices can be cached."""
        src_area = create_test_area(dict(proj="utm", zone=33),
                                    10980, 10980,
                                    (499980.0, 6490200.0, 609780.0, 6600000.0))
        crop_area = create_test_area({'proj': 'latlong'},
                                     100, 100,
                                     (15.9689, 58.5284, 16.4346, 58.6995))
        cache_glob = str(tmp_path / "geometry_slices_v1" / "*.json")
        with pyresample.config.set(cache_dir=tmp_path, cache_geometry_slices=cache_slices):
            assert len(glob(cache_glob)) == 0
            slice_x, slice_y = src_area.get_area_slices(crop_area)
            assert len(glob(cache_glob)) == int(cache_slices)
        assert slice_x == slice(5630, 8339)
        assert slice_y == slice(9261, 10980)

        if cache_slices:
            from pyresample.future.geometry._subset import get_area_slices
            with pyresample.config.set(cache_dir=tmp_path):
                get_area_slices.cache_clear()
            assert len(glob(cache_glob)) == 0

    def test_area_slices_caching_no_swaths(self, tmp_path, create_test_area, create_test_swath):
        """Test that swath inputs produce a warning when tried to use in caching."""
        from pyresample.future.geometry._subset import get_area_slices
        from pyresample.test.utils import create_test_latitude, create_test_longitude
        area = create_test_area(dict(proj="utm", zone=33),
                                10980, 10980,
                                (499980.0, 6490200.0, 609780.0, 6600000.0))
        lons = create_test_longitude(-95.0, -75.0, shape=(1000, 500))
        lats = create_test_latitude(25.0, 35.0, shape=(1000, 500))
        swath = create_test_swath(lons, lats)

        with pyresample.config.set(cache_dir=tmp_path, cache_geometry_slices=True), pytest.raises(NotImplementedError):
            with pytest.warns(UserWarning, match="unhashable"):
                get_area_slices(swath, area, None)

    @pytest.mark.parametrize("swath_as_src", [False, True])
    def test_unsupported_slice_inputs(self, create_test_area, create_test_swath, swath_as_src):
        """Test that swath inputs produce an error."""
        from pyresample.future.geometry._subset import get_area_slices
        from pyresample.test.utils import create_test_latitude, create_test_longitude
        area = create_test_area(dict(proj="utm", zone=33),
                                10980, 10980,
                                (499980.0, 6490200.0, 609780.0, 6600000.0))
        lons = create_test_longitude(-95.0, -75.0, shape=(1000, 500))
        lats = create_test_latitude(25.0, 35.0, shape=(1000, 500))
        swath = create_test_swath(lons, lats)

        with pytest.raises(NotImplementedError):
            args = (swath, area) if swath_as_src else (area, swath)
            get_area_slices(*args, None)


def test_future_to_legacy_conversion():
    """Test that future AreaDefinitions can be converted to legacy areas."""
    area_def = AreaDefinition(
        {
            'a': '6378144.0',
            'b': '6356759.0',
            'lat_0': '50.00',
            'lat_ts': '50.00',
            'lon_0': '8.00',
            'proj': 'stere'
        },
        (800, 800),
        (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001),
        attrs={"name": 'areaD'},
    )
    legacy_area = area_def.to_legacy()
    assert isinstance(legacy_area, LegacyAreaDefinition)
    assert legacy_area.area_extent == area_def.area_extent
    assert legacy_area.crs == area_def.crs
    assert legacy_area.area_id == area_def.attrs["name"]


@pytest.mark.parametrize("shape", [(100,), (100, 100, 100)])
def test_non2d_shape_error(shape):
    """Test that non-2D shapes fail."""
    with pytest.raises(NotImplementedError):
        AreaDefinition("EPSG:4326", shape, (-1000.0, -1000.0, 1000.0, 1000.0))
