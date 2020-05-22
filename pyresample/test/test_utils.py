#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015-2020 Pyresample developers
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
from unittest import mock
import io
import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import uuid

from pyresample.test.utils import create_test_longitude, create_test_latitude


def tmptiff(width=100, height=100, transform=None, crs=None, dtype=np.uint8):
    import rasterio
    array = np.ones((width, height)).astype(dtype)
    fname = '/vsimem/%s' % uuid.uuid4()
    with rasterio.open(fname, 'w', driver='GTiff', count=1, transform=transform,
                       width=width, height=height, crs=crs, dtype=dtype) as dst:
        dst.write(array, 1)
    return fname


class TestLegacyAreaParser(unittest.TestCase):
    def test_area_parser_legacy(self):
        """Test legacy area parser."""
        from pyresample import parse_area_file
        from pyresample.utils import is_pyproj2
        ease_nh, ease_sh = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'),
                                           'ease_nh', 'ease_sh')

        if is_pyproj2():
            # pyproj 2.0+ adds some extra parameters
            projection = ("{'R': '6371228', 'lat_0': '90', 'lon_0': '0', "
                          "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                          "'units': 'm', 'x_0': '0', 'y_0': '0'}")
        else:
            projection = ("{'a': '6371228.0', 'lat_0': '90.0', "
                          "'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}")
        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(ease_nh.__str__(), nh_str)
        self.assertIsInstance(ease_nh.proj_dict['lat_0'], (int, float))

        if is_pyproj2():
            projection = ("{'R': '6371228', 'lat_0': '-90', 'lon_0': '0', "
                          "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                          "'units': 'm', 'x_0': '0', 'y_0': '0'}")
        else:
            projection = ("{'a': '6371228.0', 'lat_0': '-90.0', "
                          "'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}")
        sh_str = """Area ID: ease_sh
Description: Antarctic EASE grid
Projection ID: ease_sh
Projection: {}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(ease_sh.__str__(), sh_str)
        self.assertIsInstance(ease_sh.proj_dict['lat_0'], (int, float))

    def test_load_area(self):
        from pyresample import load_area
        from pyresample.utils import is_pyproj2
        ease_nh = load_area(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'ease_nh')
        if is_pyproj2():
            # pyproj 2.0+ adds some extra parameters
            projection = ("{'R': '6371228', 'lat_0': '90', 'lon_0': '0', "
                          "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                          "'units': 'm', 'x_0': '0', 'y_0': '0'}")
        else:
            projection = ("{'a': '6371228.0', 'lat_0': '90.0', "
                          "'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}")
        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(nh_str, ease_nh.__str__())

    def test_area_file_not_found_exception(self):
        from pyresample.area_config import load_area
        self.assertRaises(FileNotFoundError, load_area,
                          "/this/file/does/not/exist.yaml")
        self.assertRaises(FileNotFoundError, load_area,
                          pathlib.Path("/this/file/does/not/exist.yaml"))

    def test_not_found_exception(self):
        from pyresample.area_config import AreaNotFound, parse_area_file
        self.assertRaises(AreaNotFound, parse_area_file,
                          os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'no_area')

    def test_commented(self):
        from pyresample import parse_area_file
        areas = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'))
        self.assertNotIn('commented', [area.name for area in areas])


class TestYAMLAreaParser(unittest.TestCase):
    def test_area_parser_yaml(self):
        """Test YAML area parser."""
        from pyresample import parse_area_file
        test_area_file = os.path.join(os.path.dirname(__file__), 'test_files', 'areas.yaml')
        test_areas = parse_area_file(test_area_file, 'ease_nh', 'ease_sh', 'test_meters', 'test_degrees',
                                     'test_latlong')
        ease_nh, ease_sh, test_m, test_deg, test_latlong = test_areas

        from pyresample.utils import is_pyproj2
        if is_pyproj2():
            # pyproj 2.0+ adds some extra parameters
            projection = ("{'R': '6371228', 'lat_0': '-90', 'lon_0': '0', "
                          "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                          "'units': 'm', 'x_0': '0', 'y_0': '0'}")
        else:
            projection = ("{'a': '6371228.0', 'lat_0': '-90.0', "
                          "'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}")
        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection: {}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(ease_nh.__str__(), nh_str)

        sh_str = """Area ID: ease_sh
Description: Antarctic EASE grid
Projection: {}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(ease_sh.__str__(), sh_str)

        m_str = """Area ID: test_meters
Description: test_meters
Projection: {}
Number of columns: 850
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(test_m.__str__(), m_str)

        deg_str = """Area ID: test_degrees
Description: test_degrees
Projection: {}
Number of columns: 850
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(test_deg.__str__(), deg_str)

        if is_pyproj2():
            # pyproj 2.0+ adds some extra parameters
            projection = ("{'ellps': 'WGS84', 'no_defs': 'None', "
                          "'pm': '-81.36', 'proj': 'longlat', "
                          "'type': 'crs'}")
        else:
            projection = ("{'ellps': 'WGS84', 'lat_0': '27.12', "
                          "'lon_0': '-81.36', 'proj': 'longlat'}")
        latlong_str = """Area ID: test_latlong
Description: Basic latlong grid
Projection: {}
Number of columns: 3473
Number of rows: 4058
Area extent: (-0.0812, 0.4039, 0.0812, 0.5428)""".format(projection)
        self.assertEqual(test_latlong.__str__(), latlong_str)

    def test_dynamic_area_parser_yaml(self):
        """Test YAML area parser on dynamic areas."""
        from pyresample import parse_area_file
        from pyresample.geometry import DynamicAreaDefinition
        test_area_file = os.path.join(os.path.dirname(__file__), 'test_files', 'areas.yaml')
        test_area = parse_area_file(test_area_file, 'test_dynamic_resolution')[0]

        self.assertIsInstance(test_area, DynamicAreaDefinition)
        self.assertTrue(hasattr(test_area, 'resolution'))
        self.assertEqual(test_area.resolution, (1000.0, 1000.0))

        # lat/lon
        from pyresample import parse_area_file
        from pyresample.geometry import DynamicAreaDefinition
        test_area_file = os.path.join(os.path.dirname(__file__), 'test_files', 'areas.yaml')
        test_area = parse_area_file(test_area_file, 'test_dynamic_resolution_ll')[0]

        self.assertIsInstance(test_area, DynamicAreaDefinition)
        self.assertTrue(hasattr(test_area, 'resolution'))
        self.assertEqual(test_area.resolution, (1.0, 1.0))

    def test_multiple_file_content(self):
        from pyresample import parse_area_file
        from pyresample.area_config import load_area_from_string
        area_list = ["""ease_sh:
  description: Antarctic EASE grid
  projection:
    a: 6371228.0
    units: m
    lon_0: 0
    proj: laea
    lat_0: -90
  shape:
    height: 425
    width: 425
  area_extent:
    lower_left_xy: [-5326849.0625, -5326849.0625]
    upper_right_xy: [5326849.0625, 5326849.0625]
    units: m
""",
                     """ease_sh2:
  description: Antarctic EASE grid
  projection:
    a: 6371228.0
    units: m
    lon_0: 0
    proj: laea
    lat_0: -90
  shape:
    height: 425
    width: 425
  area_extent:
    lower_left_xy: [-5326849.0625, -5326849.0625]
    upper_right_xy: [5326849.0625, 5326849.0625]
    units: m
"""]
        with self.assertWarns(DeprecationWarning):
            results = parse_area_file(area_list)
        self.assertEqual(len(results), 2)
        self.assertIn(results[0].area_id, ('ease_sh', 'ease_sh2'))
        self.assertIn(results[1].area_id, ('ease_sh', 'ease_sh2'))
        results2 = parse_area_file([io.StringIO(ar) for ar in area_list])
        results3 = load_area_from_string(area_list)
        self.assertEqual(results, results2)
        self.assertEqual(results, results3)


class TestPreprocessing(unittest.TestCase):
    def test_nearest_neighbor_area_area(self):
        from pyresample import utils, geometry
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
        from pyresample import utils, geometry
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
        from pyresample import utils, geometry
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
        from pyresample import utils, geometry
        lon_arr = create_test_longitude(-95.0, -85.0, (40, 50), dtype=np.float64)
        lat_arr = create_test_latitude(25.0, 35.0, (40, 50), dtype=np.float64)
        grid_dst = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)

        lon_arr = create_test_longitude(-100.0, -80.0, (400, 500), dtype=np.float64)
        lat_arr = create_test_latitude(20.0, 40.0, (400, 500), dtype=np.float64)
        grid = geometry.GridDefinition(lons=lon_arr, lats=lat_arr)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(grid, grid_dst, 12000.)


class TestMisc(unittest.TestCase):
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

        # test again but force pyproj <2 behavior
        with mock.patch.object(utils.proj4, 'CRS', None):
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

        # test again but force pyproj <2 behavior
        with mock.patch.object(utils.proj4, 'CRS', None):
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

        # test again but force pyproj <2 behavior
        with mock.patch.object(utils.proj4, 'CRS', None):
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

        # test again but force pyproj <2 behavior
        with mock.patch.object(utils.proj4, 'CRS', None):
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

        proj_str = "+proj=lcc +ellps=WGS84 +lon_0=-95 +no_defs"
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        proj_str2 = utils.proj4.proj4_dict_to_str(proj_dict)
        proj_dict2 = utils.proj4.proj4_str_to_dict(proj_str2)
        self.assertDictEqual(proj_dict, proj_dict2)
        self.assertIsInstance(proj_dict['lon_0'], float)
        self.assertIsInstance(proj_dict2['lon_0'], float)

        # EPSG
        proj_str = '+init=EPSG:4326'
        proj_dict_exp = {'init': 'EPSG:4326'}
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        self.assertEqual(proj_dict, proj_dict_exp)
        self.assertEqual(utils.proj4.proj4_dict_to_str(proj_dict), proj_str)  # round-trip

        proj_str = 'EPSG:4326'
        proj_dict_exp = {'init': 'EPSG:4326'}
        proj_dict_exp2 = {'proj': 'longlat', 'datum': 'WGS84', 'no_defs': None, 'type': 'crs'}
        proj_dict = utils.proj4.proj4_str_to_dict(proj_str)
        if 'init' in proj_dict:
            # pyproj <2.0
            self.assertEqual(proj_dict, proj_dict_exp)
        else:
            # pyproj 2.0+
            self.assertEqual(proj_dict, proj_dict_exp2)
        # input != output for this style of EPSG code
        # EPSG to PROJ.4 can be lossy
        # self.assertEqual(utils._proj4.proj4_dict_to_str(proj_dict), proj_str)  # round-trip

    def test_def2yaml_converter(self):
        from pyresample import parse_area_file, convert_def_to_yaml
        from pyresample.utils import is_pyproj2
        import tempfile
        def_file = os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg')
        filehandle, yaml_file = tempfile.mkstemp()
        os.close(filehandle)
        try:
            convert_def_to_yaml(def_file, yaml_file)
            areas_new = set(parse_area_file(yaml_file))
            areas = parse_area_file(def_file)
            for area in areas:
                if is_pyproj2():
                    # pyproj 2.0 adds units back in
                    # pyproj <2 doesn't
                    continue
                # initialize _proj_dict
                area.proj_dict  # noqa
                area._proj_dict.pop('units', None)
            areas_old = set(areas)
            areas_new = {area.area_id: area for area in areas_new}
            areas_old = {area.area_id: area for area in areas_old}
            self.assertEqual(areas_new, areas_old)
        finally:
            os.remove(yaml_file)

    def test_get_area_def_from_raster(self):
        from pyresample import utils
        from rasterio.crs import CRS
        from affine import Affine
        x_size = 791
        y_size = 718
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        crs = CRS(init='epsg:3857')
        if utils.is_pyproj2():
            # pyproj 2.0+ expands CRS parameters
            from pyproj import CRS
            proj_dict = CRS(3857).to_dict()
        else:
            proj_dict = crs.to_dict()
        source = tmptiff(x_size, y_size, transform, crs=crs)
        area_id = 'area_id'
        proj_id = 'proj_id'
        description = 'name'
        area_def = utils.rasterio.get_area_def_from_raster(
            source, area_id=area_id, name=description, proj_id=proj_id)
        self.assertEqual(area_def.area_id, area_id)
        self.assertEqual(area_def.proj_id, proj_id)
        self.assertEqual(area_def.description, description)
        self.assertEqual(area_def.width, x_size)
        self.assertEqual(area_def.height, y_size)
        self.assertDictEqual(proj_dict, area_def.proj_dict)
        self.assertTupleEqual(area_def.area_extent, (transform.c, transform.f + transform.e * y_size,
                                                     transform.c + transform.a * x_size, transform.f))

    def test_get_area_def_from_raster_extracts_proj_id(self):
        from rasterio.crs import CRS
        from pyresample import utils
        crs = CRS(init='epsg:3857')
        source = tmptiff(crs=crs)
        area_def = utils.rasterio.get_area_def_from_raster(source)
        self.assertEqual(area_def.proj_id, 'WGS 84 / Pseudo-Mercator')

    def test_get_area_def_from_raster_rotated_value_err(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.1, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        self.assertRaises(ValueError, utils.rasterio.get_area_def_from_raster, source)

    def test_get_area_def_from_raster_non_georef_value_err(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        self.assertRaises(ValueError, utils.rasterio.get_area_def_from_raster, source)

    def test_get_area_def_from_raster_non_georef_respects_proj_dict(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        proj_dict = {'init': 'epsg:3857'}
        area_def = utils.rasterio.get_area_def_from_raster(source, proj_dict=proj_dict)
        if utils.is_pyproj2():
            from pyproj import CRS
            proj_dict = CRS(3857).to_dict()
        self.assertDictEqual(area_def.proj_dict, proj_dict)


class TestProjRotation(unittest.TestCase):
    """Test loading areas with rotation specified."""

    def test_rotation_legacy(self):
        """Basic rotation in legacy format."""
        from pyresample.area_config import load_area
        legacyDef = """REGION: regionB {
        NAME:          regionB
        PCS_ID:        regionB
        PCS_DEF:       proj=merc, lon_0=-34, k=1, x_0=0, y_0=0, a=6378137, b=6378137
        XSIZE:         800
        YSIZE:         548
        ROTATION:      -45
        AREA_EXTENT:   (-7761424.714818418, -4861746.639279127, 11136477.43264252, 8236799.845095873)
        };"""
        with NamedTemporaryFile(mode="w", suffix='.cfg', delete=False) as f:
            f.write(legacyDef)
        test_area = load_area(f.name, 'regionB')
        self.assertEqual(test_area.rotation, -45)
        os.remove(f.name)

    def test_rotation_yaml(self):
        """Basic rotation in yaml format."""
        from pyresample.area_config import load_area
        yamlDef = """regionB:
          description: regionB
          projection:
            a: 6378137.0
            b: 6378137.0
            lon_0: -34
            proj: merc
            x_0: 0
            y_0: 0
            k_0: 1
          shape:
            height: 548
            width: 800
          rotation: -45
          area_extent:
            lower_left_xy: [-7761424.714818418, -4861746.639279127]
            upper_right_xy: [11136477.43264252, 8236799.845095873]
          units: m"""
        with NamedTemporaryFile(mode="w", suffix='.yaml', delete=False) as f:
            f.write(yamlDef)
        test_area = load_area(f.name, 'regionB')
        self.assertEqual(test_area.rotation, -45)
        os.remove(f.name)

    def test_norotation_legacy(self):
        """No rotation specified in legacy format."""
        from pyresample.area_config import load_area
        legacyDef = """REGION: regionB {
        NAME:          regionB
        PCS_ID:        regionB
        PCS_DEF:       proj=merc, lon_0=-34, k=1, x_0=0, y_0=0, a=6378137, b=6378137
        XSIZE:         800
        YSIZE:         548
        AREA_EXTENT:   (-7761424.714818418, -4861746.639279127, 11136477.43264252, 8236799.845095873)
        };"""
        with NamedTemporaryFile(mode="w", suffix='.cfg', delete=False) as f:
            f.write(legacyDef)
        test_area = load_area(f.name, 'regionB')
        self.assertEqual(test_area.rotation, 0)
        os.remove(f.name)

    def test_norotation_yaml(self):
        """No rotation specified in yaml format."""
        from pyresample.area_config import load_area
        yamlDef = """regionB:
          description: regionB
          projection:
            a: 6378137.0
            b: 6378137.0
            lon_0: -34
            proj: merc
            x_0: 0
            y_0: 0
            k_0: 1
          shape:
            height: 548
            width: 800
          area_extent:
            lower_left_xy: [-7761424.714818418, -4861746.639279127]
            upper_right_xy: [11136477.43264252, 8236799.845095873]
          units: m"""
        with NamedTemporaryFile(mode="w", suffix='.yaml', delete=False) as f:
            f.write(yamlDef)
        test_area = load_area(f.name, 'regionB')
        self.assertEqual(test_area.rotation, 0)
        os.remove(f.name)


# helper routines for the CF test cases
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


class TestLoadCFArea_Public(unittest.TestCase):
    """Test public API load_cf_area() for loading an AreaDefinition from netCDF/CF files."""

    def test_load_cf_from_wrong_filepath(self):
        from pyresample.utils import load_cf_area

        # wrong case #1: the path does not exist
        cf_file = os.path.join(os.path.dirname(__file__), 'test_files', 'does_not_exist.nc')
        self.assertRaises(FileNotFoundError, load_cf_area, cf_file)

        # wrong case #2: the path exists, but is not a netCDF file
        cf_file = os.path.join(os.path.dirname(__file__), 'test_files', 'areas.yaml')
        self.assertRaises(OSError, load_cf_area, cf_file)

    def test_load_cf_parameters_errors(self):
        from pyresample.utils import load_cf_area

        # prepare xarray Dataset
        cf_file = _prepare_cf_nh10km()

        # try to load from a variable= that does not exist
        self.assertRaises(KeyError, load_cf_area, cf_file, 'doesNotExist')

        # try to load from a variable= that is itself is a grid_mapping, but without y= or x=
        self.assertRaises(ValueError, load_cf_area, cf_file, 'Polar_Stereographic_Grid',)

        # try to load using a variable= that is a valid grid_mapping container, but use wrong x= and y=
        self.assertRaises(KeyError, load_cf_area, cf_file, 'Polar_Stereographic_Grid', y='doesNotExist', x='xc',)
        self.assertRaises(ValueError, load_cf_area, cf_file, 'Polar_Stereographic_Grid', y='time', x='xc',)

        # try to load using a variable= that does not define a grid mapping
        self.assertRaises(ValueError, load_cf_area, cf_file, 'lat',)

    def test_load_cf_nh10km(self):
        from pyresample.utils import load_cf_area

        def validate_nh10km_adef(adef):
            self.assertEqual(adef.shape, (1120, 760))
            xc = adef.projection_x_coords
            yc = adef.projection_y_coords
            self.assertEqual(xc[0], -3845000.0, msg="Wrong x axis (index 0)")
            self.assertEqual(xc[1], xc[0] + 10000.0, msg="Wrong x axis (index 1)")
            self.assertEqual(yc[0], 5845000.0, msg="Wrong y axis (index 0)")
            self.assertEqual(yc[1], yc[0] - 10000.0, msg="Wrong y axis (index 1)")

        # prepare xarray Dataset
        cf_file = _prepare_cf_nh10km()

        # load using a variable= that is a valid grid_mapping container
        adef, _ = load_cf_area(cf_file, 'Polar_Stereographic_Grid', y='yc', x='xc',)
        validate_nh10km_adef(adef)

        # load using a variable= that has a :grid_mapping attribute
        adef, _ = load_cf_area(cf_file, 'ice_conc')
        validate_nh10km_adef(adef)

        # load without using a variable=
        adef, _ = load_cf_area(cf_file)
        validate_nh10km_adef(adef)

    def test_load_cf_nh10km_cfinfo(self):
        from pyresample.utils import load_cf_area

        def validate_nh10km_cfinfo(cfinfo, variable='ice_conc', lat='lat', lon='lon'):
            # test some of the fields
            self.assertEqual(cf_info['variable'], variable)
            self.assertEqual(cf_info['grid_mapping_variable'], 'Polar_Stereographic_Grid')
            self.assertEqual(cf_info['type_of_grid_mapping'], 'polar_stereographic')
            self.assertEqual(cf_info['lon'], lon)
            self.assertEqual(cf_info['lat'], lat)
            self.assertEqual(cf_info['x']['varname'], 'xc')
            self.assertEqual(cf_info['x']['first'], -3845.0)
            self.assertEqual(cf_info['y']['varname'], 'yc')
            self.assertEqual(cf_info['y']['last'], -5345.0)

        # prepare xarray Dataset
        cf_file = _prepare_cf_nh10km()

        # load using a variable= that is a valid grid_mapping container
        _, cf_info = load_cf_area(cf_file, 'Polar_Stereographic_Grid', y='yc', x='xc')
        validate_nh10km_cfinfo(cf_info, variable='Polar_Stereographic_Grid', lat=None, lon=None)

        # load using a variable= that has a :grid_mapping attribute
        _, cf_info = load_cf_area(cf_file, 'ice_conc')
        validate_nh10km_cfinfo(cf_info)

        # load without using a variable=
        _, cf_info = load_cf_area(cf_file)
        validate_nh10km_cfinfo(cf_info)

    def test_load_cf_llwgs84(self):
        from pyresample.utils import load_cf_area

        def validate_llwgs84(adef, cfinfo, lat='lat', lon='lon'):
            self.assertEqual(adef.shape, (19, 37))
            xc = adef.projection_x_coords
            yc = adef.projection_y_coords
            self.assertEqual(xc[0], -180., msg="Wrong x axis (index 0)")
            self.assertEqual(xc[1], -180. + 10.0, msg="Wrong x axis (index 1)")
            self.assertEqual(yc[0], -90., msg="Wrong y axis (index 0)")
            self.assertEqual(yc[1], -90. + 10.0, msg="Wrong y axis (index 1)")
            self.assertEqual(cfinfo['lon'], lon)
            self.assertEqual(cf_info['lat'], lat)
            self.assertEqual(cf_info['type_of_grid_mapping'], 'latitude_longitude')
            self.assertEqual(cf_info['x']['varname'], 'lon')
            self.assertEqual(cf_info['x']['first'], -180.)
            self.assertEqual(cf_info['y']['varname'], 'lat')
            self.assertEqual(cf_info['y']['first'], -90.)

        # prepare xarray Dataset
        cf_file = _prepare_cf_llwgs84()

        # load using a variable= that is a valid grid_mapping container
        adef, cf_info = load_cf_area(cf_file, 'crs', y='lat', x='lon')
        validate_llwgs84(adef, cf_info, lat=None, lon=None)

        # load using a variable=temp
        adef, cf_info = load_cf_area(cf_file, 'temp')
        validate_llwgs84(adef, cf_info)

        # load using a variable=None
        adef, cf_info = load_cf_area(cf_file)
        validate_llwgs84(adef, cf_info)

    def test_load_cf_llnocrs(self):
        from pyresample.utils import load_cf_area

        def validate_llnocrs(adef, cfinfo, lat='lat', lon='lon'):
            self.assertEqual(adef.shape, (19, 37))
            xc = adef.projection_x_coords
            yc = adef.projection_y_coords
            self.assertEqual(xc[0], -180., msg="Wrong x axis (index 0)")
            self.assertEqual(xc[1], -180. + 10.0, msg="Wrong x axis (index 1)")
            self.assertEqual(yc[0], -90., msg="Wrong y axis (index 0)")
            self.assertEqual(yc[1], -90. + 10.0, msg="Wrong y axis (index 1)")
            self.assertEqual(cfinfo['lon'], lon)
            self.assertEqual(cf_info['lat'], lat)
            self.assertEqual(cf_info['type_of_grid_mapping'], 'latitude_longitude')
            self.assertEqual(cf_info['x']['varname'], 'lon')
            self.assertEqual(cf_info['x']['first'], -180.)
            self.assertEqual(cf_info['y']['varname'], 'lat')
            self.assertEqual(cf_info['y']['first'], -90.)

        # prepare xarray Dataset
        cf_file = _prepare_cf_llnocrs()

        # load using a variable=temp
        adef, cf_info = load_cf_area(cf_file, 'temp')
        validate_llnocrs(adef, cf_info)

        # load using a variable=None
        adef, cf_info = load_cf_area(cf_file)
        validate_llnocrs(adef, cf_info)


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
        from pyresample.utils.cf import _is_valid_coordinate_standardname
        from pyresample.utils.cf import _valid_cf_type_of_grid_mapping

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
