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
            projection = ("{'ellps': 'WGS84', 'lat_0': '27.12', "
                          "'lon_0': '-81.36', 'proj': 'longlat', "
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
        proj_dict = utils._proj4.proj4_str_to_dict(proj_str)
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
        proj_dict = utils._proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, grid, 12000.)

    def test_nearest_neighbor_grid_area(self):
        from pyresample import utils, geometry
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils._proj4.proj4_str_to_dict(proj_str)
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
        """Test that unicode is accepted for area creation.
        """
        from pyresample import get_area_def
        get_area_def(u"eurol", u"eurol", u"bla",
                     u'+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45',
                     1000, 1000, (-1000, -1000, 1000, 1000))

    def test_proj4_radius_parameters_provided(self):
        """Test proj4_radius_parameters with a/b."""
        from pyresample import utils
        a, b = utils._proj4.proj4_radius_parameters(
            '+proj=stere +a=6378273 +b=6356889.44891',
        )
        np.testing.assert_almost_equal(a, 6378273)
        np.testing.assert_almost_equal(b, 6356889.44891)

    def test_proj4_radius_parameters_ellps(self):
        """Test proj4_radius_parameters with ellps."""
        from pyresample import utils
        a, b = utils._proj4.proj4_radius_parameters(
            '+proj=stere +ellps=WGS84',
        )
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_radius_parameters_default(self):
        """Test proj4_radius_parameters with default parameters."""
        from pyresample import utils
        a, b = utils._proj4.proj4_radius_parameters(
            '+proj=lcc',
        )
        # WGS84
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_radius_parameters_spherical(self):
        """Test proj4_radius_parameters in case of a spherical earth."""
        from pyresample import utils
        a, b = utils._proj4.proj4_radius_parameters(
            '+proj=stere +R=6378273',
        )
        np.testing.assert_almost_equal(a, 6378273.)
        np.testing.assert_almost_equal(b, 6378273.)

    def test_convert_proj_floats(self):
        from collections import OrderedDict
        import pyresample.utils as utils

        pairs = [('proj', 'lcc'), ('ellps', 'WGS84'), ('lon_0', '-95'), ('no_defs', True)]
        expected = OrderedDict([('proj', 'lcc'), ('ellps', 'WGS84'), ('lon_0', -95.0), ('no_defs', True)])
        self.assertDictEqual(utils._proj4.convert_proj_floats(pairs), expected)

        # EPSG
        pairs = [('init', 'EPSG:4326'), ('EPSG', 4326)]
        for pair in pairs:
            expected = OrderedDict([pair])
            self.assertDictEqual(utils._proj4.convert_proj_floats([pair]), expected)

    def test_proj4_str_dict_conversion(self):
        from pyresample import utils

        proj_str = "+proj=lcc +ellps=WGS84 +lon_0=-95 +no_defs"
        proj_dict = utils._proj4.proj4_str_to_dict(proj_str)
        proj_str2 = utils._proj4.proj4_dict_to_str(proj_dict)
        proj_dict2 = utils._proj4.proj4_str_to_dict(proj_str2)
        self.assertDictEqual(proj_dict, proj_dict2)
        self.assertIsInstance(proj_dict['lon_0'], float)
        self.assertIsInstance(proj_dict2['lon_0'], float)

        # EPSG
        proj_str = '+init=EPSG:4326'
        proj_dict_exp = {'init': 'EPSG:4326'}
        proj_dict = utils._proj4.proj4_str_to_dict(proj_str)
        self.assertEqual(proj_dict, proj_dict_exp)
        self.assertEqual(utils._proj4.proj4_dict_to_str(proj_dict), proj_str)  # round-trip

        proj_str = 'EPSG:4326'
        proj_dict_exp = {'init': 'EPSG:4326'}
        proj_dict_exp2 = {'proj': 'longlat', 'datum': 'WGS84', 'no_defs': None, 'type': 'crs'}
        proj_dict = utils._proj4.proj4_str_to_dict(proj_str)
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
        area_def = utils._rasterio.get_area_def_from_raster(
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
        area_def = utils._rasterio.get_area_def_from_raster(source)
        self.assertEqual(area_def.proj_id, 'WGS 84 / Pseudo-Mercator')

    def test_get_area_def_from_raster_rotated_value_err(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.1, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        self.assertRaises(ValueError, utils._rasterio.get_area_def_from_raster, source)

    def test_get_area_def_from_raster_non_georef_value_err(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        self.assertRaises(ValueError, utils._rasterio.get_area_def_from_raster, source)

    def test_get_area_def_from_raster_non_georef_respects_proj_dict(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        proj_dict = {'init': 'epsg:3857'}
        area_def = utils._rasterio.get_area_def_from_raster(source, proj_dict=proj_dict)
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
