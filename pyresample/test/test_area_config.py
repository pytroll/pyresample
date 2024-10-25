#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015-2022 Pyresample developers
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
"""Test the area_config functions."""

import io
import os
import pathlib
import unittest

import numpy as np
import pytest

from pyresample.test.utils import TEST_FILES_PATH


class TestLegacyAreaParser(unittest.TestCase):
    """Test legacy .cfg parsing."""

    def test_area_parser_legacy(self):
        """Test legacy area parser."""
        from pyresample import parse_area_file
        ease_nh, ease_sh = parse_area_file(os.path.join(TEST_FILES_PATH, 'areas.cfg'),
                                           'ease_nh', 'ease_sh')

        # pyproj 2.0+ adds some extra parameters
        projection = ("{'R': '6371228', 'lat_0': '90', 'lon_0': '0', "
                      "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                      "'units': 'm', 'x_0': '0', 'y_0': '0'}")
        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""".format(projection)
        self.assertEqual(ease_nh.__str__(), nh_str)
        self.assertIsInstance(ease_nh.proj_dict['lat_0'], (int, float))

        projection = ("{'R': '6371228', 'lat_0': '-90', 'lon_0': '0', "
                      "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                      "'units': 'm', 'x_0': '0', 'y_0': '0'}")
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
        ease_nh = load_area(os.path.join(TEST_FILES_PATH, 'areas.cfg'), 'ease_nh')
        # pyproj 2.0+ adds some extra parameters
        projection = ("{'R': '6371228', 'lat_0': '90', 'lon_0': '0', "
                      "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                      "'units': 'm', 'x_0': '0', 'y_0': '0'}")
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
                          os.path.join(TEST_FILES_PATH, 'areas.cfg'), 'no_area')

    def test_commented(self):
        from pyresample import parse_area_file
        areas = parse_area_file(os.path.join(TEST_FILES_PATH, 'areas.cfg'))
        self.assertNotIn('commented', [area.name for area in areas])


class TestYAMLAreaParser(unittest.TestCase):
    """Test YAML parsing."""

    def test_area_parser_yaml(self):
        """Test YAML area parser."""
        from pyresample import parse_area_file
        test_area_file = os.path.join(TEST_FILES_PATH, 'areas.yaml')
        test_areas = parse_area_file(test_area_file, 'ease_nh', 'ease_sh', 'test_meters', 'test_degrees',
                                     'test_latlong')
        ease_nh, ease_sh, test_m, test_deg, test_latlong = test_areas

        # pyproj 2.0+ adds some extra parameters
        projection = ("{'R': '6371228', 'lat_0': '-90', 'lon_0': '0', "
                      "'no_defs': 'None', 'proj': 'laea', 'type': 'crs', "
                      "'units': 'm', 'x_0': '0', 'y_0': '0'}")
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

        # pyproj 2.0+ adds some extra parameters
        projection = ("{'ellps': 'WGS84', 'no_defs': 'None', "
                      "'pm': '-81.36', 'proj': 'longlat', "
                      "'type': 'crs'}")
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
        test_area_file = os.path.join(TEST_FILES_PATH, 'areas.yaml')
        test_area = parse_area_file(test_area_file, 'test_dynamic_resolution')[0]

        self.assertIsInstance(test_area, DynamicAreaDefinition)
        self.assertTrue(hasattr(test_area, 'resolution'))
        self.assertEqual(test_area.resolution, (1000.0, 1000.0))

        # lat/lon
        from pyresample import parse_area_file
        from pyresample.geometry import DynamicAreaDefinition
        test_area_file = os.path.join(TEST_FILES_PATH, 'areas.yaml')
        test_area = parse_area_file(test_area_file, 'test_dynamic_resolution_ll')[0]

        self.assertIsInstance(test_area, DynamicAreaDefinition)
        self.assertTrue(hasattr(test_area, 'resolution'))
        np.testing.assert_allclose(test_area.resolution, (1.0, 1.0))

    def test_dynamic_area_parser_passes_resolution(self):
        """Test that the resolution from the file is passed to a dynamic area."""
        from pyresample import parse_area_file
        test_area_file = os.path.join(TEST_FILES_PATH, 'areas.yaml')
        test_area = parse_area_file(test_area_file, 'omerc_bb_1000')[0]
        assert test_area.resolution == (1000, 1000)
        assert test_area.optimize_projection

    def test_dynamic_area_parser_opt_projection_nores(self):
        """Test that a dynamic area definition can be frozen when no resolution is passed via YAML."""
        from pyresample import SwathDefinition, parse_area_file
        test_area_file = os.path.join(TEST_FILES_PATH, 'areas.yaml')
        test_area = parse_area_file(test_area_file, 'omerc_bb_nores')[0]
        assert test_area.resolution is None
        assert test_area.optimize_projection
        lons, lats = np.meshgrid(np.arange(10, 20), np.arange(10, 20))
        swath_def = SwathDefinition(lons, lats)
        test_area.freeze(swath_def)

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


def test_area_def_rst_list():
    """Test output of rst generation from area list."""
    import unittest.mock as mock

    from pyresample.area_config import generate_area_def_rst_list

    areas = """msg_seviri_fes_3km:
  description:
    MSG SEVIRI Full Earth Scanning service area definition
    with 3 km resolution
  projection:
    proj: geos
    lon_0: 0.0
    a: 6378169.0
    b: 6356583.8
    h: 35785831.0
  shape:
    height: 3712
    width: 3712
  area_extent:
    lower_left_xy: [-5570248.686685662, -5567248.28340708]
    upper_right_xy: [5567248.28340708,   5570248.686685662]

australia:
  description: australia
  projection:
    proj: merc
    lat_0: -27.5
    lon_0: 132.5
    ellps: WGS84
  shape:
    height: 895
    width: 1001
  area_extent:
    lower_left_xy: [-2504688.5428486555, -5591295.9185533915]
    upper_right_xy: [2504688.5428486555, -1111475.102852225]"""

    with open("test_areas.yaml", "w") as file:
        file.write(areas)

    with mock.patch('pyresample.area_config.area_repr') as mock_area_repr:
        generate_area_def_rst_list("test_areas.yaml")
        assert mock_area_repr.call_count == 2
        call_args = mock_area_repr.call_args_list
        # check that static files are included for the first area
        assert call_args[0][1]['include_static_files']
        # check that static files are not included for the second area
        assert not call_args[1][1]['include_static_files']


def test_unused_params_warn():
    """Test unused parameters produce a warning."""
    from pyresample.area_config import load_area_from_string

    yaml_str = """ease_sh2:
  description: Antarctic EASE grid
  projection:
    a: 6371228.0
    units: m
    lon_0: 0
    proj: laea
    lat_0: -90
  revolution: 1000
  area_extent:
    lower_left_xy: [-5326849.0625, -5326849.0625]
    upper_right_xy: [5326849.0625, 5326849.0625]
    units: m"""
    with pytest.warns(UserWarning, match=r"Unused/unexpected area definition parameter.*revolution"):
        load_area_from_string(yaml_str)
