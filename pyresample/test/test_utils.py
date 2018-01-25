import os
import unittest

import numpy as np

from pyresample.test.utils import create_test_longitude, create_test_latitude


def tmp(f):
    f.tmp = True
    return f


class TestLegacyAreaParser(unittest.TestCase):
    def test_area_parser_legacy(self):
        """Test legacy area parser."""
        from pyresample import utils
        ease_nh, ease_sh = utils.parse_area_file(os.path.join(os.path.dirname(__file__),
                                                              'test_files',
                                                              'areas.cfg'), 'ease_nh', 'ease_sh')

        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {'a': '6371228.0', 'lat_0': '90', 'lon_0': '0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEquals(ease_nh.__str__(), nh_str)

        sh_str = """Area ID: ease_sh
Description: Antarctic EASE grid
Projection ID: ease_sh
Projection: {'a': '6371228.0', 'lat_0': '-90', 'lon_0': '0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEquals(ease_sh.__str__(), sh_str)

    def test_load_area(self):
        from pyresample import utils
        ease_nh = utils.load_area(os.path.join(os.path.dirname(__file__),
                                               'test_files',
                                               'areas.cfg'), 'ease_nh')
        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {'a': '6371228.0', 'lat_0': '90', 'lon_0': '0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEquals(nh_str, ease_nh.__str__())

    def test_not_found_exception(self):
        from pyresample import utils
        self.assertRaises(utils.AreaNotFound, utils.parse_area_file,
                          os.path.join(
                              os.path.dirname(__file__), 'test_files', 'areas.cfg'),
                          'no_area')


class TestYAMLAreaParser(unittest.TestCase):
    def test_area_parser_yaml(self):
        """Test YAML area parser."""
        from pyresample import utils
        ease_nh, ease_sh = utils.parse_area_file(os.path.join(os.path.dirname(__file__),
                                                              'test_files',
                                                              'areas.yaml'),
                                                 'ease_nh', 'ease_sh')

        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection: {'a': '6371228.0', 'lat_0': '90', 'lon_0': '0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEquals(ease_nh.__str__(), nh_str)

        sh_str = """Area ID: ease_sh
Description: Antarctic EASE grid
Projection: {'a': '6371228.0', 'lat_0': '-90', 'lon_0': '0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEquals(ease_sh.__str__(), sh_str)

    def test_multiple_file_content(self):
        from pyresample import utils
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
        results = utils.parse_area_file(area_list)
        self.assertEquals(len(results), 2)
        self.assertIn(results[0].area_id, ('ease_sh', 'ease_sh2'))
        self.assertIn(results[1].area_id, ('ease_sh', 'ease_sh2'))


class TestPreprocessing(unittest.TestCase):
    def test_nearest_neighbor_area_area(self):
        from pyresample import utils, geometry
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4_str_to_dict(proj_str)
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
        proj_dict = utils.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, grid, 12000.)

    def test_nearest_neighbor_grid_area(self):
        from pyresample import utils, geometry
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = utils.proj4_str_to_dict(proj_str)
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
        from pyresample import utils
        utils.get_area_def(u"eurol", u"eurol", u"bla",
                           u'+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45',
                           1000, 1000, (-1000, -1000, 1000, 1000))

    def test_proj4_radius_parameters_provided(self):
        from pyresample import utils
        a, b = utils.proj4_radius_parameters(
            '+proj=stere +a=6378273 +b=6356889.44891',
        )
        np.testing.assert_almost_equal(a, 6378273)
        np.testing.assert_almost_equal(b, 6356889.44891)

    def test_proj4_radius_parameters_ellps(self):
        from pyresample import utils
        a, b = utils.proj4_radius_parameters(
            '+proj=stere +ellps=WGS84',
        )
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_radius_parameters_default(self):
        from pyresample import utils
        a, b = utils.proj4_radius_parameters(
            '+proj=lcc',
        )
        # WGS84
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_str_dict_conversion(self):
        from pyresample import utils
        proj_str = "+proj=lcc +ellps=WGS84 +lon_0=-95 +no_defs"
        proj_dict = utils.proj4_str_to_dict(proj_str)
        proj_str2 = utils.proj4_dict_to_str(proj_dict)
        proj_dict2 = utils.proj4_str_to_dict(proj_str2)
        self.assertDictEqual(proj_dict, proj_dict2)


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestLegacyAreaParser))
    mysuite.addTest(loader.loadTestsFromTestCase(TestYAMLAreaParser))
    mysuite.addTest(loader.loadTestsFromTestCase(TestPreprocessing))
    mysuite.addTest(loader.loadTestsFromTestCase(TestMisc))

    return mysuite
