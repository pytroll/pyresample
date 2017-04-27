import os
import unittest

import numpy as np

from pyresample import utils


def tmp(f):
    f.tmp = True
    return f


class Test(unittest.TestCase):

    def test_area_parser_legacy(self):
        """Test legacy area parser."""
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

    def test_area_parser_yaml(self):
        """Test YAML area parser."""
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

    def test_load_area(self):
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
        self.assertRaises(utils.AreaNotFound, utils.parse_area_file,
                          os.path.join(
                              os.path.dirname(__file__), 'test_files', 'areas.cfg'),
                          'no_area')

    def test_wrap_longitudes(self):
        # test that we indeed wrap to [-180:+180[
        step = 60
        lons = np.arange(-360, 360 + step, step)
        self.assertTrue(
            (lons.min() < -180) and (lons.max() >= 180) and (+180 in lons))
        wlons = utils.wrap_longitudes(lons)
        self.assertFalse(
            (wlons.min() < -180) or (wlons.max() >= 180) or (+180 in wlons))

    def test_unicode_proj4_string(self):
        """Test that unicode is accepted for area creation.
        """
        utils.get_area_def(u"eurol", u"eurol", u"bla",
                           u'+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45',
                           1000, 1000, (-1000, -1000, 1000, 1000))


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
