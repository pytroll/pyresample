import os
import unittest

import numpy as np
import uuid

import pyresample.utils._proj4
import pyresample.utils._rasterio
from pyresample.test.utils import create_test_longitude, create_test_latitude


def tmp(f):
    f.tmp = True
    return f


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
        ease_nh, ease_sh = parse_area_file(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'),
                                           'ease_nh', 'ease_sh')

        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {'a': '6371228.0', 'lat_0': '90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(ease_nh.__str__(), nh_str)
        self.assertIsInstance(ease_nh.proj_dict['lat_0'], float)

        sh_str = """Area ID: ease_sh
Description: Antarctic EASE grid
Projection ID: ease_sh
Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(ease_sh.__str__(), sh_str)
        self.assertIsInstance(ease_sh.proj_dict['lat_0'], float)

    def test_load_area(self):
        from pyresample import load_area
        ease_nh = load_area(os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 'ease_nh')
        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection ID: ease_nh
Projection: {'a': '6371228.0', 'lat_0': '90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(nh_str, ease_nh.__str__())

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
                                     'test_radians', 'test_latlong')
        ease_nh, ease_sh, test_m, test_deg, test_rad, test_latlong = test_areas

        nh_str = """Area ID: ease_nh
Description: Arctic EASE grid
Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(ease_nh.__str__(), nh_str)

        sh_str = """Area ID: ease_sh
Description: Antarctic EASE grid
Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(ease_sh.__str__(), sh_str)

        m_str = """Area ID: test_meters
Description: test_meters
Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 850
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(test_m.__str__(), m_str)

        deg_str = """Area ID: test_degrees
Description: test_degrees
Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 850
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(test_deg.__str__(), deg_str)

        rad_str = """Area ID: test_radians
Description: test_radians
Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
Number of columns: 850
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)"""
        self.assertEqual(test_rad.rotation, 45)
        self.assertEqual(test_rad.__str__(), rad_str)

        latlong_str = """Area ID: test_latlong
Description: Basic latlong grid
Projection: {'ellps': 'WGS84', 'lat_0': '27.12', 'lon_0': '-81.36', 'proj': 'longlat'}
Number of columns: 3473
Number of rows: 4058
Area extent: (1.4186, 0.007, 1.4214, 0.0095)"""
        self.assertEqual(test_latlong.__str__(), latlong_str)

    def test_multiple_file_content(self):
        from pyresample import parse_area_file
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
        results = parse_area_file(area_list)
        self.assertEqual(len(results), 2)
        self.assertIn(results[0].area_id, ('ease_sh', 'ease_sh2'))
        self.assertIn(results[1].area_id, ('ease_sh', 'ease_sh2'))


class TestPreprocessing(unittest.TestCase):
    def test_nearest_neighbor_area_area(self):
        from pyresample import utils, geometry
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = pyresample.utils._proj4.proj4_str_to_dict(proj_str)
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
        proj_dict = pyresample.utils._proj4.proj4_str_to_dict(proj_str)
        extents = [0, 0, 1000. * 5000, 1000. * 5000]
        area_def = geometry.AreaDefinition('CONUS', 'CONUS', 'CONUS',
                                           proj_dict, 400, 500, extents)
        rows, cols = utils.generate_nearest_neighbour_linesample_arrays(area_def, grid, 12000.)

    def test_nearest_neighbor_grid_area(self):
        from pyresample import utils, geometry
        proj_str = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs"
        proj_dict = pyresample.utils._proj4.proj4_str_to_dict(proj_str)
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
        a, b = pyresample.utils._proj4.proj4_radius_parameters(
            '+proj=stere +a=6378273 +b=6356889.44891',
        )
        np.testing.assert_almost_equal(a, 6378273)
        np.testing.assert_almost_equal(b, 6356889.44891)

    def test_proj4_radius_parameters_ellps(self):
        from pyresample import utils
        a, b = pyresample.utils._proj4.proj4_radius_parameters(
            '+proj=stere +ellps=WGS84',
        )
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_radius_parameters_default(self):
        from pyresample import utils
        a, b = pyresample.utils._proj4.proj4_radius_parameters(
            '+proj=lcc',
        )
        # WGS84
        np.testing.assert_almost_equal(a, 6378137.)
        np.testing.assert_almost_equal(b, 6356752.314245, decimal=6)

    def test_proj4_str_dict_conversion(self):
        from pyresample import utils
        proj_str = "+proj=lcc +ellps=WGS84 +lon_0=-95 +no_defs"
        proj_dict = pyresample.utils._proj4.proj4_str_to_dict(proj_str)
        proj_str2 = pyresample.utils._proj4.proj4_dict_to_str(proj_dict)
        proj_dict2 = pyresample.utils._proj4.proj4_str_to_dict(proj_str2)
        self.assertDictEqual(proj_dict, proj_dict2)
        self.assertIsInstance(proj_dict['lon_0'], float)
        self.assertIsInstance(proj_dict2['lon_0'], float)

    def test_def2yaml_converter(self):
        from pyresample import parse_area_file, convert_def_to_yaml
        import tempfile
        def_file = os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg')
        filehandle, yaml_file = tempfile.mkstemp()
        os.close(filehandle)
        try:
            convert_def_to_yaml(def_file, yaml_file)
            areas_new = set(parse_area_file(yaml_file))
            areas = parse_area_file(def_file)
            for area in areas:
                area.proj_dict.pop('units', None)
            areas_old = set(areas)
            self.assertEqual(areas_new, areas_old)
        finally:
            os.remove(yaml_file)

    def test_get_area_def_from_raster(self):
        from rasterio.crs import CRS
        from affine import Affine
        x_size = 791
        y_size = 718
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        crs = CRS(init='epsg:3857')
        source = tmptiff(x_size, y_size, transform, crs=crs)
        area_id = 'area_id'
        proj_id = 'proj_id'
        name = 'name'
        area_def = pyresample.utils._rasterio.get_area_def_from_raster(
            source, area_id=area_id, name=name, proj_id=proj_id)
        self.assertEqual(area_def.area_id, area_id)
        self.assertEqual(area_def.proj_id, proj_id)
        self.assertEqual(area_def.name, name)
        self.assertEqual(area_def.width, x_size)
        self.assertEqual(area_def.height, y_size)
        self.assertDictEqual(crs.to_dict(), area_def.proj_dict)
        self.assertTupleEqual(area_def.area_extent, (transform.c, transform.f + transform.e * y_size,
                                                     transform.c + transform.a * x_size, transform.f))

    def test_get_area_def_from_raster_extracts_proj_id(self):
        from rasterio.crs import CRS
        from pyresample import utils
        crs = CRS(init='epsg:3857')
        source = tmptiff(crs=crs)
        area_def = pyresample.utils._rasterio.get_area_def_from_raster(source)
        self.assertEqual(area_def.proj_id, 'WGS 84 / Pseudo-Mercator')

    def test_get_area_def_from_raster_rotated_value_err(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.1, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        self.assertRaises(ValueError, pyresample.utils._rasterio.get_area_def_from_raster, source)

    def test_get_area_def_from_raster_non_georef_value_err(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        self.assertRaises(ValueError, pyresample.utils._rasterio.get_area_def_from_raster, source)

    def test_get_area_def_from_raster_non_georef_respects_proj_dict(self):
        from pyresample import utils
        from affine import Affine
        transform = Affine(300.0379266750948, 0.0, 101985.0,
                           0.0, -300.041782729805, 2826915.0)
        source = tmptiff(transform=transform)
        proj_dict = {'init': 'epsg:3857'}
        area_def = pyresample.utils._rasterio.get_area_def_from_raster(source, proj_dict=proj_dict)
        self.assertDictEqual(area_def.proj_dict, proj_dict)


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


if __name__ == '__main__':
    unittest.main()
