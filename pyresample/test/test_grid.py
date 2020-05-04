import unittest

import numpy as np

from pyresample import grid, geometry, utils


def mp(f):
    f.mp = True
    return f


def tmp(f):
    f.tmp = True
    return f


class Test(unittest.TestCase):

    area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
                                       {'a': '6378144.0',
                                        'b': '6356759.0',
                                        'lat_0': '50.00',
                                        'lat_ts': '50.00',
                                        'lon_0': '8.00',
                                        'proj': 'stere'},
                                       800,
                                       800,
                                       [-1370912.72,
                                           -909968.64000000001,
                                           1029087.28,
                                           1490031.3600000001])

    area_def2 = geometry.AreaDefinition('areaD2', 'Europe (3km, HRV, VTC)', 'areaD2',
                                        {'a': '6378144.0',
                                         'b': '6356759.0',
                                         'lat_0': '50.00',
                                         'lat_ts': '50.00',
                                         'lon_0': '8.00',
                                         'proj': 'stere'},
                                        5,
                                        5,
                                        [-1370912.72,
                                            -909968.64000000001,
                                            1029087.28,
                                            1490031.3600000001])

    msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
                                       'msg_full',
                                       {'a': '6378169.0',
                                        'b': '6356584.0',
                                        'h': '35785831.0',
                                        'lon_0': '0',
                                        'proj': 'geos'},
                                       3712,
                                       3712,
                                       [-5568742.4000000004,
                                           -5568742.4000000004,
                                           5568742.4000000004,
                                           5568742.4000000004]
                                       )

    def test_linesample(self):
        data = np.fromfunction(lambda y, x: y * x, (40, 40))
        rows = np.array([[1, 2], [3, 4]])
        cols = np.array([[25, 26], [27, 28]])
        res = grid.get_image_from_linesample(rows, cols, data)
        expected = np.array([[25., 52.], [81., 112.]])
        self.assertTrue(np.array_equal(res, expected), 'Linesample failed')

    def test_linesample_multi(self):
        data1 = np.fromfunction(lambda y, x: y * x, (40, 40))
        data2 = np.fromfunction(lambda y, x: 2 * y * x, (40, 40))
        data3 = np.fromfunction(lambda y, x: 3 * y * x, (40, 40))
        data = np.zeros((40, 40, 3))
        data[:, :, 0] = data1
        data[:, :, 1] = data2
        data[:, :, 2] = data3
        rows = np.array([[1, 2], [3, 4]])
        cols = np.array([[25, 26], [27, 28]])
        res = grid.get_image_from_linesample(rows, cols, data)
        expected = np.array([[[25., 50., 75.],
                              [52., 104., 156.]],
                             [[81., 162., 243.],
                              [112.,  224.,  336.]]])
        self.assertTrue(np.array_equal(res, expected), 'Linesample failed')

    def test_from_latlon(self):
        data = np.fromfunction(lambda y, x: y * x, (800, 800))
        lons = np.fromfunction(lambda y, x: x, (10, 10))
        lats = np.fromfunction(lambda y, x: 50 - (5.0 / 10) * y, (10, 10))
        #source_def = grid.AreaDefinition.get_from_area_def(self.area_def)
        source_def = self.area_def
        res = grid.get_image_from_lonlats(lons, lats, source_def, data)
        expected = np.array([[129276.,  141032.,  153370.,  165804.,  178334.,  190575.,
                              202864.,  214768.,  226176.,  238080.],
                             [133056.,  146016.,  158808.,  171696.,  184320.,  196992.,
                              209712.,  222480.,  234840.,  247715.],
                             [137026.,  150150.,  163370.,  177215.,  190629.,  203756.,
                              217464.,  230256.,  243048.,  256373.],
                             [140660.,  154496.,  168714.,  182484.,  196542.,  210650.,
                              224257.,  238464.,  251712.,  265512.],
                             [144480.,  158484.,  173148.,  187912.,  202776.,  217358.,
                              231990.,  246240.,  259920.,  274170.],
                             [147968.,  163261.,  178398.,  193635.,  208616.,  223647.,
                              238728.,  253859.,  268584.,  283898.],
                             [151638.,  167121.,  182704.,  198990.,  214775.,  230280.,
                              246442.,  261617.,  276792.,  292574.],
                             [154980.,  171186.,  187860.,  204016.,  220542.,  237120.,
                              253125.,  269806.,  285456.,  301732.],
                             [158500.,  175536.,  192038.,  209280.,  226626.,  243697.,
                              260820.,  277564.,  293664.,  310408.],
                             [161696.,  179470.,  197100.,  214834.,  232320.,  250236.,
                              267448.,  285090.,  302328.,  320229.]])
        self.assertTrue(
            np.array_equal(res, expected), 'Sampling from lat lon failed')

    def test_proj_coords(self):
        #res = grid.get_proj_coords(self.area_def2)
        res = self.area_def2.get_proj_coords()
        cross_sum = res[0].sum() + res[1].sum()
        expected = 2977965.9999999963
        self.assertAlmostEqual(
            cross_sum, expected, msg='Calculation of proj coords failed')

    def test_latlons(self):
        #res = grid.get_lonlats(self.area_def2)
        res = self.area_def2.get_lonlats()
        cross_sum = res[0].sum() + res[1].sum()
        expected = 1440.8280578215431
        self.assertAlmostEqual(
            cross_sum, expected, msg='Calculation of lat lons failed')

    @mp
    def test_latlons_mp(self):
        #res = grid.get_lonlats(self.area_def2, nprocs=2)
        res = self.area_def2.get_lonlats(nprocs=2)
        cross_sum = res[0].sum() + res[1].sum()
        expected = 1440.8280578215431
        self.assertAlmostEqual(
            cross_sum, expected, msg='Calculation of lat lons failed')

    def test_resampled_image(self):
        data = np.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        target_def = self.area_def
        source_def = self.msg_area
        res = grid.get_resampled_image(
            target_def, source_def, data, segments=1)
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(
            cross_sum, expected, msg='Resampling of image failed')

    def test_resampled_image_masked(self):
        # Generate test image with masked elements
        data = np.ma.ones(self.msg_area.shape)
        data.mask = np.zeros(data.shape)
        data.mask[253:400, 1970:2211] = 1

        # Resample image using multiple segments
        target_def = self.area_def
        source_def = self.msg_area
        res = grid.get_resampled_image(
            target_def, source_def, data, segments=4, fill_value=None)

        # Make sure the mask has been preserved
        self.assertGreater(res.mask.sum(), 0,
                           msg='Resampling did not preserve the mask')

    @tmp
    def test_generate_linesample(self):
        data = np.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        row_indices, col_indices = utils.generate_quick_linesample_arrays(self.msg_area,
                                                                          self.area_def)
        res = data[row_indices, col_indices]
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(
            cross_sum, expected, msg='Generate linesample failed')
        self.assertFalse(row_indices.dtype != np.uint16 or col_indices.dtype != np.uint16,
                         'Generate linesample failed. Downcast to uint16 expected')

    @mp
    def test_resampled_image_mp(self):
        data = np.fromfunction(lambda y, x: y * x * 10 ** -6, (3712, 3712))
        target_def = self.area_def
        source_def = self.msg_area
        res = grid.get_resampled_image(
            target_def, source_def, data, nprocs=2, segments=1)
        cross_sum = res.sum()
        expected = 399936.39392500359
        self.assertAlmostEqual(
            cross_sum, expected, msg='Resampling of image mp failed')

    def test_single_lonlat(self):
        lon, lat = self.area_def.get_lonlat(400, 400)
        self.assertAlmostEqual(
            lon, 5.5028467120975835, msg='Resampling of single lon failed')
        self.assertAlmostEqual(
            lat, 52.566998432390619, msg='Resampling of single lat failed')

    def test_proj4_string(self):
        """Test 'proj_str' property of AreaDefinition."""
        from pyresample.utils import is_pyproj2
        proj4_string = self.area_def.proj_str
        expected_string = '+a=6378144.0 +b=6356759.0 +lat_ts=50.0 +lon_0=8.0 +proj=stere +lat_0=50.0'
        if is_pyproj2():
            expected_string = '+a=6378144 +k=1 +lat_0=50 +lon_0=8 ' \
                              '+no_defs +proj=stere +rf=298.253168108487 ' \
                              '+type=crs +units=m +x_0=0 +y_0=0'
        self.assertEqual(
            frozenset(proj4_string.split()), frozenset(expected_string.split()))
