import os
import unittest

from pyresample import utils


def tmp(f):
    f.tmp = True
    return f

class Test(unittest.TestCase):

    def test_area_parser(self):
        ease_nh, ease_sh = utils.parse_area_file(os.path.join(os.path.dirname(__file__), 
                                                              'test_files', 
                                                              'areas.cfg'), 'ease_nh', 'ease_sh')
        
        nh_found = (ease_nh.__str__() =="""Area ID: ease_nh
Name: Arctic EASE grid
Projection ID: ease_nh
Projection: {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '90'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""")
        
        sh_found = (ease_sh.__str__() =="""Area ID: ease_sh
Name: Antarctic EASE grid
Projection ID: ease_sh
Projection: {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""")
        
        self.assertTrue(nh_found and sh_found, msg='Failed to parse areas correctly')
        
    def test_not_found_exception(self):
        self.assertRaises(utils.AreaNotFound, utils.parse_area_file, 
                          os.path.join(os.path.dirname(__file__), 'test_files', 'areas.cfg'), 
                          'no_area')
        
    def test_area_dict_to_area_def(self):
        area_dict = {'NAME':  'Antarctic EASE grid',
                     'PCS_ID': 'ease_sh',
                     'PCS_DEF': {'proj': 'laea', 'lat_0': '-90', 'lon_0': '0', 
                                 'a': '6371228.0', 'units': 'm'},
                     'XSIZE': 425,
                     'YSIZE': 425,
                     'AREA_EXTENT': (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
                     }
        area_def = utils.area_dict_to_area_def(area_dict)
        is_correct_area = (area_def.__str__() =="""Area ID: ease_sh
Name: Antarctic EASE grid
Projection ID: ease_sh
Projection: {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
Number of columns: 425
Number of rows: 425
Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)""")
        
        self.assertTrue(is_correct_area, msg='Failed to create AreaDefinition from area_dict')