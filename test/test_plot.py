import unittest
import os

from pyresample import plot, utils


class Test(unittest.TestCase):
    
    def test_ellps2axis(self):
        a, b = plot.ellps2axis('WGS84')
        self.failUnlessAlmostEqual(a, 6378137.0, 
                                   msg='Failed to get semi-major axis of ellipsis')
        self.failUnlessAlmostEqual(b, 6356752.3142451793, 
                                   msg='Failed to get semi-minor axis of ellipsis')
        
    def test_area_def2basemap(self):
        area_def = utils.parse_area_file(os.path.join(os.path.dirname(__file__), 
                                         'test_files', 'areas.cfg'), 'ease_sh')[0]
        bmap = plot.area_def2basemap(area_def)
        self.failUnless(bmap.rmajor == bmap.rminor and 
                        bmap.rmajor == 6371228.0, 
                        'Failed to create Basemap boject')
        
        