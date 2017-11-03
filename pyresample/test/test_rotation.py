import unittest, os
from pyresample.utils import load_area

class TestProjRotation(unittest.TestCase):

    def test_rotation_legacy (self):
        legacyDef = """REGION: regionB {
        NAME:          regionB
        PCS_ID:        regionB
        PCS_DEF:       proj=merc, lon_0=-34, k=1, x_0=0, y_0=0, a=6378137, b=6378137
        XSIZE:         800
        YSIZE:         548
        ROTATION:      -45
        AREA_EXTENT:   (-7761424.714818418, -4861746.639279127, 11136477.43264252, 8236799.845095873)
        };"""
        flegacy = "/tmp/TestProjRotation_test_rotation_legacy.txt"
        f = open(flegacy,"w") 
        f.write(legacyDef)
        f.close()
        test_area = load_area(flegacy, 'regionB')     
        self.assertEqual(test_area.rotation, -45)
        os.remove(flegacy)

    def test_rotation_yaml (self):
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
        fyaml = "/tmp/TestProjRotation_test_rotation_yaml.txt"
        f = open(fyaml,"w")
        f.write(yamlDef)
        f.close()
        test_area = load_area(fyaml, 'regionB')
        self.assertEqual(test_area.rotation, -45)
        os.remove(fyaml)

    def test_norotation_legacy (self):
        legacyDef = """REGION: regionB {
        NAME:          regionB
        PCS_ID:        regionB
        PCS_DEF:       proj=merc, lon_0=-34, k=1, x_0=0, y_0=0, a=6378137, b=6378137
        XSIZE:         800
        YSIZE:         548
        AREA_EXTENT:   (-7761424.714818418, -4861746.639279127, 11136477.43264252, 8236799.845095873)
        };"""
        flegacy = "/tmp/TestProjRotation_test_rotation_legacy.txt"
        f = open(flegacy,"w")
        f.write(legacyDef)
        f.close()
        test_area = load_area(flegacy, 'regionB')
        self.assertEqual(test_area.rotation, 0)
        os.remove(flegacy)

    def test_norotation_yaml (self):
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
        fyaml = "/tmp/TestProjRotation_test_rotation_yaml.txt"
        f = open(fyaml,"w")
        f.write(yamlDef)
        f.close()
        test_area = load_area(fyaml, 'regionB')
        self.assertEqual(test_area.rotation, 0)
        os.remove(fyaml)



if __name__ == '__main__':
    unittest.main()
