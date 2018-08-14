from __future__ import with_statement

import numpy as np
import unittest
import math

from pyresample.spherical_geometry import Coordinate, Arc
from pyresample import geometry


class TestOverlap(unittest.TestCase):

    """Testing overlapping functions in pyresample.
    """

    def assert_raises(self, exception, call_able, *args):
        """assertRaises() has changed from py2.6 to 2.7! Here is an attempt to
        cover both"""
        import sys
        if sys.version_info < (2, 7):
            self.assertRaises(exception, call_able, *args)
        else:
            with self.assertRaises(exception):
                call_able(*args)

    def test_inside(self):
        """Testing if a point is inside an area.
        """
        lons = np.array([[-11, 11], [-11, 11]])
        lats = np.array([[11, 11], [-11, -11]])
        area = geometry.SwathDefinition(lons, lats)

        point = Coordinate(0, 0)

        self.assertTrue(point in area)

        point = Coordinate(0, 12)
        self.assertFalse(point in area)

        lons = np.array([[-179, 179], [-179, 179]])
        lats = np.array([[1, 1], [-1, -1]])
        area = geometry.SwathDefinition(lons, lats)

        point = Coordinate(180, 0)
        self.assertTrue(point in area)

        point = Coordinate(180, 12)
        self.assertFalse(point in area)

        point = Coordinate(-180, 12)
        self.assertFalse(point in area)

        self.assert_raises(ValueError, Coordinate, 0, 192)

        self.assert_raises(ValueError, Coordinate, 15, -91)

        # case of the north pole
        lons = np.array([[0, 90], [-90, 180]])
        lats = np.array([[89, 89], [89, 89]])
        area = geometry.SwathDefinition(lons, lats)

        point = Coordinate(90, 90)
        self.assertTrue(point in area)

    def test_overlaps(self):
        """Test if two areas overlap.
        """
        lons1 = np.array([[0, 90], [-90, 180]])
        lats1 = np.array([[89, 89], [89, 89]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[45, 135], [-45, -135]])
        lats2 = np.array([[89, 89], [89, 89]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertTrue(area1.overlaps(area2))
        self.assertTrue(area2.overlaps(area1))

        lons1 = np.array([[0, 45], [135, 90]])
        lats1 = np.array([[89, 89], [89, 89]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[180, -135], [-45, -90]])
        lats2 = np.array([[89, 89], [89, 89]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertFalse(area1.overlaps(area2))
        self.assertFalse(area2.overlaps(area1))

        lons1 = np.array([[-1, 1], [-1, 1]])
        lats1 = np.array([[1, 1], [-1, -1]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[0, 2], [0, 2]])
        lats2 = np.array([[0, 0], [2, 2]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertTrue(area1.overlaps(area2))
        self.assertTrue(area2.overlaps(area1))

        lons1 = np.array([[-1, 0], [-1, 0]])
        lats1 = np.array([[1, 2], [-1, 0]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[1, 2], [1, 2]])
        lats2 = np.array([[1, 2], [-1, 0]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertFalse(area1.overlaps(area2))
        self.assertFalse(area2.overlaps(area1))

    def test_overlap_rate(self):
        """Test how much two areas overlap.
        """

        lons1 = np.array([[-1, 1], [-1, 1]])
        lats1 = np.array([[1, 1], [-1, -1]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[0, 2], [0, 2]])
        lats2 = np.array([[0, 0], [2, 2]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertAlmostEqual(area1.overlap_rate(area2), 0.25, 3)
        self.assertAlmostEqual(area2.overlap_rate(area1), 0.25, 3)

        lons1 = np.array([[82.829699999999974, 36.888300000000001],
                          [98.145499999999984, 2.8773]])
        lats1 = np.array([[60.5944, 52.859999999999999],
                          [80.395899999999997, 66.7547]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[7.8098183315148422, 26.189349044600252],
                          [7.8098183315148422, 26.189349044600252]])
        lats2 = np.array([[62.953206630716465, 62.953206630716465],
                          [53.301561187195546, 53.301561187195546]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertAlmostEqual(area1.overlap_rate(area2), 0.07, 2)
        self.assertAlmostEqual(area2.overlap_rate(area1), 0.012, 3)

        lons1 = np.array([[82.829699999999974, 36.888300000000001],
                          [98.145499999999984, 2.8773]])
        lats1 = np.array([[60.5944, 52.859999999999999],
                          [80.395899999999997, 66.7547]])
        area1 = geometry.SwathDefinition(lons1, lats1)

        lons2 = np.array([[12.108984194981202, 30.490647126520301],
                          [12.108984194981202, 30.490647126520301]])
        lats2 = np.array([[65.98228561983025, 65.98228561983025],
                          [57.304862819933433, 57.304862819933433]])
        area2 = geometry.SwathDefinition(lons2, lats2)

        self.assertAlmostEqual(area1.overlap_rate(area2), 0.509, 2)
        self.assertAlmostEqual(area2.overlap_rate(area1), 0.0685, 3)


class TestSphereGeometry(unittest.TestCase):

    """Testing sphere geometry from this module.
    """

    def test_angle(self):
        """Testing the angle value between two arcs.
        """

        base = 0

        p0_ = Coordinate(base, base)
        p1_ = Coordinate(base, base + 1)
        p2_ = Coordinate(base + 1, base)
        p3_ = Coordinate(base, base - 1)
        p4_ = Coordinate(base - 1, base)

        arc1 = Arc(p0_, p1_)
        arc2 = Arc(p0_, p2_)
        arc3 = Arc(p0_, p3_)
        arc4 = Arc(p0_, p4_)

        self.assertAlmostEqual(arc1.angle(arc2), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc2.angle(arc3), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc3.angle(arc4), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc4.angle(arc1), math.pi / 2,
                               msg="this should be pi/2")

        self.assertAlmostEqual(arc1.angle(arc4), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc4.angle(arc3), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc3.angle(arc2), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc2.angle(arc1), -math.pi / 2,
                               msg="this should be -pi/2")

        self.assertAlmostEqual(arc1.angle(arc3), math.pi,
                               msg="this should be pi")
        self.assertAlmostEqual(arc3.angle(arc1), math.pi,
                               msg="this should be pi")
        self.assertAlmostEqual(arc2.angle(arc4), math.pi,
                               msg="this should be pi")
        self.assertAlmostEqual(arc4.angle(arc2), math.pi,
                               msg="this should be pi")

        p5_ = Coordinate(base + 1, base + 1)
        p6_ = Coordinate(base + 1, base - 1)
        p7_ = Coordinate(base - 1, base - 1)
        p8_ = Coordinate(base - 1, base + 1)

        arc5 = Arc(p0_, p5_)
        arc6 = Arc(p0_, p6_)
        arc7 = Arc(p0_, p7_)
        arc8 = Arc(p0_, p8_)

        self.assertAlmostEqual(arc1.angle(arc5), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc5.angle(arc2), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc2.angle(arc6), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc6.angle(arc3), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc3.angle(arc7), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc7.angle(arc4), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc4.angle(arc8), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc8.angle(arc1), math.pi / 4, 3,
                               msg="this should be pi/4")

        self.assertAlmostEqual(arc1.angle(arc6), 3 * math.pi / 4, 3,
                               msg="this should be 3pi/4")

        c0_ = Coordinate(180, 0)
        c1_ = Coordinate(180, 1)
        c2_ = Coordinate(-179, 0)
        c3_ = Coordinate(-180, -1)
        c4_ = Coordinate(179, 0)

        arc1 = Arc(c0_, c1_)
        arc2 = Arc(c0_, c2_)
        arc3 = Arc(c0_, c3_)
        arc4 = Arc(c0_, c4_)

        self.assertAlmostEqual(arc1.angle(arc2), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc2.angle(arc3), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc3.angle(arc4), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc4.angle(arc1), math.pi / 2,
                               msg="this should be pi/2")

        self.assertAlmostEqual(arc1.angle(arc4), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc4.angle(arc3), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc3.angle(arc2), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc2.angle(arc1), -math.pi / 2,
                               msg="this should be -pi/2")

        # case of the north pole

        c0_ = Coordinate(0, 90)
        c1_ = Coordinate(0, 89)
        c2_ = Coordinate(-90, 89)
        c3_ = Coordinate(180, 89)
        c4_ = Coordinate(90, 89)

        arc1 = Arc(c0_, c1_)
        arc2 = Arc(c0_, c2_)
        arc3 = Arc(c0_, c3_)
        arc4 = Arc(c0_, c4_)

        self.assertAlmostEqual(arc1.angle(arc2), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc2.angle(arc3), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc3.angle(arc4), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc4.angle(arc1), math.pi / 2,
                               msg="this should be pi/2")

        self.assertAlmostEqual(arc1.angle(arc4), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc4.angle(arc3), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc3.angle(arc2), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc2.angle(arc1), -math.pi / 2,
                               msg="this should be -pi/2")

        self.assertAlmostEqual(Arc(c1_, c2_).angle(arc1), math.pi / 4, 3,
                               msg="this should be pi/4")

        self.assertAlmostEqual(Arc(c4_, c3_).angle(arc4), -math.pi / 4, 3,
                               msg="this should be -pi/4")

        self.assertAlmostEqual(Arc(c1_, c4_).angle(arc1), -math.pi / 4, 3,
                               msg="this should be -pi/4")

    def test_intersects(self):
        """Test if two arcs intersect.
        """
        p0_ = Coordinate(0, 0)
        p1_ = Coordinate(0, 1)
        p2_ = Coordinate(1, 0)
        p3_ = Coordinate(0, -1)
        p4_ = Coordinate(-1, 0)
        p5_ = Coordinate(1, 1)
        p6_ = Coordinate(1, -1)

        arc13 = Arc(p1_, p3_)
        arc24 = Arc(p2_, p4_)

        arc32 = Arc(p3_, p2_)
        arc41 = Arc(p4_, p1_)

        arc40 = Arc(p4_, p0_)
        arc56 = Arc(p5_, p6_)

        arc45 = Arc(p4_, p5_)
        arc02 = Arc(p0_, p2_)

        arc35 = Arc(p3_, p5_)

        self.assertTrue(arc13.intersects(arc24))

        self.assertFalse(arc32.intersects(arc41))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc45.intersects(arc02))

        self.assertTrue(arc35.intersects(arc24))

        p0_ = Coordinate(180, 0)
        p1_ = Coordinate(180, 1)
        p2_ = Coordinate(-179, 0)
        p3_ = Coordinate(-180, -1)
        p4_ = Coordinate(179, 0)
        p5_ = Coordinate(-179, 1)
        p6_ = Coordinate(-179, -1)

        arc13 = Arc(p1_, p3_)
        arc24 = Arc(p2_, p4_)

        arc32 = Arc(p3_, p2_)
        arc41 = Arc(p4_, p1_)

        arc40 = Arc(p4_, p0_)
        arc56 = Arc(p5_, p6_)

        arc45 = Arc(p4_, p5_)
        arc02 = Arc(p0_, p2_)

        arc35 = Arc(p3_, p5_)

        self.assertTrue(arc13.intersects(arc24))

        self.assertFalse(arc32.intersects(arc41))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc45.intersects(arc02))

        self.assertTrue(arc35.intersects(arc24))

        # case of the north pole

        p0_ = Coordinate(0, 90)
        p1_ = Coordinate(0, 89)
        p2_ = Coordinate(90, 89)
        p3_ = Coordinate(180, 89)
        p4_ = Coordinate(-90, 89)
        p5_ = Coordinate(45, 89)
        p6_ = Coordinate(135, 89)

        arc13 = Arc(p1_, p3_)
        arc24 = Arc(p2_, p4_)

        arc32 = Arc(p3_, p2_)
        arc41 = Arc(p4_, p1_)

        arc40 = Arc(p4_, p0_)
        arc56 = Arc(p5_, p6_)

        arc45 = Arc(p4_, p5_)
        arc02 = Arc(p0_, p2_)

        arc35 = Arc(p3_, p5_)

        self.assertTrue(arc13.intersects(arc24))

        self.assertFalse(arc32.intersects(arc41))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc45.intersects(arc02))

        self.assertTrue(arc35.intersects(arc24))


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestOverlap))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSphereGeometry))

    return mysuite


if __name__ == '__main__':
    unittest.main()
