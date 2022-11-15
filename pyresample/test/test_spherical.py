#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-2021 Pyresample Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test cases for spherical geometry."""

import unittest

import numpy as np

from pyresample.spherical import Arc, CCoordinate, SCoordinate, SphPolygon

VERTICES_TEST_IS_INSIDE1 = np.array([[-1.54009253, 82.62402855],
                                     [3.4804808, 82.8105746],
                                     [20.7214892, 83.00875812],
                                     [32.8857629, 82.7607758],
                                     [41.53844302, 82.36024339],
                                     [47.92062759, 81.91317164],
                                     [52.82785062, 81.45769791],
                                     [56.75107895, 81.00613046],
                                     [59.99843787, 80.56042986],
                                     [62.76998034, 80.11814453],
                                     [65.20076209, 79.67471372],
                                     [67.38577498, 79.22428],
                                     [69.39480149, 78.75981318],
                                     [71.28163984, 78.27283234],
                                     [73.09016378, 77.75277976],
                                     [74.85864685, 77.18594725],
                                     [76.62327682, 76.55367303],
                                     [78.42162204, 75.82918893],
                                     [80.29698409, 74.97171721],
                                     [82.30538638, 73.9143231],
                                     [84.52973107, 72.53535661],
                                     [87.11696138, 70.57600156],
                                     [87.79163209, 69.98712409],
                                     [72.98142447, 67.1760143],
                                     [61.79517279, 63.2846272],
                                     [53.50600609, 58.7098766],
                                     [47.26725347, 53.70533139],
                                     [42.44083259, 48.42199571],
                                     [38.59682041, 42.95008531],
                                     [35.45189206, 37.3452509],
                                     [32.43435578, 30.72373327],
                                     [31.73750748, 30.89485287],
                                     [29.37284023, 31.44344415],
                                     [27.66001308, 31.81016309],
                                     [26.31358296, 32.08057499],
                                     [25.1963477, 32.29313986],
                                     [24.23118049, 32.46821821],
                                     [23.36993508, 32.61780082],
                                     [22.57998837, 32.74952569],
                                     [21.8375532, 32.86857867],
                                     [21.12396693, 32.97868717],
                                     [20.42339605, 33.08268331],
                                     [19.72121983, 33.18284728],
                                     [19.00268283, 33.28113306],
                                     [18.2515215, 33.3793305],
                                     [17.4482606, 33.47919405],
                                     [16.56773514, 33.58255576],
                                     [15.57501961, 33.6914282],
                                     [14.4180087, 33.8080799],
                                     [13.01234319, 33.93498577],
                                     [11.20625437, 34.0742239],
                                     [8.67990371, 34.22415978],
                                     [7.89344478, 34.26018768],
                                     [8.69446485, 41.19823568],
                                     [9.25707165, 47.17351118],
                                     [9.66283477, 53.14128114],
                                     [9.84134875, 59.09937166],
                                     [9.65054241, 65.04458004],
                                     [8.7667375, 70.97023122],
                                     [6.28280904, 76.85731403]], dtype='float64')

VERTICES_TEST_IS_INSIDE2 = np.array([[49.94506701, 46.52610743],
                                     [51.04293649, 46.52610743],
                                     [62.02163129, 46.52610743],
                                     [73.0003261, 46.52610743],
                                     [83.9790209, 46.52610743],
                                     [85.05493299, 46.52610743],
                                     [85.05493299, 45.76549301],
                                     [85.05493299, 37.58315571],
                                     [85.05493299, 28.39260587],
                                     [85.05493299, 18.33178739],
                                     [85.05493299, 17.30750918],
                                     [83.95706351, 17.30750918],
                                     [72.97836871, 17.30750918],
                                     [61.9996739, 17.30750918],
                                     [51.0209791, 17.30750918],
                                     [49.94506701, 17.30750918],
                                     [49.94506701, 18.35262921],
                                     [49.94506701, 28.41192025],
                                     [49.94506701, 37.60055422],
                                     [49.94506701, 45.78080831]], dtype='float64')


class TestRadiansFunctions(unittest.TestCase):
    """Test utils functions processing radians arrays."""

    def test_radian_unwrapping(self):
        from pyresample.spherical import _unwrap_radians

        # Test points at np.pi are converted to -np.pi
        assert np.allclose(_unwrap_radians(-np.pi), -np.pi)
        assert np.allclose(_unwrap_radians(np.pi), -np.pi)


class TestSCoordinate(unittest.TestCase):
    """Test SCoordinates."""

    def test_distance(self):
        """Test Vincenty formula."""
        d = SCoordinate(0, 0).distance(SCoordinate(1, 1))
        np.testing.assert_almost_equal(d, 1.2745557823062943)

    def test_hdistance(self):
        """Test Haversine formula."""
        d = SCoordinate(0, 0).hdistance(SCoordinate(1, 1))
        np.testing.assert_allclose(d, 1.2745557823062943)

    def test_crosscart(self):
        """Test cross product and conversion to cartesian."""
        # Test crossproduct between poles
        p = SCoordinate(np.deg2rad(50), np.deg2rad(90.0))
        p1 = SCoordinate(np.deg2rad(50), np.deg2rad(-90.0))
        cp = p.cross2cart(p1)
        assert np.allclose(cp.cart, [0, 0, 0])

        # Test crossproduct between points along the equator
        p = SCoordinate(np.deg2rad(-180.0), np.deg2rad(0.0))
        p1 = SCoordinate(np.deg2rad(0.0), np.deg2rad(0.0))
        cp = p.cross2cart(p1)
        assert np.allclose(cp.cart, [0, 0, 0])

        # Test crossproduct between same points
        p = SCoordinate(np.deg2rad(50), np.deg2rad(20.0))
        p1 = SCoordinate(np.deg2rad(50), np.deg2rad(20.0))
        cp = p.cross2cart(p1)
        assert np.allclose(cp.cart, [0, 0, 0])

        p = SCoordinate(np.deg2rad(-180), np.deg2rad(90.0))
        p1 = SCoordinate(np.deg2rad(180), np.deg2rad(90.0))
        p.cross2cart(p1)
        assert np.allclose(cp.cart, [0, 0, 0])

    def test_str(self):
        """Check the string representation."""
        d = SCoordinate(0, 0)
        self.assertEqual(str(d), "(0.0, 0.0)")

    def test_repr(self):
        """Check the representation."""
        d = SCoordinate(0, 0)
        self.assertEqual(repr(d), "(0.0, 0.0)")

    def test_equality_at_antimeridian(self):
        """Test equality of points at the antimeridian."""
        point_start = SCoordinate(-np.pi, 0)
        point_end = SCoordinate(np.pi, 0)
        assert point_start == point_end


class TestCCoordinate(unittest.TestCase):
    """Test SCoordinates."""

    def test_str(self):
        """Check the string representation."""
        d = CCoordinate((0, 0, 0))
        self.assertEqual(str(d), "[0 0 0]")

    def test_repr(self):
        """Check the representation."""
        d = CCoordinate((0, 0, 0))
        self.assertEqual(repr(d), "[0 0 0]")

    def test_norm(self):
        """Euclidean norm of a cartesian vector."""
        d = CCoordinate((1, 0, 0))
        self.assertEqual(d.norm(), 1.0)

    def test_normalize(self):
        """Normalize a cartesian vector."""
        d = CCoordinate((2., 0., 0.))
        np.testing.assert_allclose(d.normalize().cart, [1, 0, 0])

    def test_cross(self):
        """Test cross product in cartesian coordinates."""
        d = CCoordinate((1., 0., 0.))
        c = CCoordinate((0., 1., 0.))
        np.testing.assert_allclose(d.cross(c).cart, [0., 0., 1.])

    def test_dot(self):
        """Test the dot product of two cartesian vectors."""
        d = CCoordinate((1., 0., 0.))
        c = CCoordinate((0., 1., 0.))
        self.assertEqual(d.dot(c), 0)

    def test_ne(self):
        """Test inequality of two cartesian vectors."""
        d = CCoordinate((1., 0., 0.))
        c = CCoordinate((0., 1., 0.))
        assert c != d

    def test_eq(self):
        """Test equality of two cartesian vectors."""
        d = CCoordinate((1., 0., 0.))
        c = CCoordinate((0., 1., 0.))
        assert not c.__eq__(d)

    def test_add(self):
        """Test adding cartesian vectors."""
        d = CCoordinate((1., 0., 0.))
        c = CCoordinate((0., 1., 0.))
        b = CCoordinate((1., 1., 0.))
        np.testing.assert_allclose((d + c).cart, b.cart)
        np.testing.assert_allclose((d + (0, 1, 0)).cart, b.cart)
        np.testing.assert_allclose(((0, 1, 0) + d).cart, b.cart)

    def test_mul(self):
        """Test multiplying (element-wise) cartesian vectors."""
        d = CCoordinate((1., 0., 0.))
        c = CCoordinate((0., 1., 0.))
        b = CCoordinate((0., 0., 0.))
        np.testing.assert_allclose((d * c).cart, b.cart)
        np.testing.assert_allclose((d * (0, 1, 0)).cart, b.cart)
        np.testing.assert_allclose(((0, 1, 0) * d).cart, b.cart)

    def test_to_spherical(self):
        """Test converting to spherical coordinates."""
        d = CCoordinate((1., 0., 0.))
        c = SCoordinate(0, 0)
        self.assertEqual(d.to_spherical(), c)


class TestArc(unittest.TestCase):
    """Test arcs."""

    def test_eq(self):
        arc1 = Arc(SCoordinate(0, 0),
                   SCoordinate(np.deg2rad(10), np.deg2rad(10)))
        arc2 = Arc(SCoordinate(0, np.deg2rad(10)),
                   SCoordinate(np.deg2rad(10), 0))

        assert not arc1.__eq__(arc2)
        assert arc1 == arc1

    def test_ne(self):
        arc1 = Arc(SCoordinate(0, 0),
                   SCoordinate(np.deg2rad(10), np.deg2rad(10)))
        arc2 = Arc(SCoordinate(0, np.deg2rad(10)),
                   SCoordinate(np.deg2rad(10), 0))

        assert arc1 != arc2
        assert not arc1.__ne__(arc1)

    def test_str(self):
        arc1 = Arc(SCoordinate(0, 0),
                   SCoordinate(np.deg2rad(10), np.deg2rad(10)))
        self.assertEqual(str(arc1), str(arc1.start) + " -> " + str(arc1.end))
        self.assertEqual(repr(arc1), str(arc1.start) + " -> " + str(arc1.end))

    def test_intersection(self):
        arc1 = Arc(SCoordinate(0, 0),
                   SCoordinate(np.deg2rad(10), np.deg2rad(10)))
        arc2 = Arc(SCoordinate(0, np.deg2rad(10)),
                   SCoordinate(np.deg2rad(10), 0))
        lon, lat = arc1.intersection(arc2)

        np.testing.assert_allclose(np.rad2deg(lon), 5)
        self.assertEqual(np.rad2deg(lat).round(7), round(5.0575148968282093, 7))

        arc1 = Arc(SCoordinate(0, 0),
                   SCoordinate(np.deg2rad(10), np.deg2rad(10)))

        assert (arc1.intersection(arc1) is None)

        arc1 = Arc(SCoordinate(np.deg2rad(24.341215776575297),
                               np.deg2rad(44.987819588259327)),
                   SCoordinate(np.deg2rad(18.842727517611817),
                               np.deg2rad(46.512483610284178)))
        arc2 = Arc(SCoordinate(np.deg2rad(20.165961750361905),
                               np.deg2rad(46.177305385810541)),
                   SCoordinate(np.deg2rad(20.253297585831707),
                               np.deg2rad(50.935830837274324)))
        inter = SCoordinate(np.deg2rad(20.165957021925202),
                            np.deg2rad(46.177022633103398))
        self.assertEqual(arc1.intersection(arc2), inter)

        arc1 = Arc(SCoordinate(np.deg2rad(-2.4982818108326734),
                               np.deg2rad(48.596644847869655)),
                   SCoordinate(np.deg2rad(-2.9571441235622835),
                               np.deg2rad(49.165688435261394)))
        arc2 = Arc(SCoordinate(np.deg2rad(-3.4976667413531688),
                               np.deg2rad(48.562704872921373)),
                   SCoordinate(np.deg2rad(-5.893976312685715),
                               np.deg2rad(48.445795283217116)))

        assert (arc1.intersection(arc2) is None)

    def test_angle(self):
        arc1 = Arc(SCoordinate(np.deg2rad(157.5),
                               np.deg2rad(89.234600944314138)),
                   SCoordinate(np.deg2rad(90),
                               np.deg2rad(89)))
        arc2 = Arc(SCoordinate(np.deg2rad(157.5),
                               np.deg2rad(89.234600944314138)),
                   SCoordinate(np.deg2rad(135),
                               np.deg2rad(89)))

        self.assertAlmostEqual(np.rad2deg(arc1.angle(arc2)), -44.996385007218926)

        arc1 = Arc(SCoordinate(np.deg2rad(112.5),
                               np.deg2rad(89.234600944314138)),
                   SCoordinate(np.deg2rad(90), np.deg2rad(89)))
        arc2 = Arc(SCoordinate(np.deg2rad(112.5),
                               np.deg2rad(89.234600944314138)),
                   SCoordinate(np.deg2rad(45), np.deg2rad(89)))

        self.assertAlmostEqual(np.rad2deg(arc1.angle(arc2)), 44.996385007218883)

        arc1 = Arc(SCoordinate(0, 0), SCoordinate(1, 0))
        self.assertAlmostEqual(arc1.angle(arc1), 0)

        arc2 = Arc(SCoordinate(1, 0), SCoordinate(0, 0))
        self.assertAlmostEqual(arc1.angle(arc2), 0)

        arc2 = Arc(SCoordinate(0, 0), SCoordinate(-1, 0))
        self.assertAlmostEqual(arc1.angle(arc2), np.pi)

        arc2 = Arc(SCoordinate(2, 0), SCoordinate(1, 0))
        self.assertAlmostEqual(arc1.angle(arc2), np.pi)

        arc2 = Arc(SCoordinate(2, 0), SCoordinate(3, 0))
        self.assertRaises(ValueError, arc1.angle, arc2)

    def test_disjoint_arcs_crossing_antimeridian(self):
        import copy
        arc1 = Arc(SCoordinate(*np.deg2rad((143.76, 0))),
                   SCoordinate(*np.deg2rad((143.95, 7.33)))
                   )
        arc2 = Arc(SCoordinate(*np.deg2rad((170.34, 71.36))),
                   SCoordinate(*np.deg2rad((-171.03, 76.75)))
                   )
        arc1_orig = copy.deepcopy(arc1)
        arc2_orig = copy.deepcopy(arc2)
        point = arc1.intersection(arc2)
        # Assert original arcs are unaffected
        assert arc1_orig.end.lon == arc1.end.lon
        assert arc2_orig.end.lon == arc2.end.lon
        # Assert disjoint arcs returns None
        assert isinstance(point, type(None))

    def test_intersecting_arcs_crossing_antimeridian(self):
        import copy
        arc1 = Arc(SCoordinate(*np.deg2rad((-180.0, -90.0))),
                   SCoordinate(*np.deg2rad((-180.0, 90.0)))
                   )
        arc2 = Arc(SCoordinate(*np.deg2rad((-171.03, -76.75))),
                   SCoordinate(*np.deg2rad((170.34, -71.36)))
                   )
        arc1_orig = copy.deepcopy(arc1)
        arc2_orig = copy.deepcopy(arc2)
        point = arc1.intersection(arc2)
        # Assert original arcs are unaffected
        assert arc1_orig.end.lon == arc1.end.lon
        assert arc2_orig.end.lon == arc2.end.lon
        # Assert intersection result
        assert point == SCoordinate(*np.deg2rad((-180.0, -74.7884716)))


class TestSphericalPolygon(unittest.TestCase):
    """Test the spherical polygon."""

    def test_area(self):
        """Test the area function."""
        vertices = np.array([[1, 2, 3, 4, 3, 2],
                             [3, 4, 3, 2, 1, 2]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.00121732523118, polygon.area())

        vertices = np.array([[1, 2, 3, 2],
                             [3, 4, 3, 2]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.000608430665842, polygon.area())

        vertices = np.array([[0, 0, 1, 1],
                             [0, 1, 1, 0]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.000304609684862, polygon.area())

        # Across the dateline

        vertices = np.array([[179.5, -179.5, -179.5, 179.5],
                             [1, 1, 0, 0]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.000304609684862, polygon.area())

        vertices = np.array([[0, 90, 90, 0],
                             [1, 1, 0, 0]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.0349012696772, polygon.area())

        vertices = np.array([[90, 0, 0],
                             [0, 0, 90]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(np.pi / 2, polygon.area())

        # Around the north pole

        vertices = np.array([[0, -90, 180, 90],
                             [89, 89, 89, 89]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.000609265770322, polygon.area())

        # Around the south pole

        vertices = np.array([[0, 90, 180, -90],
                             [-89, -89, -89, -89]]).T
        polygon = SphPolygon(np.deg2rad(vertices))

        self.assertAlmostEqual(0.000609265770322, polygon.area())

    def test_is_inside(self):
        """Test checking if a polygon is inside of another."""
        vertices = np.array([[1, 1, 20, 20],
                             [1, 20, 20, 1]]).T

        rad_verts = np.deg2rad(vertices)
        polygon1 = SphPolygon(rad_verts)
        # make sure 64-bit floats don't get copied
        assert polygon1.vertices is rad_verts

        vertices = np.array([[0, 0, 30, 30],
                             [0, 30, 30, 0]]).T

        polygon2 = SphPolygon(np.deg2rad(vertices))

        assert polygon1._is_inside(polygon2)

        assert not polygon2._is_inside(polygon1)

        # Why checking the areas here!? It has nothing to do with the is_inside function!
        assert polygon2.area() > polygon1.area()

        polygon2.invert()
        assert not polygon1._is_inside(polygon2)
        assert not polygon2._is_inside(polygon1)

        vertices = np.array([[0, 0, 30, 30],
                             [21, 30, 30, 21]]).T

        polygon2 = SphPolygon(np.deg2rad(vertices))
        assert not polygon1._is_inside(polygon2)
        assert not polygon2._is_inside(polygon1)

        polygon2.invert()

        assert polygon1._is_inside(polygon2)
        assert not polygon2._is_inside(polygon1)

        vertices = np.array([[100, 100, 130, 130],
                             [41, 50, 50, 41]]).T

        polygon2 = SphPolygon(np.deg2rad(vertices))

        assert not polygon1._is_inside(polygon2)
        assert not polygon2._is_inside(polygon1)

        polygon2.invert()

        assert polygon1._is_inside(polygon2)
        assert not polygon2._is_inside(polygon1)

        vertices = VERTICES_TEST_IS_INSIDE1

        polygon1 = SphPolygon(np.deg2rad(vertices))

        vertices = VERTICES_TEST_IS_INSIDE2

        polygon2 = SphPolygon(np.deg2rad(vertices))

        assert not polygon2._is_inside(polygon1)
        assert not polygon1._is_inside(polygon2)

    def test_is_inside_float32(self):
        """Test that precision dependent calculations work.

        Some of the precision math can fail with only 32-bit floats.

        """
        b_verts = np.array([[-1.3440281, 0.12873407],
                            [-1.3714675, 0.17091802],
                            [-1.5773474, 0.14104952],
                            [-1.5913332, 0.09375837],
                            [-1.598127, 0.12360403],
                            [-1.7618076, 0.65335757],
                            [-1.7746785, 0.6819248],
                            [-1.7645605, 0.73133504],
                            [-1.4870182, 0.77076954],
                            [-1.4529741, 0.7284483],
                            [-1.4474869, 0.69782615],
                            [-1.3499607, 0.15860939]], dtype=np.float32)
        t_verts = np.array([[-1.80760407, 0.82227004],
                            [-1.65620456, 0.82549202],
                            [-1.34904288, 0.81162246],
                            [-1.2014294, 0.79483332],
                            [-1.21090379, 0.75978692],
                            [-1.26056844, 0.54041217],
                            [-1.30037964, 0.31453188],
                            [-1.33298796, 0.09364747],
                            [-1.33784689, 0.05810091],
                            [-1.44211119, 0.07002157],
                            [-1.656768, 0.07992753],
                            [-1.76233241, 0.07762154],
                            [-1.7639324, 0.11370358],
                            [-1.77469158, 0.33765409],
                            [-1.78788255, 0.5660454],
                            [-1.8044336, 0.78705058]], dtype=np.float32)
        b_poly = SphPolygon(b_verts)
        t_poly = SphPolygon(t_verts)
        assert b_poly._is_inside(t_poly)

    def test_union_polygons_overlap_partially(self):
        """Test the union method."""
        vertices = np.array([[180, 90, 0, -90],
                             [89, 89, 89, 89]]).T
        poly1 = SphPolygon(np.deg2rad(vertices))
        vertices = np.array([[-45, -135, 135, 45],
                             [89, 89, 89, 89]]).T
        poly2 = SphPolygon(np.deg2rad(vertices))

        uni = np.array([[157.5, 89.23460094],
                        [135., 89.],
                        [112.5, 89.23460094],
                        [90., 89.],
                        [67.5, 89.23460094],
                        [45., 89.],
                        [22.5, 89.23460094],
                        [0., 89.],
                        [-22.5, 89.23460094],
                        [-45., 89.],
                        [-67.5, 89.23460094],
                        [-90., 89.],
                        [-112.5, 89.23460094],
                        [-135., 89.],
                        [-157.5, 89.23460094],
                        [-180., 89.]])

        poly_union = poly1.union(poly2)
        assert np.allclose(poly_union.vertices, np.deg2rad(uni))

    def test_union_polygons_overlaps_completely(self):
        """Test the union method when one polygon is entirely inside the other."""
        vertices = np.array([[1, 1, 20, 20],
                             [1, 20, 20, 1]]).T

        poly1 = SphPolygon(np.deg2rad(vertices))

        vertices = np.array([[0, 0, 30, 30],
                             [0, 30, 30, 0]]).T
        poly2 = SphPolygon(np.deg2rad(vertices))

        poly_union1 = poly1.union(poly2)
        poly_union2 = poly2.union(poly1)

        expected = np.deg2rad(np.array([[0, 0, 30, 30],
                                        [0, 30, 30, 0]]).T)

        np.testing.assert_allclose(poly_union1.vertices, expected)
        np.testing.assert_allclose(poly_union2.vertices, expected)

    def test_intersection(self):
        """Test the intersection function."""
        vertices = np.array([[180, 90, 0, -90],
                             [89, 89, 89, 89]]).T
        poly1 = SphPolygon(np.deg2rad(vertices))
        vertices = np.array([[-45, -135, 135, 45],
                             [89, 89, 89, 89]]).T
        poly2 = SphPolygon(np.deg2rad(vertices))

        inter = np.array([[157.5, 89.23460094],
                          [112.5, 89.23460094],
                          [67.5, 89.23460094],
                          [22.5, 89.23460094],
                          [-22.5, 89.23460094],
                          [-67.5, 89.23460094],
                          [-112.5, 89.23460094],
                          [-157.5, 89.23460094]])
        poly_inter = poly1.intersection(poly2)

        np.testing.assert_allclose(poly_inter.vertices, np.deg2rad(inter))

        # Test 2 polygons sharing 2 contiguous edges.

        vertices1 = np.array([[-10, 10],
                              [-5, 10],
                              [0, 10],
                              [5, 10],
                              [10, 10],
                              [10, -10],
                              [-10, -10]])

        vertices2 = np.array([[-5, 10],
                              [0, 10],
                              [5, 10],
                              [5, -5],
                              [-5, -5]])

        vertices3 = np.array([[5, 10],
                              [5, -5],
                              [-5, -5],
                              [-5, 10],
                              [0, 10]])

        poly1 = SphPolygon(np.deg2rad(vertices1))
        poly2 = SphPolygon(np.deg2rad(vertices2))
        poly_inter = poly1.intersection(poly2)

        np.testing.assert_allclose(poly_inter.vertices, np.deg2rad(vertices3))

        # Test when last node of the intersection is the last vertice of the
        # second polygon.

        swath_vertices = np.array([[-115.32268301, 66.32946139],
                                   [-61.48397172, 58.56799254],
                                   [-60.25004314, 58.00754686],
                                   [-71.35057076, 49.60229517],
                                   [-113.746486, 56.03008985]])
        area_vertices = np.array([[-68.32812107, 52.3480829],
                                  [-67.84993896, 53.07015692],
                                  [-55.54651296, 64.9254637],
                                  [-24.63341856, 74.24628796],
                                  [-31.8996363, 27.99907764],
                                  [-39.581043, 37.0639821],
                                  [-50.90185988, 45.56296169],
                                  [-67.43022017, 52.12399581]])

        res = np.array([[-62.77837918, 59.12607053],
                        [-61.48397172, 58.56799254],
                        [-60.25004314, 58.00754686],
                        [-71.35057076, 49.60229517],
                        [-113.746486, 56.03008985],
                        [-115.32268301, 66.32946139]])

        poly1 = SphPolygon(np.deg2rad(swath_vertices))
        poly2 = SphPolygon(np.deg2rad(area_vertices))

        poly_inter = poly1.intersection(poly2)
        np.testing.assert_allclose(poly_inter.vertices, np.deg2rad(res))

        poly_inter = poly2.intersection(poly1)
        np.testing.assert_allclose(poly_inter.vertices, np.deg2rad(res))

        # Test when intersection occurs across the antimeridian
        vertices_epsg_4326 = np.deg2rad(np.array([[-180, -90],
                                                  [-180, 90],
                                                  [0, 90],
                                                  [180, 90],
                                                  [180, -90]]))
        vertices_goes_west = np.array([[2.50919361e+00, 1.19782309e-16],
                                       [2.51252770e+00, 1.27991660e-01],
                                       [2.97300443e+00, 1.24550237e+00],
                                       [-2.98515478e+00, 1.33966326e+00],
                                       [-1.00821045e+00, -0.00000000e+00],
                                       [-2.98515478e+00, -1.33966326e+00],
                                       [2.97300443e+00, -1.24550237e+00],
                                       [2.51252770e+00, -1.27991660e-01]])

        goes_west_poly = SphPolygon(vertices_goes_west)
        epsg_4326_poly = SphPolygon(vertices_epsg_4326)

        intersection_poly = epsg_4326_poly.intersection(goes_west_poly)

        # Assert found the correct intersection point
        assert np.allclose(intersection_poly.vertices[2, :],
                           np.array([-2.98515478, 1.33966326]))

        # Check bounded between -180 and 180
        assert np.all(intersection_poly.vertices >= -np.pi)
        assert np.all(intersection_poly.vertices <= np.pi)

    def test_consistent_radius(self):
        poly1 = np.array([(-50, 69), (-36, 69), (-36, 64), (-50, 64)])
        poly2 = np.array([(-46, 68), (-40, 68), (-40, 65), (-45, 65)])
        poly_outer = SphPolygon(np.deg2rad(poly1), radius=6371)
        poly_inner = SphPolygon(np.deg2rad(poly2), radius=6371)
        poly_inter = poly_outer.intersection(poly_inner)
        self.assertAlmostEqual(poly_inter.radius, poly_inner.radius)
        # Well, now when we are at it.
        self.assertAlmostEqual(poly_inter.area(), poly_inner.area())
