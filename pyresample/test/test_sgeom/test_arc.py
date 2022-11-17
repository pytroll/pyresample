#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Pyresample Developers
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

from pyresample.spherical import Arc, SCoordinate


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
