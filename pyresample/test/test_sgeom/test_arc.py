#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Pyresample developers
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
"""Define tests for the SArc class."""

import copy

import numpy as np
import pytest
from shapely.geometry import LineString

from pyresample.future.spherical import SArc, SPoint


class TestSArc:
    """Test SArc class."""

    # TODO: Fixtures defined here?
    equator_arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
    equator_arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(0.0, 0.0))

    # pole_to_pole_arc1 = SArc(SPoint.from_degrees(0.0, -90.0), SPoint.from_degrees(0.0, 90.0))
    # pole_to_pole_arc2 = SArc(SPoint.from_degrees(60.0, -90.0), SPoint.from_degrees(60.0, 90.0))

    def test_unvalid_point_arc(self):
        """Check raise error if start and end point are equal."""
        p = SPoint.from_degrees(0.0, 80.0)
        with pytest.raises(ValueError):
            SArc(p, p)

    def test_unvalid_180degree_equator_arc(self):
        """Check raise error if the points lies on the equator and are 180° apart."""
        p1 = SPoint.from_degrees(0, 0)
        p2 = SPoint.from_degrees(180, 0)
        with pytest.raises(ValueError):
            SArc(p1, p2)

        p1 = SPoint.from_degrees(-10, 0)
        p2 = SPoint.from_degrees(170, 0)
        with pytest.raises(ValueError):
            SArc(p1, p2)

        p1 = SPoint.from_degrees(10, 0)
        p2 = SPoint.from_degrees(-170, 0)
        with pytest.raises(ValueError):
            SArc(p1, p2)

    def test_unvalid_pole_to_pole_arc(self):
        """Check raise error if the points defines a pole to pole arc."""
        p1 = SPoint.from_degrees(0.0, -90.0)
        p2 = SPoint.from_degrees(0.0, 90.0)
        with pytest.raises(ValueError):
            SArc(p1, p2)

    def test_is_on_equator_arc(self):
        """Check if the arc lies on the equator."""
        equator_arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
        arc = SArc(SPoint.from_degrees(0, 10), SPoint.from_degrees(10, 0))
        assert equator_arc1.is_on_equator()
        assert not arc.is_on_equator()

    def test_eq(self):
        """Test SArc equality."""
        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        arc2 = SArc(SPoint.from_degrees(0, 10), SPoint.from_degrees(10, 0))
        assert not arc1.__eq__(arc2)
        assert arc1 == arc1

    def test_ne(self):
        """Test SArc disequality."""
        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        arc2 = SArc(SPoint.from_degrees(0, 10), SPoint.from_degrees(10, 0))
        assert arc1 != arc2
        assert not arc1.__ne__(arc1)

    def test_str(self):
        """Test SArc __str__ representation."""
        arc = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        expected_str = str(arc.start) + " -> " + str(arc.end)
        assert str(arc) == expected_str

    def test_repr(self):
        """Test SArc __repr__ representation."""
        arc = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        expected_repr = str(arc.start) + " -> " + str(arc.end)
        assert repr(arc) == expected_repr

    def test_vertices(self):
        """Test vertices property."""
        start_point_vertices = np.deg2rad((-180.0, 0.0))
        end_point_vertices = np.deg2rad((-180.0, 90.0))
        arc = SArc(SPoint(*start_point_vertices), SPoint(*end_point_vertices))
        start_vertices, end_vertices = arc.vertices
        assert np.allclose(start_vertices, start_point_vertices)
        assert np.allclose(end_vertices, end_point_vertices)

    def test_vertices_in_degrees(self):
        """Test vertices_in_degrees property."""
        start_point_vertices = np.deg2rad((-180.0, 0.0))
        end_point_vertices = np.deg2rad((-180.0, 90.0))
        arc = SArc(SPoint.from_degrees(* start_point_vertices),
                   SPoint.from_degrees(* end_point_vertices))
        start_vertices, end_vertices = arc.vertices_in_degrees
        assert np.allclose(start_vertices, start_point_vertices)
        assert np.allclose(end_vertices, end_point_vertices)

    def test_to_shapely(self):
        """Test conversion to shapely."""
        start_point_vertices = np.deg2rad((-180.0, 0.0))
        end_point_vertices = np.deg2rad((-180.0, 90.0))
        arc = SArc(SPoint.from_degrees(* start_point_vertices),
                   SPoint.from_degrees(* end_point_vertices))
        shapely_line = arc.to_shapely()
        expected_line = LineString((start_point_vertices, end_point_vertices))
        assert shapely_line.equals_exact(expected_line, tolerance=1e-10)

    def test_hash(self):
        """Test arc hash."""
        arc = SArc(SPoint.from_degrees(-180.0, -90.0), SPoint.from_degrees(-180.0, 0.0))
        assert hash(arc) == -3096892178517935054

    def test_midpoint(self):
        """Test arc midpoint."""
        start_point_vertices = np.deg2rad((-180.0, 0.0))
        end_point_vertices = np.deg2rad((-180.0, 90.0))
        arc = SArc(SPoint(*start_point_vertices), SPoint(*end_point_vertices))
        midpoint = arc.midpoint()
        assert isinstance(midpoint, SPoint)
        assert np.allclose(midpoint.vertices_in_degrees[0], (-180, 45))

    def test_intersection_point(self):
        """Test SArc(s) intersection point."""
        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        arc2 = SArc(SPoint.from_degrees(0, 10), SPoint.from_degrees(10, 0))
        lon, lat = arc1.intersection_point(arc2)

        np.testing.assert_allclose(np.rad2deg(lon), 5)
        assert np.rad2deg(lat).round(7) == round(5.0575148968282093, 7)

        # Test when same SArc
        arc1 = SArc(SPoint(0, 0), SPoint.from_degrees(10, 10))
        assert (arc1.intersection_point(arc1) is None)

        # Test intersecting SArc(s)
        arc1 = SArc(SPoint.from_degrees(24.341215776575297, 44.987819588259327),
                    SPoint.from_degrees(18.842727517611817, 46.512483610284178))
        arc2 = SArc(SPoint.from_degrees(20.165961750361905, 46.177305385810541),
                    SPoint.from_degrees(20.253297585831707, 50.935830837274324))
        p = SPoint.from_degrees(20.165957021925202, 46.177022633103398)
        assert arc1.intersection_point(arc2) == p
        assert arc1.intersects(arc2)

        # Test non-intersecting SArc(s)
        arc1 = SArc(SPoint.from_degrees(-2.4982818108326734, 48.596644847869655),
                    SPoint.from_degrees(-2.9571441235622835, 49.165688435261394))
        arc2 = SArc(SPoint.from_degrees(-3.4976667413531688, 48.562704872921373),
                    SPoint.from_degrees(-5.893976312685715, 48.445795283217116))
        assert arc1.intersection_point(arc2) is None
        assert not arc1.intersects(arc2)
        assert not bool(None)  # this occurs within the intersects method

    def test_disjoint_arcs_crossing_antimeridian(self):
        """Test SArc(s) intersection point when disjoint arcs cross the antimeridian."""
        arc1 = SArc(SPoint.from_degrees(143.76, 0),
                    SPoint.from_degrees(143.95, 7.33))
        arc2 = SArc(SPoint.from_degrees(170.34, 71.36),
                    SPoint.from_degrees(-171.03, 76.75))
        arc1_orig = copy.deepcopy(arc1)
        arc2_orig = copy.deepcopy(arc2)
        point = arc1.intersection_point(arc2)

        # Assert original arcs are unaffected
        assert arc1_orig.end.lon == arc1.end.lon
        assert arc2_orig.end.lon == arc2.end.lon

        # Assert disjoint arcs returns None
        assert isinstance(point, type(None))

    def test_intersecting_arcs_crossing_antimeridian(self):
        """Test SArc(s) intersection point when intersecting arcs cross the antimeridian."""
        arc1 = SArc(SPoint.from_degrees(-180.0, -89.0),
                    SPoint.from_degrees(-180.0, 89.0))
        arc2 = SArc(SPoint.from_degrees(-171.03, -76.75),
                    SPoint.from_degrees(170.34, -71.36))
        arc1_orig = copy.deepcopy(arc1)
        arc2_orig = copy.deepcopy(arc2)
        point = arc1.intersection_point(arc2)
        # Assert original arcs are unaffected
        assert arc1_orig.end.lon == arc1.end.lon
        assert arc2_orig.end.lon == arc2.end.lon
        # Assert intersection result
        assert point == SPoint.from_degrees(-180.0, -74.78847163)

    def test_great_circles_equator_arcs_cases(self):
        """Test behaviour when 2 arcs are lying around the equator."""
        equator_arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
        equator_arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(0.0, 0.0))

        # - If one arc is reversed, the results order is swapped
        p1, p2 = equator_arc1._great_circle_intersections(equator_arc2)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = equator_arc2._great_circle_intersections(equator_arc1)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - If both arc are reversed, results are the same
        p1, p2 = equator_arc2.reverse_direction()._great_circle_intersections(equator_arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing the direction of one arc, lead inversion of points
        p1, p2 = equator_arc2._great_circle_intersections(equator_arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

    def test_intersection_equator_arcs_case(self):
        """Test intersection point with 2 arcs lying around the equator."""
        equator_arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
        equator_arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(0.0, 0.0))
        equator_arc3 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(-20, 0.0))

        # Touching arcs
        p = equator_arc2.intersection_point(equator_arc1)
        assert np.allclose(p.vertices_in_degrees, (0, 0))

        p = equator_arc2.intersection_point(equator_arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (0, 0))

        # Overlapping arcs
        p = equator_arc2.intersection_point(equator_arc3)
        assert isinstance(p, type(None))

        # Disjoint arcs
        p = equator_arc3.intersection_point(equator_arc1)
        assert isinstance(p, type(None))

    # ---------------------------------------------------------------------.
    # TEST FOR EDGE CASE SAME ARC

    def test_intersection_same_arcs(self):
        """Test behaviour when same arc."""
        nh_arc = SArc(SPoint.from_degrees(0.0, 20.0), SPoint.from_degrees(10.0, 40.0))
        # TODO BUG ! CHECK LEGAGY BEHAVIOUR
        # - Define behaviour when same arc !!!
        p1, p2 = nh_arc._great_circle_intersections(nh_arc)  # (0,0), (-180, 0)
        p = nh_arc.intersection_point(nh_arc)  # None
        assert isinstance(p, type(None))

        p1, p2 = nh_arc._great_circle_intersections(nh_arc.reverse_direction())  # (0,0), (-180, 0)
        p = nh_arc.intersection_point(nh_arc.reverse_direction())  # None
        assert isinstance(p, type(None))

    # ---------------------------------------------------------------------.
    # TEST FOR TOUCHING CASES

    def test_touching_arc_extremes_case(self):
        """Test behaviour when two arc touches at the extremes."""
        # Test arc1.end == arc2.start
        arc1 = SArc(SPoint.from_degrees(0.0, 10.0), SPoint.from_degrees(0.0, 20.0))
        arc2 = SArc(SPoint.from_degrees(0.0, 20.0), SPoint.from_degrees(0.0, 30.0))

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))
        assert not arc1.intersects(arc2)

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        # Test arc1.start == arc2.start
        arc1 = SArc(SPoint.from_degrees(0.0, 10.0), SPoint.from_degrees(0.0, 20.0))
        arc2 = SArc(SPoint.from_degrees(0.0, 10.0), SPoint.from_degrees(0.0, -20.0))
        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))
        assert not arc1.intersects(arc2)

        # TODO: BUG Bad behaviour at the equator.
        equator_arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
        arc = SArc(SPoint.from_degrees(5, 0.0), SPoint.from_degrees(0.0, 30))

        p = equator_arc1.intersection_point(arc)
        assert np.allclose(p.vertices_in_degrees, (5.0, 0))

        p = arc.intersection_point(equator_arc1)
        assert np.allclose(p.vertices_in_degrees, (5.0, 0))

    def test_touching_arc_midpoint_case(self):
        """Test behaviour when touches in the mid of the arc."""
        arc1 = SArc(SPoint.from_degrees(0.0, 10.0), SPoint.from_degrees(0.0, 20.0))
        midpoint = arc1.midpoint()  # (0, 15)
        arc2 = SArc(midpoint, SPoint(-np.pi / 2, midpoint.lat))
        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (0, 15))
