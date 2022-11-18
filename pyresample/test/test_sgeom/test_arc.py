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

from pyresample.future.spherical.arc import SArc
from pyresample.future.spherical.point import SMultiPoint, SPoint


class TestSArc:
    """Test SArc class."""

    # TODO in future
    # - Fixtures defined here or outside the class
    # - function that test intersection_point in all direction

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

    def test_antipode_points_arcs(self):
        """Check raise error if the points are the antipodes."""
        p1 = SPoint.from_degrees(45, 45.0)
        p2 = SPoint.from_degrees(-135.0, -45)
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

    # ----------------------------------------------------------------------.
    # Test SArc manipulations

    def test_midpoint(self):
        """Test arc midpoint."""
        start_point_vertices = np.deg2rad((-180.0, 0.0))
        end_point_vertices = np.deg2rad((-180.0, 90.0))
        arc = SArc(SPoint(*start_point_vertices), SPoint(*end_point_vertices))
        midpoint = arc.midpoint()
        assert isinstance(midpoint, SPoint)
        assert np.allclose(midpoint.vertices_in_degrees[0], (-180, 45))

    def test_arc_forward_points(self):
        """Test SArc forward points."""
        arc = SArc(SPoint.from_degrees(10, 10), SPoint.from_degrees(20, 20))

        # forward outside arc SPoint
        p = arc.forward_points(distance=100)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (20.00065043, 20.00065971))

        # forward inside arc SPoint
        p = arc.forward_points(distance=-100)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (19.99934958, 19.99934029))

        # Forward outside arc SMultiPoint
        p = arc.forward_points(distance=[100 * 100, 1000 * 1000])
        expected_coords = np.array([[20.06506968, 20.06595905],
                                    [26.81520629, 26.45975315]])
        assert isinstance(p, SMultiPoint)
        assert np.allclose(p.vertices_in_degrees, expected_coords)

        # Forward inside arc SPoint
        p = arc.forward_points(distance=[-100 * 100, -1000 * 1000])
        expected_coords = np.array([[19.93498483, 19.93401721],
                                    [13.73281401, 13.30073419]])
        assert isinstance(p, SMultiPoint)
        assert np.allclose(p.vertices_in_degrees, expected_coords)

    def test_arc_backward_points(self):
        """Test SArc backward points."""
        arc = SArc(SPoint.from_degrees(10, 10), SPoint.from_degrees(20, 20))

        # Backward outside arc SPoint
        p = arc.backward_points(distance=100)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (9.99934958, 9.99936874))

        # Backward inside arc SPoint
        p = arc.backward_points(distance=-100)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (10.00065043, 10.00063126))

    def test_arc_forward_points_at_equator(self):
        """Test SArc forward points at equator."""
        arc = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(0.0, 0.0))

        # Forward outside arc SPoint
        p = arc.forward_points(distance=100)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (0.00089932, 0.0))

        # Forward inside arc SPoint
        p = arc.forward_points(distance=-100)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (-0.00089932, 0.0))

    def test_arc_forward_points_at_pole(self):
        """Test SArc forward points at pole."""
        # North pole
        arc = SArc(SPoint.from_degrees(0.0, 50.0), SPoint.from_degrees(0.0, 90))

        # Forward outside arc SPoint
        p = arc.forward_points(distance=1000 * 1000)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (-180, 81.00677971))

        # Forward inside arc SPoint
        p = arc.forward_points(distance=-1000 * 1000)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (0, 81.00677971))

        # South pole
        arc = SArc(SPoint.from_degrees(0.0, 50.0), SPoint.from_degrees(0.0, -90))

        # Forward outside arc SPoint
        p = arc.forward_points(distance=1000 * 1000)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (-180, -81.00677971))

        # Forward inside arc SPoint
        p = arc.forward_points(distance=-1000 * 1000)
        assert isinstance(p, SPoint)
        assert np.allclose(p.vertices_in_degrees, (0, -81.00677971))

    def test_arc_extend(self):
        """Test SArc extend method."""
        arc = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(20.0, 0.0))

        # Test unvalid direction
        with pytest.raises(ValueError):
            arc.extend(distance=0, direction="bot")

        # Test backward
        extended_arc = arc.extend(distance=1000 * 1000, direction="backward")
        assert extended_arc.start == SPoint.from_degrees(-8.99322029, 0)
        assert extended_arc.end == arc.end

        # Test forward
        extended_arc = arc.extend(distance=1000 * 1000, direction="forward")
        assert extended_arc.start == arc.start
        assert extended_arc.end == SPoint.from_degrees(28.99322029, 0)

        # Test both
        extended_arc = arc.extend(distance=1000 * 1000, direction="both")
        assert extended_arc.start == SPoint.from_degrees(-8.99322029, 0)
        assert extended_arc.end == SPoint.from_degrees(28.99322029, 0)

    def test_arc_shorten(self):
        """Test SArc shorten method."""
        arc = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(20.0, 0.0))

        # Test unvalid direction
        with pytest.raises(ValueError):
            arc.shorten(distance=0, direction="bot")

        # Test backward
        shortened_arc = arc.shorten(distance=1000 * 1000, direction="backward")
        assert shortened_arc.start == SPoint.from_degrees(8.99322029, 0)
        assert shortened_arc.end == arc.end

        # Test forward
        shortened_arc = arc.shorten(distance=1000 * 1000, direction="forward")
        assert shortened_arc.start == arc.start
        assert shortened_arc.end == SPoint.from_degrees(11.00677971, 0)

        # Test both
        shortened_arc = arc.shorten(distance=1000 * 1000, direction="both")
        assert shortened_arc.start == SPoint.from_degrees(8.99322029, 0)
        assert shortened_arc.end == SPoint.from_degrees(11.00677971, 0)
    # ----------------------------------------------------------------------.
    # Test great circle intersections method

    def test_great_circles_intersections_of_crossing_arcs(self):
        """Test behaviour when the 2 crossing arcs are not on the same great-circle."""
        # If the two arcs cross along the arc
        # - one intersection point is the crossing point
        # - the other intersection point is at the antipodes

        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        arc2 = SArc(SPoint.from_degrees(0, 10), SPoint.from_degrees(10, 0))
        crossing_point = (5., 5.0575149)

        # - Changing the arc used to call the method changes the results order
        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, (-175., -5.0575149))
        assert np.allclose(p2.vertices_in_degrees, crossing_point)

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, crossing_point)
        assert np.allclose(p2.vertices_in_degrees, (-175., -5.0575149))

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-175., -5.0575149))
        assert np.allclose(p2.vertices_in_degrees, crossing_point)

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, crossing_point)
        assert np.allclose(p2.vertices_in_degrees, (-175., -5.0575149))

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, crossing_point)
        assert np.allclose(p2.vertices_in_degrees, (-175., -5.0575149))

        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-175., -5.0575149))
        assert np.allclose(p2.vertices_in_degrees, crossing_point)

    def test_great_circles_intersections_on_touching_arcs(self):
        """Test behaviour when the 2 arcs not on the same great-circle but touch at the extremes."""
        # If the two arcs are touching at the arc extremes
        # - one intersection point is the touching point
        # - the other intersection point is at the antipodes

        touching_point = (10.0, 20.0)
        arc1 = SArc(SPoint.from_degrees(10.0, 10.0), SPoint.from_degrees(*touching_point))
        arc2 = SArc(SPoint.from_degrees(50.0, 60.0), SPoint.from_degrees(*touching_point))

        # - Changing the arc used to call the method changes the results order
        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, touching_point)
        assert np.allclose(p2.vertices_in_degrees, (-170, -20))

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, (-170, -20))
        assert np.allclose(p2.vertices_in_degrees, touching_point)

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, touching_point)
        assert np.allclose(p2.vertices_in_degrees, (-170, -20))

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-170, -20))
        assert np.allclose(p2.vertices_in_degrees, touching_point)

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-170, -20))
        assert np.allclose(p2.vertices_in_degrees, touching_point)

        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, touching_point)
        assert np.allclose(p2.vertices_in_degrees, (-170, -20))

    def test_great_circles_intersections_of_disjoint_arcs(self):
        """Test behaviour when the 2 disjoint arcs are not on the same great-circle."""
        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))  # north hemisphere
        arc2 = SArc(SPoint.from_degrees(-10, -10), SPoint.from_degrees(-10, -20))  # south hemisphere

        # - Changing the arc used to call the method changes the results order
        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, (170., 10.))
        assert np.allclose(p2.vertices_in_degrees, (-10., -10.))

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, (-10., -10.))
        assert np.allclose(p2.vertices_in_degrees, (170., 10.))

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (170., 10.))
        assert np.allclose(p2.vertices_in_degrees, (-10., -10.))

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-10., -10.))
        assert np.allclose(p2.vertices_in_degrees, (170., 10.))

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-10., -10.))
        assert np.allclose(p2.vertices_in_degrees, (170., 10.))

        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (170., 10.))
        assert np.allclose(p2.vertices_in_degrees, (-10., -10.))

    # - with arcs on the  same great circle cases
    def test_great_circles_intersections_of_disjoint_arcs_on_same_great_circle(self):
        """Test behaviour when the 2 disjoint arcs (same great-circle case)."""
        arc1 = SArc(SPoint.from_degrees(5, 1.0), SPoint.from_degrees(5.0, 20.0))
        start_point = arc1.forward_points(distance=1000 * 1000)
        end_point = arc1.forward_points(distance=3000 * 1000)
        arc2 = SArc(start_point, end_point)

        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

    def test_great_circles_intersections_of_touching_equator_arcs(self):
        """Test behaviour of touching equatorial arcs (same great-circle case)."""
        arc1 = SArc(SPoint.from_degrees(1.0, 0.0), SPoint.from_degrees(50, 0.0))
        arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(1.0, 0.0))

        # - Changing the arc used to call the method does not change the results order
        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

    def test_great_circles_intersections_on_overlapping_arcs(self):
        """Test behaviour with 2 overlapping arcs (same great-circle case)."""
        arc1 = SArc(SPoint.from_degrees(5, 1.0), SPoint.from_degrees(5.0, 20.0))
        arc2 = arc1.extend(distance=1000 * 1000, direction="forward")

        # - Changing the arc used to call the method does not change the results order
        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

    def test_great_circles_intersections_on_contain_within_arcs(self):
        """Test behaviour with arcs contain/within each other (same great-circle case)."""
        arc1 = SArc(SPoint.from_degrees(5, 1.0), SPoint.from_degrees(5.0, 20.0))
        arc2 = arc1.extend(distance=1000 * 1000, direction="both")

        # - Changing the arc used to call the method does not change the results order
        p1, p2 = arc1._great_circle_intersections(arc2)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2._great_circle_intersections(arc1)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - If both arc are reversed, the results are the same
        p1, p2 = arc1.reverse_direction()._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc2.reverse_direction()._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing the direction of one arc changes the order of the points
        p1, p2 = arc2._great_circle_intersections(arc1.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

        p1, p2 = arc1._great_circle_intersections(arc2.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (-180, 0))
        assert np.allclose(p2.vertices_in_degrees, (0, 0))

    def test_great_circles_intersections_with_equal_arc(self):
        """Test great_circles intersection points with equal arcs."""
        arc = SArc(SPoint.from_degrees(1.0, 20.0), SPoint.from_degrees(10.0, 40.0))

        p1, p2 = arc._great_circle_intersections(arc)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing both arcs does not change  the results
        p1, p2 = arc.reverse_direction()._great_circle_intersections(arc.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        # - Reversing the direction of one arc does not change  the results
        p1, p2 = arc._great_circle_intersections(arc.reverse_direction())
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

        p1, p2 = arc.reverse_direction()._great_circle_intersections(arc)
        assert np.allclose(p1.vertices_in_degrees, (0, 0))
        assert np.allclose(p2.vertices_in_degrees, (-180, 0))

    # ----------------------------------------------------------------------.
    # Test intersection_point method
    def test_disjoint_arcs(self):
        """Test disjoint arcs behaviour."""
        arc1 = SArc(SPoint.from_degrees(-2.4982818108326734, 48.596644847869655),
                    SPoint.from_degrees(-2.9571441235622835, 49.165688435261394))
        arc2 = SArc(SPoint.from_degrees(-3.4976667413531688, 48.562704872921373),
                    SPoint.from_degrees(-5.893976312685715, 48.445795283217116))
        assert arc1.intersection_point(arc2) is None
        assert not arc1.intersects(arc2)
        assert not bool(None)  # this occurs within the intersects method

    def test_disjoint_equator_arcs(self):
        """Test behaviour with disjoint arc along the equator."""
        equator_arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
        equator_arc3 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(-20, 0.0))
        p = equator_arc3.intersection_point(equator_arc1)
        assert isinstance(p, type(None))

    def test_disjoint_arcs_crossing_the_antimeridian(self):
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

    def test_crossing_arcs(self):
        """Test crossing arcs behaviour."""
        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(10, 10))
        arc2 = SArc(SPoint.from_degrees(0, 10), SPoint.from_degrees(10, 0))
        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (5, 5.0575148968282093))

        arc1 = SArc(SPoint.from_degrees(24.341215776575297, 44.987819588259327),
                    SPoint.from_degrees(18.842727517611817, 46.512483610284178))
        arc2 = SArc(SPoint.from_degrees(20.165961750361905, 46.177305385810541),
                    SPoint.from_degrees(20.253297585831707, 50.935830837274324))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (20.165957021925202, 46.177022633103398))
        assert arc1.intersects(arc2)

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (20.165957021925202, 46.177022633103398))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (20.165957021925202, 46.177022633103398))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (20.165957021925202, 46.177022633103398))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (20.165957021925202, 46.177022633103398))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (20.165957021925202, 46.177022633103398))

    def test_crossing_arcs_at_equator(self):
        """Test arcs crossing at the equator."""
        arc1 = SArc(SPoint.from_degrees(0, 0), SPoint.from_degrees(50, 0))
        arc2 = SArc(SPoint.from_degrees(0, -10), SPoint.from_degrees(10, 10))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (5, 0))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (5, 0))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (5, 0))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (5, 0))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (5, 0))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (5, 0))

    def test_crossing_arcs_intersecting_at_the_antimeridian(self):
        """Test SArc(s) intersection point when intersecting at the antimeridian."""
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

    def test_touching_arcs(self):
        """Test behaviour when touches not at the arc extremes."""
        # Touching point at (15, 0)
        arc1 = SArc(SPoint.from_degrees(0.0, 10.0), SPoint.from_degrees(0.0, 20.0))
        midpoint = arc1.midpoint()  # (0, 15)
        arc2 = SArc(midpoint, SPoint(-np.pi / 2, midpoint.lat))
        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (0, 15))

    def test_touching_arcs_at_the_equator(self):
        """Test arcs touching at the equator in the mid of one arc."""
        # Touching point at (25, 0)
        arc1 = SArc(SPoint.from_degrees(0.0, 0.0), SPoint.from_degrees(50, 0.0))
        arc2 = SArc(SPoint.from_degrees(25, 0.0), SPoint.from_degrees(0.0, 30))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (25.0, 0))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (25.0, 0))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (25.0, 0))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (25.0, 0))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (25.0, 0))

    def test_touching_arcs_at_extremes(self):
        """Test arcs touching at the arc extremes (not on the same great-circle)."""
        # Touching point at (10.0, 20.0)
        arc1 = SArc(SPoint.from_degrees(10.0, 10.0), SPoint.from_degrees(10.0, 20.0))
        arc2 = SArc(SPoint.from_degrees(50.0, 60.0), SPoint.from_degrees(10.0, 20.0))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (10.0, 20.0))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (10.0, 20.0))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (10.0, 20.0))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (10.0, 20.0))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (10.0, 20.0))

    def test_touching_arcs_at_extremes_which_cross_antimeridian(self):
        """Test arcs crossing the antimeridian, touching at the arc extremes."""
        # Touching point at (-175., 45)
        arc1 = SArc(SPoint.from_degrees(-150, 10.0), SPoint.from_degrees(-175, 45))
        arc2 = SArc(SPoint.from_degrees(150, 10.0), SPoint.from_degrees(-175, 45))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (-175, 45))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (-175, 45))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (-175, 45))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (-175, 45))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (-175, 45))

    def test_touching_arcs_at_extremes_at_the_pole(self):
        """Test arcs touching at the arc extremes at the pole."""
        # Touching point at (xxx, 90)
        arc1 = SArc(SPoint.from_degrees(0.0, 10.0), SPoint.from_degrees(0.0, 90.0))
        arc2 = SArc(SPoint.from_degrees(60.0, 10.0), SPoint.from_degrees(10.0, 90.0))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (0., 90))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (0., 90))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (0., 90))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (0., 90))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (0., 90))

    def test_touching_arcs_at_extremes_at_the_antimeridian(self):
        """Test arcs touching at the arc extremes at the antimeridian."""
        # Touching point at (-180., 45)
        arc1 = SArc(SPoint.from_degrees(-150, 10.0), SPoint.from_degrees(-180, 45))
        arc2 = SArc(SPoint.from_degrees(150, 10.0), SPoint.from_degrees(180, 45))

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (-180., 45))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (-180., 45))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (-180., 45))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (-180., 45))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (-180., 45))

    def test_touching_arcs_at_extremes_at_the_equator(self):
        """Test arcs touching at the equator at the arc extremes."""
        # Touching point at the equator (5, 0)
        arc1 = SArc(SPoint.from_degrees(5.0, 0.0), SPoint.from_degrees(0.0, 20.0))
        arc2 = SArc(SPoint.from_degrees(5.0, 0.0), SPoint.from_degrees(0.0, -20.0))
        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, (5.0, 0))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, (5.0, 0))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (5.0, 0))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, (5.0, 0))

    # - with arcs on the same great circle
    def test_touching_arcs_at_extremes_on_same_great_circle(self):
        """Test arcs touching at the arc extremes with arc on the same great circle."""
        touching_point = 5.0, 20.0
        start_point = SPoint.from_degrees(6.0, 1.0)
        end_point = SPoint.from_degrees(*touching_point)
        arc1 = SArc(start_point, end_point)
        forward_point = arc1.forward_points(distance=3000 * 1000)
        arc2 = SArc(end_point, forward_point)

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        # Define  another touching arc on the same great circle
        # BUG [TO SOLVE]
        start_point = SPoint.from_degrees(5.0, 1.0)
        end_point = SPoint.from_degrees(5.0, 20.0)
        arc1 = SArc(start_point, end_point)
        forward_point = arc1.forward_points(distance=3000 * 1000)
        arc2 = SArc(end_point, forward_point)

        #  arc1._great_circle_intersections(arc2) # (0,0) , (-180,0)

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))
        assert not arc1.intersects(arc2)

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert isinstance(p, type(None))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

    def test_touching_arcs_at_extremes_with_equator_arcs(self):
        """Test touching behaviour with arc lying around the equator."""
        touching_point = (0.0, 0.0)
        arc1 = SArc(SPoint.from_degrees(*touching_point), SPoint.from_degrees(50, 0.0))
        arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(*touching_point))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        touching_point = (-180.0, 0.0)
        arc1 = SArc(SPoint.from_degrees(*touching_point), SPoint.from_degrees(50, 0.0))
        arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(*touching_point))

        p = arc2.intersection_point(arc1)
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc1.intersection_point(arc2)
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc1.intersection_point(arc2.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert np.allclose(p.vertices_in_degrees, touching_point)

        # BUG [TO SOLVE] (WHATEVER LON OUTSIDE 0/180 return None)
        touching_point = (-90.0, 0.0)
        arc1 = SArc(SPoint.from_degrees(*touching_point), SPoint.from_degrees(50, 0.0))
        arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(*touching_point))

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))
        assert not arc1.intersects(arc2)

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert isinstance(p, type(None))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        # p = arc2.intersection_point(arc1)
        # assert np.allclose(p.vertices_in_degrees, touching_point)

        # p = arc1.intersection_point(arc2)
        # assert np.allclose(p.vertices_in_degrees, touching_point)

        # p = arc2.intersection_point(arc1.reverse_direction())
        # assert np.allclose(p.vertices_in_degrees, touching_point)

        # p = arc1.intersection_point(arc2.reverse_direction())
        # assert np.allclose(p.vertices_in_degrees, touching_point)

        # p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        # assert np.allclose(p.vertices_in_degrees, touching_point)

    def test_touching_arcs_at_extremes_with_meridian_arcs(self):
        """Test arcs touching at the arc extremes with arcs pointing vertically to the pole."""
        # BUG [TO SOLVE]

        # Touching at point at (1, 20)
        arc1 = SArc(SPoint.from_degrees(1.0, 10.0), SPoint.from_degrees(1.0, 20.0))
        arc2 = SArc(SPoint.from_degrees(1.0, 20.0), SPoint.from_degrees(1.0, 30.0))

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))
        assert not arc1.intersects(arc2)

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert isinstance(p, type(None))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        # Touching at point at (1, -20)
        arc1 = SArc(SPoint.from_degrees(1.0, -10.0), SPoint.from_degrees(1.0, -20.0))
        arc2 = SArc(SPoint.from_degrees(1.0, -20.0), SPoint.from_degrees(1.0, -30.0))

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))
        assert not arc1.intersects(arc2)

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert isinstance(p, type(None))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

    def test_contain_within_arcs(self):
        """Test behaviour with arc contained/within each other."""
        arc1 = SArc(SPoint.from_degrees(5, 1.0), SPoint.from_degrees(5.0, 20.0))
        arc2 = arc1.extend(distance=1000 * 1000, direction="both")

        # Arc2 contain arc1
        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))

        # Arc1 within arc1
        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

    def test_overlapping_arcs(self):
        """Test behaviour with arc partially overlapping."""
        arc1 = SArc(SPoint.from_degrees(5, 1.0), SPoint.from_degrees(5.0, 20.0))
        arc2 = arc1.extend(distance=1000 * 1000, direction="forward")

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

    def test_overlapping_with_meridian_arcs(self):
        """Test behaviour with arc overlapping along a meridian."""
        # Overlapping between (5, 10) and (5, 20)
        arc1 = SArc(SPoint.from_degrees(5, 0.0), SPoint.from_degrees(5.0, 20.0))
        arc2 = SArc(SPoint.from_degrees(5, 10.0), SPoint.from_degrees(5.0, 30.0))

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert isinstance(p, type(None))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

    def test_overlapping_equator_arcs(self):
        """Test behaviour with arc overlapping along the equator."""
        arc1 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(0.0, 0.0))
        arc2 = SArc(SPoint.from_degrees(-50, 0.0), SPoint.from_degrees(-20, 0.0))

        p = arc1.intersection_point(arc2)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1)
        assert isinstance(p, type(None))

        p = arc2.intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

        p = arc1.intersection_point(arc2.reverse_direction())
        assert isinstance(p, type(None))

        p = arc2.reverse_direction().intersection_point(arc1.reverse_direction())
        assert isinstance(p, type(None))

    def test_equal_arcs(self):
        """Test intersection behaviour with equal arcs."""
        arc = SArc(SPoint.from_degrees(1.0, 20.0), SPoint.from_degrees(10.0, 40.0))

        p = arc.intersection_point(arc)
        assert isinstance(p, type(None))

        p = arc.intersection_point(arc.reverse_direction())
        assert isinstance(p, type(None))

        p = arc.reverse_direction().intersection_point(arc)
        assert isinstance(p, type(None))
