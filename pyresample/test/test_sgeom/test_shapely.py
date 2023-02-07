#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-2022 Pyresample Developers
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
"""Test cases for shapely."""
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


class TestShapely:
    """Test Shapely methods."""

    def test_polygon_unary_union(self):
        """Test MultiPolygon unary union."""
        bounds1 = (-180.0, -90.0, 0.0, 90.0)  # extent1 = [-180, 0, -90, 90]
        bounds2 = (0.0, 0.0, 180.0, 90.0)     # extent2 = [0, 180, -90, 90]
        polygon1 = Polygon.from_bounds(*bounds1)
        polygon2 = Polygon.from_bounds(*bounds2)
        multipolygon = MultiPolygon([polygon1, polygon2])
        unified_polygon = unary_union(multipolygon)
        assert isinstance(unified_polygon, Polygon)
        vertices = np.array(unified_polygon.exterior.coords)
        expected_vertices = np.array([[-180., 90.],
                                      [0., 90.],
                                      [180., 90.],
                                      [180., 0.],
                                      [0., 0.],
                                      [0., -90.],
                                      [-180., -90.],
                                      [-180., 90.]])
        np.testing.assert_allclose(expected_vertices, vertices)

    def test_polygon_from_bounds(self):
        """Test Polygon definition from bounds."""
        global_polygon = Polygon.from_bounds(-180, -90, 180, 90)
        assert global_polygon.bounds == (-180.0, -90.0, 180.0, 90.0)

    def test_polygon_equals(self):
        # First polygon goes from -180 to 0 longitude
        vertices1 = np.array([[-180., -90.],
                              [-180., 90.],
                              [0., 90.],
                              [0., -90.],
                              [-180., -90.]])
        polygon1 = Polygon(vertices1)
        # Second polygon goes from 0 to 180 longitude
        vertices2 = np.array([[0., -90.],
                              [0., 90.],
                              [180., 90.],
                              [180., -90.],
                              [0., -90.]])
        polygon2 = Polygon(vertices2)

        # Global separate polygon from -180 to 180
        global_separate = MultiPolygon([polygon1, polygon2])

        # Global single polygon from -180 to 180
        global_polygon = Polygon.from_bounds(-180, -90, 180, 90)

        # Unioned polygon
        global_union = unary_union(global_separate)

        # Checks topological equality
        assert global_polygon.equals(global_union)
        assert global_polygon.equals(global_separate)
