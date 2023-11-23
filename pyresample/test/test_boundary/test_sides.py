#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2022 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test the BoundarySides objects."""

import pytest
import numpy as np
from pyresample.boundary.sides import BoundarySides


class TestBoundarySides:
    """Test suite for the BoundarySides class with 1D numpy arrays for sides."""

    def test_initialization_valid_input(self):
        """Test initialization with valid 1D numpy array inputs."""
        sides = [np.array([1, 2, 3]),  # top
                 np.array([3, 4, 5]),  # right
                 np.array([5, 6, 7]),  # bottom
                 np.array([7, 8, 1])]  # left
        boundary = BoundarySides(sides)
        assert all(np.array_equal(boundary[i], sides[i]) for i in range(4))

    def test_initialization_invalid_input(self):
        """Test initialization with invalid inputs, such as wrong number of sides or non-1D arrays."""
        with pytest.raises(ValueError):
            BoundarySides([np.array([1, 2]),  # Invalid number of sides
                           np.array([2, 3])])

        with pytest.raises(ValueError):
            BoundarySides([np.array([1, 2]),  # Non-1D arrays
                           np.array([[2, 3], [4, 5]]),
                           np.array([5, 6]),
                           np.array([6, 7])])

        with pytest.raises(ValueError):
            BoundarySides([np.array([1, 2]),  # Invalid side connection
                           np.array([3, 4]),
                           np.array([4, 6]),
                           np.array([6, 1])])

    def test_property_accessors(self):
        """Test property accessors with 1D numpy arrays."""
        sides = [np.array([1, 2, 3]),  # top
                 np.array([3, 4, 5]),  # right
                 np.array([5, 6, 7]),  # bottom
                 np.array([7, 8, 1])]  # left
        boundary = BoundarySides(sides)
        assert np.array_equal(boundary.top, sides[0])
        assert np.array_equal(boundary.right, sides[1])
        assert np.array_equal(boundary.bottom, sides[2])
        assert np.array_equal(boundary.left, sides[3])

    def test_vertices_property(self):
        """Test the vertices property with concatenated 1D numpy arrays."""
        sides = [np.array([1, 2, 3]),  # top
                 np.array([3, 4, 5]),  # right
                 np.array([5, 6, 7]),  # bottom
                 np.array([7, 8, 1])]  # left
        boundary = BoundarySides(sides)
        expected_vertices = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        assert np.array_equal(boundary.vertices, expected_vertices)

    def test_iteration(self):
        """Test iteration over the 1D numpy array sides."""
        sides = [np.array([1, 2, 3]),  # top
                 np.array([3, 4, 5]),  # right
                 np.array([5, 6, 7]),  # bottom
                 np.array([7, 8, 1])]  # left
        boundary = BoundarySides(sides)
        for i, side in enumerate(boundary):
            assert np.array_equal(side, sides[i])

    def test_indexing_valid(self):
        """Test valid indexing with 1D numpy arrays."""
        sides = [np.array([1, 2, 3]),  # top
                 np.array([3, 4, 5]),  # right
                 np.array([5, 6, 7]),  # bottom
                 np.array([7, 8, 1])]  # left
        boundary = BoundarySides(sides)
        for i in range(4):
            assert np.array_equal(boundary[i], sides[i])

    def test_indexing_invalid(self):
        """Test indexing with invalid indices."""
        sides = [np.array([1, 2, 3]),  # top
                 np.array([3, 4, 5]),  # right
                 np.array([5, 6, 7]),  # bottom
                 np.array([7, 8, 1])]  # left
        boundary = BoundarySides(sides)
        with pytest.raises(IndexError):
            boundary[4]  # Invalid index
