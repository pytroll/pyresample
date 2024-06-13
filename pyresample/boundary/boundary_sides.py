#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2023 Pyresample developers
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
"""Define the BoundarySides class."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BoundarySides:
    """A class to represent the sides of an area boundary.

    The sides are stored as a tuple of 4 numpy arrays, each representing the
    coordinate (geographic or projected) of the vertices of the boundary side.
    The sides must be stored in the order (top, right, left, bottom),
    which refers to the side position with respect to the coordinate array.
    The first row of the coordinate array correspond to the top side, the last row to the bottom side,
    the first column to the left side and the last column to the right side.
    Please note that the last vertex of each side must be equal to the first vertex of the next side.
    """
    __slots__ = ['_sides']

    def __init__(self, sides: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]):
        """Initialize the BoundarySides object."""
        if len(sides) != 4 or not all(isinstance(side, np.ndarray) and side.ndim == 1 for side in sides):
            raise ValueError("Sides must be a list of four numpy arrays.")

        if not all(np.array_equal(sides[i][-1], sides[(i + 1) % 4][0]) for i in range(4)):
            raise ValueError("The last element of each side must be equal to the first element of the next side.")

        self._sides = tuple(sides)  # Store as a tuple

    @property
    def top(self):
        """Return the vertices of the top side."""
        return self._sides[0]

    @property
    def right(self):
        """Return the vertices of the right side."""
        return self._sides[1]

    @property
    def bottom(self):
        """Return the vertices of the bottom side."""
        return self._sides[2]

    @property
    def left(self):
        """Return the vertices of the left side."""
        return self._sides[3]

    @property
    def vertices(self):
        """Return the vertices of the concatenated sides.

        Note that the last element of each side is discarded to avoid duplicates.
        """
        return np.concatenate([side[:-1] for side in self._sides])

    def __iter__(self):
        """Return an iterator over the sides."""
        return iter(self._sides)

    def __getitem__(self, index):
        """Return the side at the given index."""
        if not isinstance(index, int) or not 0 <= index < 4:
            raise IndexError("Index must be an integer from 0 to 3.")
        return self._sides[index]
