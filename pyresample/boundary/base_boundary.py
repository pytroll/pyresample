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
"""Define the BaseBoundary class."""

import logging

import numpy as np

from pyresample.boundary.sides import BoundarySides


logger = logging.getLogger(__name__)


class BaseBoundary:
    __slots__ = ["_sides_x", "_sides_y"]
    
    def __init__(self, sides_x, sides_y, order=None):
        self._sides_x = BoundarySides(sides_x)
        self._sides_y = BoundarySides(sides_y)
        
        self.is_clockwise = self._check_is_boundary_clockwise(sides_x, sides_y)
        self.is_counterclockwise = not self.is_clockwise
        self._set_order(order)
                    
    def _check_is_boundary_clockwise(self, sides_x, sides_y): 
        raise NotImplementedError()
    
    def _set_order(self, order):
        if self.is_clockwise:
            self._actual_order = "clockwise"
        else:
            self._actual_order = "counterclockwise"

        if order is None:
            self._wished_order = self._actual_order
        else:
            if order not in ["clockwise", "counterclockwise"]:
                raise ValueError("Valid 'order' is 'clockwise' or 'counterclockwise'")
            self._wished_order = order

    def set_clockwise(self):
        """Set clockwise order for vertices retrieval."""
        self._wished_order = "clockwise"
        return self

    def set_counterclockwise(self):
        """Set counterclockwise order for vertices retrieval."""
        self._wished_order = "counterclockwise"
        return self
    
    @property
    def sides(self):
        """Return the boundary sides as a tuple of (sides_x, sides_y) arrays."""
        return self._sides_x, self._sides_y
    
    @property
    def _x(self):
        """Retrieve boundary x vertices."""
        xs = self._sides_x.vertices
        if self._wished_order == self._actual_order:
            return xs
        else:
            return xs[::-1]

    @property
    def _y(self):
        """Retrieve boundary y vertices."""
        ys = self._sides_y.vertices
        if self._wished_order == self._actual_order:
            return ys
        else:
            return ys[::-1]

    @property
    def vertices(self):
        """Return boundary vertices 2D array [x, y]."""
        vertices = np.vstack((self._x, self._y)).T
        vertices = vertices.astype(np.float64, copy=False)  # Important for spherical ops.
        return vertices
    
    def contour(self, closed=False):
        """Return the (x, y) tuple of the boundary object.

        If excludes the last element of each side because it's included in the next side.
        If closed=False (the default), the last vertex is not equal to the first vertex
        If closed=True, the last vertex is set to be equal to the first
        closed=True is required for shapely Polygon creation.
        closed=False is required for pyresample SPolygon creation.
        """
        x = self._x
        y = self._y
        if closed:
            x = np.hstack((x, x[0]))
            y = np.hstack((y, y[0]))
        return x, y

    def _to_shapely_polygon(self):
        """Define a Shapely Polygon."""
        from shapely.geometry import Polygon
        self = self.set_counterclockwise() # FIXME: add exception for pole wrapping polygons
        x, y = self.contour(closed=True)
        return Polygon(zip(x, y))

