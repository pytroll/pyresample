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
"""Define the ProjectionBoundary class."""

import logging

import numpy as np

from pyresample.boundary.sides import BoundarySides

logger = logging.getLogger(__name__)


class ProjectionBoundary():
    """Projection Boundary object.

    The inputs must be the x and y sides of the projection.
    It expects the projection coordinates to be planar (i.e. metric, radians).
    """

    def __init__(self, sides_x, sides_y, wished_order=None, crs=None):

        self.crs = crs  # TODO needed to plot
        
        self.sides_x = BoundarySides(sides_x)
        self.sides_y = BoundarySides(sides_y)

        # Check if it is clockwise/counterclockwise
        self.is_clockwise = self._is_projection_boundary_clockwise()
        self.is_counterclockwise = not self.is_clockwise

        # Define wished order
        if self.is_clockwise:
            self._actual_order = "clockwise"
        else:
            self._actual_order = "counterclockwise"

        if wished_order is None:
            self._wished_order = self._actual_order
        else:
            if wished_order not in ["clockwise", "counterclockwise"]:
                raise ValueError("Valid order is 'clockwise' or 'counterclockwise'")
            self._wished_order = wished_order

    def _is_projection_boundary_clockwise(self):
        """Determine if the boundary is clockwise-defined in projection coordinates."""
        from shapely.geometry import Polygon

        x = np.concatenate([xs[:-1] for xs in self.sides_x])
        y = np.concatenate([ys[:-1] for ys in self.sides_y])
        x = np.hstack((x, x[0]))
        y = np.hstack((y, y[0]))
        polygon = Polygon(zip(x, y))
        return not polygon.exterior.is_ccw

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
        return self.sides_x, self.sides_y

    @property
    def x(self):
        """Retrieve boundary x vertices."""
        xs = np.concatenate([xs[:-1] for xs in self.sides_x])
        if self._wished_order == self._actual_order:
            return xs
        else:
            return xs[::-1]

    @property
    def y(self):
        """Retrieve boundary y vertices."""
        ys = np.concatenate([ys[:-1] for ys in self.sides_y])
        if self._wished_order == self._actual_order:
            return ys
        else:
            return ys[::-1]

    @property
    def vertices(self):
        """Return boundary vertices 2D array [x, y]."""
        vertices = np.vstack((self.x, self.y)).T
        vertices = vertices.astype(np.float64, copy=False)
        return vertices

    def contour(self, closed=False):
        """Return the (x, y) tuple of the boundary object.

        If excludes the last element of each side because it's included in the next side.
        If closed=False (the default), the last vertex is not equal to the first vertex
        If closed=True, the last vertex is set to be equal to the first
        closed=True is required for shapely Polygon creation.
        """
        x = self.x
        y = self.y
        if closed:
            x = np.hstack((x, x[0]))
            y = np.hstack((y, y[0]))
        return x, y

    def _to_shapely_polygon(self):
        from shapely.geometry import Polygon
        self = self.set_counterclockwise()
        x, y = self.contour(closed=True)
        return Polygon(zip(x, y))

    def polygon(self, shapely=True):
        """Return the boundary polygon."""
        if shapely:
            return self._to_shapely_polygon()
        else:
            raise NotImplementedError("Only shapely polygon available.")

    def plot(self, ax=None, subplot_kw=None, crs=None, **kwargs):
        """Plot the the boundary."""
        from pyresample.visualization.geometries import plot_geometries

        if self.crs is None and crs is None:
            raise ValueError("Projection 'crs' is required to display projection boundary.")
        if crs is None:
            crs = self.crs

        geom = self.polygon(shapely=True)
        p = plot_geometries(geometries=[geom], crs=crs,
                            ax=ax, subplot_kw=subplot_kw, **kwargs)
        return p



