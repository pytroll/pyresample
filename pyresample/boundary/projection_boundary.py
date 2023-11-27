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

from pyresample.boundary.base_boundary import BaseBoundary

logger = logging.getLogger(__name__)


def _is_projection_boundary_clockwise(sides_x, sides_y):
    """Determine if the boundary is clockwise-defined in planar coordinates."""
    from shapely.geometry import Polygon
    x = np.concatenate([xs[:-1] for xs in sides_x])
    y = np.concatenate([ys[:-1] for ys in sides_y])
    x = np.hstack((x, x[0]))
    y = np.hstack((y, y[0]))
    polygon = Polygon(zip(x, y))
    return not polygon.exterior.is_ccw


class ProjectionBoundary(BaseBoundary):
    """Projection Boundary object.

    The inputs must be the x and y sides of the projection.
    It expects the projection coordinates to be planar (i.e. metric, radians).
    """

    @classmethod
    def _check_is_boundary_clockwise(cls, sides_x, sides_y, area=None):
        """GeographicBoundary specific implementation."""
        return _is_projection_boundary_clockwise(sides_x=sides_x, sides_y=sides_y)

    @classmethod
    def _compute_boundary_sides(cls, area, vertices_per_side):
        sides_x, sides_y = area._get_projection_sides(vertices_per_side=vertices_per_side)
        return sides_x, sides_y

    def __init__(self, area, vertices_per_side=None):
        super().__init__(area=area, vertices_per_side=vertices_per_side)

        self.sides_x = self._sides_x
        self.sides_y = self._sides_y
        self.crs = self._area.crs
        self.cartopy_crs = self._area.to_cartopy_crs()

    @property
    def _point_inside(self):
        x, y = self._area.get_proj_coords(data_slice=(int(self.shape[0] / 2), int(self.shape[1] / 2)))
        return x.squeeze(), y.squeeze()

    @property
    def x(self):
        """Retrieve boundary x vertices."""
        return self._x

    @property
    def y(self):
        """Retrieve boundary y vertices."""
        return self._y

    def plot(self, ax=None, subplot_kw=None, crs=None, alpha=0.6, **kwargs):
        """Plot the the boundary. crs must be a Cartopy CRS !"""
        from pyresample.visualization.geometries import plot_geometries

        if self.cartopy_crs is None and crs is None:
            raise ValueError("Projection Cartopy 'crs' is required to display projection boundary.")
        if crs is None:
            crs = self.cartopy_crs
        if subplot_kw is None:
            subplot_kw = {"projection": crs}

        geom = self.to_shapely_polygon()
        p = plot_geometries(geometries=[geom], crs=crs,
                            ax=ax, subplot_kw=subplot_kw, alpha=alpha, **kwargs)
        return p
