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
"""Define the GeographicBoundary class."""

import logging

import numpy as np
from pyproj import CRS

from pyresample.boundary.area_boundary import Boundary as OldBoundary
from pyresample.boundary.base_boundary import BaseBoundary
from pyresample.spherical import SphPolygon

logger = logging.getLogger(__name__)


def _get_swath_local_inside_point_from_sides(sides_x, sides_y, start_idx=0):
    """Retrieve point inside boundary close to the point used to determine order.

    This is required for global areas (spanning more than 180 degrees) and swaths.
    The start_idx indicates the opposites sides corners (start_idx, start_idx+2) from
    which to try identify a point inside the polygon.
    """
    # NOTE: the name of the object here refers to start_idx=0
    from pyresample.spherical import Arc, SCoordinate
    idx1 = start_idx
    idx2 = (start_idx + 2) % 4

    top_corner = sides_x[idx1][0], sides_y[idx1][0]
    top_right_point = sides_x[idx1][1], sides_y[idx1][1]
    bottom_corner = sides_x[idx2][-1], sides_y[idx2][-1]
    bottom_right_point = sides_x[idx2][-2], sides_y[idx2][-2]
    point_top_corner = SCoordinate(*np.deg2rad(top_corner))
    point_top_right_point = SCoordinate(*np.deg2rad(top_right_point))
    point_bottom_corner = SCoordinate(*np.deg2rad(bottom_corner))
    point_bottom_right_point = SCoordinate(*np.deg2rad(bottom_right_point))
    arc1 = Arc(point_top_corner, point_bottom_right_point)
    arc2 = Arc(point_top_right_point, point_bottom_corner)
    point_inside = arc1.intersection(arc2)
    if point_inside is not None:
        point_inside = point_inside.vertices_in_degrees[0]
    return point_inside


def _try_get_local_inside_point(sides_x, sides_y):
    """Try to get a local inside point from one of the 4 boundary sides corners."""
    for start_idx in range(0, 4):
        point_inside = _get_swath_local_inside_point_from_sides(sides_x, sides_y, start_idx=start_idx)
        if point_inside is not None:
            return point_inside, start_idx
    else:
        return None, None


def _is_clockwise_order(first_point, second_point, point_inside):
    """Determine if polygon coordinates follow a clockwise path.

    This uses :class:`pyresample.spherical.Arc` to determine the angle
    between a polygon arc segment and a point known to be inside the polygon.

    Note: pyresample.spherical assumes angles are positive if counterclockwise.
    Note: if the longitude distance between the first_point/second_point and point_inside is
    larger than 180°, the function likely return a wrong unexpected result !
    """
    from pyresample.spherical import Arc, SCoordinate
    point1 = SCoordinate(*np.deg2rad(first_point))
    point2 = SCoordinate(*np.deg2rad(second_point))
    point3 = SCoordinate(*np.deg2rad(point_inside))
    arc12 = Arc(point1, point2)
    arc23 = Arc(point2, point3)
    angle = arc12.angle(arc23)
    is_clockwise = -np.pi < angle < 0
    return is_clockwise


def _check_is_clockwise(area, sides_x, sides_y):
    from pyresample import SwathDefinition

    if isinstance(area, SwathDefinition):
        point_inside, start_idx = _try_get_local_inside_point(sides_x, sides_y)
        first_point = sides_x[start_idx][0], sides_y[start_idx][0]
        second_point = sides_x[start_idx][1], sides_y[start_idx][1]
        return _is_clockwise_order(first_point, second_point, point_inside)
    else:
        if area.is_geostationary:
            point_inside = area.get_lonlat(row=int(area.shape[0] / 2), col=int(area.shape[1] / 2))
            first_point = sides_x[0][0], sides_y[0][0]
            second_point = sides_x[0][1], sides_y[0][1]
            return _is_clockwise_order(first_point, second_point, point_inside)
        else:
            return True


class GeographicBoundary(BaseBoundary, OldBoundary):
    """GeographicBoundary object.

    The inputs must be the list of longitude and latitude boundary sides.
    """
    # NOTES:
    # - Boundary provide the ancient method contour_poly and draw
    #   from the old interface for compatibility to AreaBoundary

    @classmethod
    def _check_is_boundary_clockwise(cls, sides_x, sides_y, area):
        """GeographicBoundary specific implementation."""
        return _check_is_clockwise(area, sides_x, sides_y)

    @classmethod
    def _compute_boundary_sides(cls, area, vertices_per_side):
        sides_lons, sides_lats = area._get_geographic_sides(vertices_per_side=vertices_per_side)
        return sides_lons, sides_lats

    def __init__(self, area, vertices_per_side=None):
        super().__init__(area=area, vertices_per_side=vertices_per_side)

        self.sides_lons = self._sides_x
        self.sides_lats = self._sides_y

        # Define CRS
        if self.is_swath:
            crs = self._area.crs
        else:
            crs = CRS(proj="longlat", ellps="WGS84")  # FIXME: AreaDefinition.get_lonlat for geographic projections?
        self.crs = crs

        # Backcompatibility with old AreaBoundary
        self._contour_poly = None

    @property
    def is_swath(self):
        """Determine if is the boundary of a swath."""
        return self._area.__class__.__name__ == "SwathDefinition"

    @property
    def lons(self):
        """Retrieve boundary longitude vertices."""
        return self._x

    @property
    def lats(self):
        """Retrieve boundary latitude vertices."""
        return self._y

    @property
    def polygon(self):
        """Return the boundary spherical polygon."""
        self = self.set_clockwise()
        if self._contour_poly is None:
            self._contour_poly = SphPolygon(np.deg2rad(self.vertices))
        return self._contour_poly

    def plot(self, ax=None, subplot_kw=None, alpha=0.6, **kwargs):
        """Plot the the boundary."""
        import cartopy.crs as ccrs

        from pyresample.visualization.geometries import plot_geometries

        geom = self.to_shapely_polygon()
        crs = ccrs.Geodetic()
        p = plot_geometries(geometries=[geom], crs=crs,
                            ax=ax, subplot_kw=subplot_kw, alpha=alpha, **kwargs)
        return p
