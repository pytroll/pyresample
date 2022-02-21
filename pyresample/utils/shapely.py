#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2022 Pyresample developers
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
"""Tools to deal with geometry operations."""
import math

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import MultiPoint  # Point
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)

# ---------------------------------------------------------------------------.
# Vertices Processing


def _close_vertices(vertices):
    """Ensure last vertex is equal to first."""
    if not np.allclose(vertices[0, :], vertices[-1, :]):
        return np.vstack((vertices, vertices[0, :]))
    else:
        return vertices


def _unclose_vertices(vertices):
    """Ensure last vertex is not equal to first."""
    if np.allclose(vertices[0, :], vertices[-1, :]):
        return vertices[:-1, :]
    else:
        return vertices


def remove_duplicated_vertices(arr, tol=1e-08):
    """Remove sequential duplicated (lon, lat) coordinates."""
    x = arr[:, 0]
    y = arr[:, 1]
    x_idx = np.where(np.abs(np.ediff1d(x)) <= tol)[0] + 1
    y_idx = np.where(np.abs(np.ediff1d(y)) <= tol)[0] + 1
    duplicate_idx = np.intersect1d(x_idx, y_idx)
    if len(duplicate_idx) > 0:
        print("There are duplicated vertices... removing it.")  # TODO remove
        arr = np.delete(arr, duplicate_idx, 0)
    if len(arr) == 0:
        raise ValueError("All duplicated values.")
    return arr

# ---------------------------------------------------------------------------.
# Antimeridian Processing


def get_idx_antimeridian_crossing(vertices, radians=False):
    """Return the indices at which the boundary cross the antimeridian.

    Assumption:
    - Any two consecutive points with over 180 degrees difference in
      longitude are assumed to cross the antimeridian.
    """
    if radians:
        thr = np.pi
    else:
        thr = 180
    idx_crossing = np.where(np.abs(np.diff(vertices[:, 0])) > thr)[0]
    return idx_crossing


def get_number_antimeridian_crossing(vertices, radians=False):
    """Return the number of times the boundary cross the antimeridian.

    Assumption:
    - Any two consecutive points with over 180 degrees difference in
      longitude are assumed to cross the antimeridian.
    """
    idx_crossing = get_idx_antimeridian_crossing(vertices, radians=radians)
    return len(idx_crossing)


def is_antimeridian_crossing(vertices, radians=False):
    """Check if antimeridian crossing.

    Assumption:
    - Any two consecutive points with over 180 degrees difference in
      longitude are assumed to cross the antimeridian.
    """
    n_crossing = get_number_antimeridian_crossing(vertices, radians=radians)
    if n_crossing == 0:
        return False
    else:
        return True


def _get_antimeridian_crossing_point(coord_start, coord_end):
    """Compute the point where the arc cross the antimeridian.

    It expects coordinates in degree !
    """
    from pyresample.spherical import SArc, SPoint

    # Retrieve start and end lon/lat coordinates of the arc crossing the antimeridian
    lon_start = coord_start[0]
    lat_start = coord_start[1]
    lon_end = coord_end[0]
    lat_end = coord_end[1]

    # Define spherical arcs
    antimeridian_arc = SArc(SPoint(np.deg2rad(180), np.deg2rad(-90)),
                            SPoint(np.deg2rad(180), np.deg2rad(90))
                            )

    crossing_arc = SArc(SPoint(np.deg2rad(lon_start), np.deg2rad(lat_start)),
                        SPoint(np.deg2rad(lon_end), np.deg2rad(lat_end))
                        )

    # Retrieve crossing point
    crossing_point = crossing_arc.intersection(antimeridian_arc)
    clat = np.rad2deg(crossing_point.lat)

    # Identify direction
    # -1 --> toward East --> 180
    #  1 --> toward West --> -180
    direction = math.copysign(1, lon_end - lon_start)
    if direction == -1:
        clon = 180
    else:
        clon = -180

    # Crossing point at the antimeridian
    split_coord = [clon, clat]

    # Return
    return split_coord


def get_antimeridian_safe_line_vertices(vertices):
    """Split lines at the antimeridian.

    It return a list of line vertices not crossing the antimeridian.
    It expects vertices in degree !
    """
    # Retrieve line vertices when crossing the antimeridian
    idx_crossing = np.where(np.abs(np.diff(vertices[:, 0])) > 180)[0]
    # If no crossing, return vertices
    n_crossing = len(idx_crossing)
    if n_crossing == 0:
        return [vertices]
    # Split line at anti-meridians
    previous_idx = 0
    previous_split_coord = None
    list_vertices = []
    for i in range(0, n_crossing):
        # - Retrieve coordinates around anti-meridian crossing
        tmp_idx_crossing = idx_crossing[i]
        coord1 = vertices[tmp_idx_crossing]
        coord2 = vertices[tmp_idx_crossing + 1]
        # - Retrieve anti-meridian split coordinates
        split_coord = _get_antimeridian_crossing_point(coord1, coord2)
        # - Retrieve line vertices
        new_coords = vertices[previous_idx:tmp_idx_crossing + 1, :]
        if previous_split_coord is None:
            new_vertices = np.vstack((new_coords, split_coord))
        else:
            new_vertices = np.vstack((previous_split_coord, new_coords, split_coord))
        # - Update previous idx
        previous_idx = tmp_idx_crossing + 1
        # - Update previous split coords
        previous_split_coord = split_coord
        previous_split_coord[0] = -180 if split_coord[0] == 180 else 180
        # - Append polygon vertices to the list
        list_vertices.append(new_vertices)
    # Add last vertices
    new_coords = vertices[previous_idx:, :]
    new_vertices = np.vstack((previous_split_coord, new_coords))
    list_vertices.append(new_vertices)
    # Return list of vertices
    return list_vertices


def get_antimeridian_safe_polygon_vertices(vertices):
    """Split polygons at the antimeridian.

    It return a list of polygon vertices not crossing the antimeridian.
    Each vertices array is unwrapped (last vertex is not equal to first.)

    If the polygon enclose a pole, the processing can fail or return wrong output
      without warnings. !!!
    It also does not account for holes in the polygons !
    It expects vertices in degree !
    """
    from spherical1 import unwrap_longitude_degree

    # Wrap vertices
    vertices = _close_vertices(vertices)

    # Ensure 180 longitude is converted to -180
    # - So that if longitude are [-180 180, -180 180, ...] are considered equal !
    vertices[:, 0] = unwrap_longitude_degree(vertices[:, 0])

    # Retrieve line vertices when crossing the antimeridian
    idx_crossing = np.where(np.abs(np.diff(vertices[:, 0])) > 180)[0]
    # If no crossing, return vertices
    n_crossing = len(idx_crossing)
    if n_crossing == 0:
        return [_unclose_vertices(vertices)]
    if n_crossing == 1:
        # print("Can not deal with polygons enclosing the poles yet")
        # raise NotImplementedError # TODO
        return [_unclose_vertices(vertices)]

    # Check that there are an even number of antimeridian crossing points
    if (n_crossing % 2) != 0:
        raise ValueError("Expecting a even number of antimeridian crossing point of polygon vertices.")

    # Split polygons at anti-meridians
    previous_idx = 0
    previous_idx_rev = -1
    previous_split_coord = None
    previous_split_coord_rev = None
    list_vertices = []
    for i in range(0, int(n_crossing / 2) + 1):
        # - Define index for reverse coordinate
        j = -1 - i
        # - Retrieve coordinates around anti-meridian crossing
        tmp_idx_crossing = idx_crossing[i]
        tmp_idx_crossing_rev = idx_crossing[j]
        coord1 = vertices[tmp_idx_crossing]
        coord2 = vertices[tmp_idx_crossing + 1]
        coord1_rev = vertices[tmp_idx_crossing_rev + 1]
        coord2_rev = vertices[tmp_idx_crossing_rev]
        # - Retrieve anti-meridian split coordinates
        split_coord = _get_antimeridian_crossing_point(coord1, coord2)
        split_coord_rev = _get_antimeridian_crossing_point(coord1_rev, coord2_rev)
        # - Retrieve polygon vertices
        new_coords = vertices[previous_idx:tmp_idx_crossing + 1, :]
        new_coords_rev = vertices[tmp_idx_crossing_rev + 1:previous_idx_rev, :]
        if i != int(n_crossing / 2):
            if previous_split_coord is None:
                new_vertices = np.vstack((new_coords, split_coord, split_coord_rev, new_coords_rev))
            else:
                new_vertices = np.vstack((previous_split_coord, new_coords, split_coord,
                                          split_coord_rev, new_coords_rev, previous_split_coord_rev))
        else:
            new_vertices = np.vstack((previous_split_coord, new_coords, split_coord))
        # - Update previous idx
        previous_idx = tmp_idx_crossing + 1
        previous_idx_rev = tmp_idx_crossing_rev
        # - Update previous split coords
        previous_split_coord = split_coord
        previous_split_coord_rev = split_coord_rev
        previous_split_coord[0] = -180 if split_coord[0] == 180 else 180
        previous_split_coord_rev[0] = -180 if split_coord_rev[0] == 180 else 180
        # - Append polygon vertices to the list
        list_vertices.append(new_vertices)

    # Return list of (unwrapped) vertices
    list_vertices = [_unclose_vertices(vertices) for vertices in list_vertices]

    return list_vertices

# --------------------------------------------------------------------------.
#  Extent tools


def bounds_from_extent(extent):
    """Get shapely bounds from a matplotlib/cartopy extent.

    # Shapely bounds
    bounds = [min_x, min_y, max_x, max_y]

    # Matplotlib extent
    extent = [min_x, max_x, min_y, max_y]
    """
    bounds = [extent[0], extent[2], extent[1], extent[3]]
    return bounds


def extent_from_bounds(bounds, x_margin=None, y_margin=None):
    """Get matplotlib/cartopy extent from shapely bounds.

    x_margin and ymargin enable to extend the extent by custom degrees.

    # Shapely bounds
    bounds = [min_x, min_y, max_x, max_y]

    # Matplotlib extent
    extent = [min_x, max_x, min_y, max_y]
    """
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    if (x_margin is not None) or (y_margin is not None):
        extent = extend_extent(extent, x_margin=x_margin, y_margin=y_margin)
    return extent


def extend_extent(extent, x_margin=0.1, y_margin=0.1):
    """Extend an extent on x and y sides by x/y degree.

    Extent is defined as: [x_min, x_max, y_min, y_max]
    x and y are clipped at [-180, 180] and [-90,90]

    """
    if x_margin is None:
        x_margin = 0
    if y_margin is None:
        y_margin = 0
    extent[0] = extent[0] - x_margin
    extent[1] = extent[1] + x_margin
    extent[2] = extent[2] - y_margin
    extent[3] = extent[3] + y_margin
    if extent[0] < -180:
        extent[0] = -180
    if extent[1] > 180:
        extent[1] = 180
    if extent[2] < -90:
        extent[2] = -90
    if extent[3] > 90:
        extent[3] = 90
    return extent


def _get_extent_from_vertices(vertices):
    """Compute extent using min/max on lon and lat columns."""
    extent = [np.min(vertices[:, 0]),
              np.max(vertices[:, 0]),
              np.min(vertices[:, 1]),
              np.max(vertices[:, 1])]
    return extent


def get_indices_with_side_duplicates(x):
    """Return indices with duplicate values on both sides."""
    indices = []  # [0, len(y)-1]
    for i in range(1, len(x) - 1):
        if x[i - 1] == x[i] and x[i] == x[i + 1]:
            indices.append(i)
    return indices


def simplify_rectangle_vertices(vertices):
    """Simplify rectangle vertices to corners vertices."""
    vertices = _unclose_vertices(vertices)
    # Remove duplicated vertices along lats (>3 sequential duplicates)
    while vertices[0, 1] == vertices[-1, 1]:
        vertices = np.roll(vertices, 1, axis=0)
    idx_to_remove = get_indices_with_side_duplicates(vertices[:, 1])
    vertices = np.delete(vertices, idx_to_remove, axis=0)
    # Remove duplicated vertices along lons (>3 sequential duplicates)
    while vertices[0, 0] == vertices[-1, 0]:
        vertices = np.roll(vertices, 1, axis=0)
    idx_to_remove = get_indices_with_side_duplicates(vertices[:, 0])
    vertices = np.delete(vertices, idx_to_remove, axis=0)
    return vertices


def get_rectangle_splitter_line(start, end):
    """Create a LineString to split rectilinear polygons."""
    if start[0] == end[0]:
        line = LineString([(start[0], -90), (start[0], 90)])
    else:
        line = LineString([(-180, start[1]), (180, start[1])])
    return line


def decompose_rectilinear_polygons_into_rectangles(polygon):
    """Decompose rectilinear polygons into many rectangles."""
    polygon = shapely.ops.unary_union(polygon)
    if isinstance(polygon, MultiPolygon):
        list_polygons = list(polygon.geoms)
    else:
        list_polygons = [polygon]

    # Initialize lists
    list_extent_polygons = []
    list_to_split = []

    # Identify polygons with > 4 vertices to be splitted
    for p in list_polygons:
        vertices = np.array(p.exterior.coords)
        vertices = simplify_rectangle_vertices(vertices)
        # If only -180 and 180 longitudes, infer rectangle vertices from extent
        if np.all(np.isin(vertices[:, 0], [-180, 180])):
            extent = _get_extent_from_vertices(vertices)
            p = Polygon.from_bounds(*bounds_from_extent(extent))
        else:
            p = Polygon(vertices)
        if len(p.exterior.coords) == 5:
            list_extent_polygons.append(p)
        else:
            list_to_split.append(p)

    # Initialize coords index
    i = 0
    while len(list_to_split) > 0:
        # Get a polygon with more than 4 vertices
        polygon_to_split = list_to_split[0]
        list_coords = list(polygon_to_split.exterior.coords)

        start, end = list_coords[i:i + 2]

        splitter_line = get_rectangle_splitter_line(start, end)
        splitted_geom = shapely.ops.split(polygon_to_split, splitter_line)
        if len(splitted_geom.geoms) == 1:
            splitted_geom = Polygon(splitted_geom.geoms[0])
        else:
            splitted_geom = MultiPolygon(splitted_geom)
        # If some splitting occur, update list_to_split
        if not splitted_geom.equals_exact(polygon_to_split, tolerance=1e-8):
            del list_to_split[0]
            i = 0
            if isinstance(splitted_geom, MultiPolygon) or isinstance(splitted_geom, GeometryCollection):
                for geom in splitted_geom.geoms:
                    vertices = np.array(geom.exterior.coords)
                    vertices = simplify_rectangle_vertices(vertices)
                    geom = Polygon(vertices)
                    if len(geom.exterior.coords) == 5:
                        list_extent_polygons.append(geom)
                    else:
                        list_to_split.append(geom)
            elif isinstance(splitted_geom, Polygon):
                if len(splitted_geom.exterior.coords) == 5:
                    list_extent_polygons.append(splitted_geom)
                    raise ValueError("This should not happen.")
                else:
                    raise ValueError("This should not happen.")
        else:
            i += 1

    return MultiPolygon(list_extent_polygons)


def get_non_overlapping_list_extents(list_extent):
    """Given a list of extents, return a list of non-overlapping extents."""
    p = MultiPolygon([Polygon.from_bounds(*bounds_from_extent(ext)) for ext in list_extent])
    p = shapely.ops.unary_union(p)
    p = decompose_rectilinear_polygons_into_rectangles(p)

    if isinstance(p, MultiPolygon):
        list_extent = [extent_from_bounds(geom.bounds) for geom in p.geoms]
    else:  # Polygon
        list_extent = [extent_from_bounds(p.bounds)]
    return list_extent


def _check_valid_extent(extent, use_radians=False):
    """Check lat/lon extent validity."""
    if len(extent) != 4:
        raise ValueError("'extent' must have length 4: [lon_min, lon_max, lat_min, lat_max].")
    if not isinstance(extent, (tuple, list, np.ndarray)):
        raise TypeError("'extent' must be a list, tuple or np.array. [lon_min, lon_max, lat_min, lat_max].")
    extent = np.array(extent)
    # Check extent order validity
    if extent[0] > extent[1]:
        raise ValueError('extent[0] (aka lon_min) must be smaller than extent[1] (aka lon_max).')
    if extent[2] > extent[3]:
        raise ValueError('extent[2] (aka lat_min) must be smaller than extent[2] (aka lat_max).')
    # Check min max values
    if use_radians:
        if extent[0] < -np.pi:
            raise ValueError('extent[0] (aka lon_min) must be equal or larger than -π.')
        if extent[1] > np.pi:
            raise ValueError('extent[1] (aka lon_max) must be equal or smaller than π.')
        if extent[2] < -np.pi / 2:
            raise ValueError('extent[2] (aka lat_min) must be equal or larger than -π/2.')
        if extent[3] < -np.pi / 2:
            raise ValueError('extent[3] (aka lat_max) must be equal or larger than π/2.')
    else:
        if extent[0] < -180:
            raise ValueError('extent[0] (aka lon_min) must be equal or larger than -180.')
        if extent[1] > 180:
            raise ValueError('extent[1] (aka lon_max) must be equal or smaller than 180.')
        if extent[2] < -90:
            raise ValueError('extent[2] (aka lat_min) must be equal or larger than -90.')
        if extent[3] < -180:
            raise ValueError('extent[3] (aka lat_max) must be equal or larger than 90.')
    return extent.tolist()

# --------------------------------------------------------------------------.
# Shapely utils à la POSTGIS / sf style


def st_add_x_offset(geom, offset):
    """Add an offset to x coordinates."""
    if isinstance(geom, (MultiPolygon, MultiLineString, MultiPoint)):
        return type(geom)([shapely.affinity.translate(geom[i], xoff=offset) for i in range(len(geom.geoms))])
    else:
        return shapely.affinity.translate(geom, xoff=offset)


def st_add_y_offset(geom, offset):
    """Add an offset on y coordinates."""
    if isinstance(geom, (MultiPolygon, MultiLineString, MultiPoint)):
        return type(geom)([shapely.affinity.translate(geom[i], yoff=offset) for i in range(len(geom.geoms))])
    else:
        return shapely.affinity.translate(geom, yoff=offset)


def st_polygon_clockwise(geom):
    """Ensure shapely Polygon or MultiPolygon to be clockwise oriented."""
    # Check geometry type
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise TypeError("Expects Polygon or MultiPolygon.")
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([shapely.geometry.polygon.orient(geom[i], -1) for i in range(len(geom.geoms))])
    else:
        return shapely.geometry.polygon.orient(geom, -1)


def st_polygon_counterclockwise(geom):
    """Ensure shapely Polygon or MultiPolygon to be counterclockwise oriented."""
    # Check geometry type
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise TypeError("Expects Polygon or MultiPolygon.")
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([shapely.geometry.polygon.orient(geom[i], 1) for i in range(len(geom.geoms))])
    else:
        return shapely.geometry.polygon.orient(geom, 1)


def st_polygon_antimeridian_safe(geom):
    """Sanitize shapely polygons crossing the antimeridian.

    Given a Shapely Polygon or MultiPolygon representation of a polygon,
    returns a MultiPolygon of 'antimeridian-safe' constituent polygons splitted at the anti-meridian.
    The returned MultiPolygon ensure compliance with GeoJSON standards
    GeoJSON standards: https://tools.ietf.org/html/rfc7946#section-3.1.9

    Assumptions:
      - Any two consecutive points with over 180 degrees difference in
        longitude are assumed to cross the antimeridian.
      - The polygon can wrap multiple time across the globe and cross the antimeridian on multiple occasions.
      - If the polygon enclose a pole, the processing can fail or return wrong output
         without warnings. !!!
      - Does not account for holes in the polygons !

    Returns:
        MultiPolygon: antimeridian-safe polygon(s)
    """
    # Check geometry type
    if not isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
        raise TypeError("Expects Polygon or MultiPolygon.")
    # Get list of vertices
    if isinstance(geom, shapely.geometry.Polygon):
        list_vertices = [np.array(geom.exterior.coords)]
    else:
        list_vertices = [np.array(geom.geoms[i].exterior.coords) for i in range(len(geom.geoms))]
    # Split lines at the antimeridian
    list_vertices = [get_antimeridian_safe_polygon_vertices(vertices) for vertices in list_vertices]
    # Flat the list
    list_vertices = [item for sublist in list_vertices for item in sublist]
    # Return a MultiLineString object
    return MultiPolygon([Polygon(vertices) for vertices in list_vertices])


def st_line_antimeridian_safe(geom):
    """Sanitize shapely lines crossing the antimeridian.

    Given a Shapely LineString or MultiLineString representation of a Line,
    returns a MultiLineString of 'antimeridian-safe' constituent lines splitted at the anti-meridian.
    The returned MultiString ensure compliance with GeoJSON standards
    GeoJSON standards: https://tools.ietf.org/html/rfc7946#section-3.1.9

    Assumptions:
      - Any two consecutive points with over 180 degrees difference in
        longitude are assumed to cross the antimeridian.
      - The line can wrap multiple time across the globe and cross the antimeridian on multiple occasions.

    Returns:
        MultiLineString: antimeridian-safe line(s)
    """
    # Check geometry type
    if not isinstance(geom, (shapely.geometry.LineString, shapely.geometry.MultiLineString)):
        raise TypeError("Expects LineString or MultiLineString.")
    # Get list of vertices
    if isinstance(geom, shapely.geometry.LineString):
        list_vertices = [np.array(geom.coords)]
    else:
        list_vertices = [np.array(geom.geoms[i].coords) for i in range(len(geom.geoms))]
    # Split lines at the antimeridian
    list_vertices = [get_antimeridian_safe_line_vertices(vertices) for vertices in list_vertices]
    # Flat the list
    list_vertices = [item for sublist in list_vertices for item in sublist]
    # Return a MultiLineString object
    return MultiLineString(list_vertices)

# --------------------------------------------------------------------------.
