#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 - 2022 Pyresample developers
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
"""Spherical Geometry Classes for spherical computation à la Shapely sytle.

Base type is a numpy array of size (n, 2) (2 for lon and lats)
(lon,lat) units are expected as radians.

"""
import copy
import logging
import warnings

import numpy as np
import pyproj
from shapely.geometry import MultiLineString  # LineString
from shapely.geometry import MultiPolygon, Polygon

from pyresample.utils.shapely import (
    _check_valid_extent,
    _close_vertices,
    _get_extent_from_vertices,
    _unclose_vertices,
    bounds_from_extent,
    get_antimeridian_safe_line_vertices,
    get_idx_antimeridian_crossing,
    get_non_overlapping_list_extents,
    remove_duplicated_vertices,
)

logger = logging.getLogger(__name__)

EPSILON = 0.0000001

# ----------------------------------------------------------------------------.


def unwrap_longitude_degree(x, period=360):
    """Unwrap longitude array."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def unwrap_radians(x, period=2 * np.pi):
    """Unwrap radians array between -period/2 and period/2."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def _hdistance_matrix(vertices, ref_vertices):
    """Compute a distance matrix between vertices using haverstine formula.

    Vertices matrix must have shape (n,2) with (lon,lat) in radians !
    The returned distance matrix has shape n_vertices x n_ref_vertices)
    """
    lon = vertices[:, 0]
    lat = vertices[:, 1]
    ref_lat = ref_vertices[:, 1]
    diff_lon = lon[:, None] - ref_vertices[:, 0]
    diff_lat = lat[:, None] - ref_vertices[:, 1]
    d = np.sin(diff_lat / 2)**2 + np.cos(lat[:, None]) * np.cos(ref_lat) * np.sin(diff_lon / 2)**2
    return d


def get_list_extent_from_vertices(vertices, radians=False):
    """Get list of extent for SExtent creation."""
    vertices = _close_vertices(vertices)
    cross_indices = get_idx_antimeridian_crossing(vertices, radians=radians)
    if len(cross_indices) == 0:
        return [_get_extent_from_vertices(vertices)]
    else:
        if radians:
            vertices = np.rad2deg(vertices)
        # Hereafter, computations are done in degrees
        l_extent = []
        list_vertices = get_antimeridian_safe_line_vertices(vertices)
        list_vertices[0] = np.vstack((list_vertices[-1], list_vertices[0]))
        del list_vertices[len(list_vertices) - 1]
        l_extent = [_get_extent_from_vertices(v) for v in list_vertices]

        # TODO If North Pole enclosing add 90 (or pi/2) to y_max

        # TODO If South Pole enclosing, add -90 (or pi/2) to y_min

        # Union the list of extents in order to be non-overlapping
        if len(l_extent) > 1:
            l_extent = get_non_overlapping_list_extents(l_extent)

        # Backconversion to radians if asked
        if radians:
            l_extent = [np.deg2rad(ext) for ext in l_extent]

    return l_extent


def _polygon_contain_point(vertices, point):
    # TODO [HELP NEEDED !!!]
    # - Should not use/call a SPolygon method/class !
    # - To be used in get_list_extent_from_vertices
    # - To be called within a SPolygon.contains_point method
    return None


def _check_extent_topology_validity(list_extent):
    p = MultiPolygon([Polygon.from_bounds(*bounds_from_extent(ext)) for ext in list_extent])
    try:
        p.intersects(p)
    except Exception:  # TODO: specify Error !!! Shapely TopologicalError
        raise ValueError("The Sextent is not valid. The composing extents must not overlap each other.")


def get_list_connected_pairs(pairs):
    """Return a list of connected element pairs.

    Given a list of pairs, it returns a list of connected pairs.
    Example: [(1,0), (0,2), (4,5), (3,1)] --> [[1,0,2,3], [4,5]]

    Taken from:
    https://stackoverflow.com/questions/28980797/given-n-tuples-representing-pairs-return-a-list-with-connected-tuples
    """
    lists_by_element = {}

    def make_new_list_for(x, y):
        lists_by_element[x] = lists_by_element[y] = [x, y]

    def add_element_to_list(lst, el):
        lst.append(el)
        lists_by_element[el] = lst

    def merge_lists(lst1, lst2):
        merged_list = lst1 + lst2
        for el in merged_list:
            lists_by_element[el] = merged_list

    for x, y in pairs:
        xList = lists_by_element.get(x)
        yList = lists_by_element.get(y)

        if not xList and not yList:
            make_new_list_for(x, y)

        if xList and not yList:
            add_element_to_list(xList, y)

        if yList and not xList:
            add_element_to_list(yList, x)

        if xList and yList and xList != yList:
            merge_lists(xList, yList)

    # return the unique lists present in the dictionary
    return list(set(tuple(el) for el in lists_by_element.values()))


def _cascade_polygon_union(list_polygons):
    """Union list of SPolygon(s) togethers."""
    import itertools

    # Retrieve possible polygon intersections
    n_polygons = len(list_polygons)
    list_combo_pairs = list(itertools.combinations(range(n_polygons), 2))
    idx_combo_pairs_inter = np.where([list_polygons[i].intersects(list_polygons[j]) for i, j in list_combo_pairs])[0]

    # If no intersection, returns all polygons
    if len(idx_combo_pairs_inter) == 0:
        return SMultiPolygon(list_polygons)

    # Otherwise, union intersecting polygons
    list_combo_pairs_inter = [list_combo_pairs[i] for i in idx_combo_pairs_inter]
    list_inter_indices = get_list_connected_pairs(list_combo_pairs_inter)

    list_union = []
    for inter_indices in list_inter_indices:
        union_p = list_polygons[inter_indices[0]]
        for i in inter_indices[1:]:
            union_p = list_polygons[i].union(union_p)  # This return always a SPolygon
        list_union.append(union_p)

    # Identify non-intersecting polygons
    idx_poly_inter = np.unique(list_combo_pairs_inter)
    idx_poly_non_inter = list(set(range(n_polygons)).difference(set(idx_poly_inter)))

    # Add non-intersecting polygons
    if len(idx_poly_non_inter) > 0:
        _ = [list_union.append(list_polygons[i]) for i in idx_poly_non_inter]

    # Return unioned polygon
    return SMultiPolygon(list_union)


class SExtent(object):
    """Spherical Extent.

    SExtent longitudes are defined between -180 and 180 degree.
    A spherical geometry crossing the anti-meridian will have an SExtent
     composed of [..., 180, ...,...] and [-180, ..., ..., ...]

    Intersection between SExtents does not include touching extents !
    The extents composing an SExtent can not intersect/overlap each other.
    There is not an upper limit on the number of extents composing SExtent.
     They just need to not overlap.

    """

    def __init__(self, *args):
        list_extent = list(list(locals().values())[1:][0])
        self.list_extent = list_extent
        if len(list_extent) == 0:
            raise ValueError("No argument passed to SExtent.")
        if isinstance(list_extent[0], (int, float, str)):
            raise TypeError("You need to pass [lon_min, lon_max, lat_min, lat_max] list(s) to SExtent.")
        if list_extent[0] is None:
            raise ValueError("SExtent does not accept None as input argument.")
        if len(list_extent[0]) == 0:
            raise ValueError("An empty extent passed to SExtent.")

        # Check valid data range
        list_extent = [_check_valid_extent(ext) for ext in list_extent]

        # Check extents does not overlaps
        _check_extent_topology_validity(self.list_extent)

        # If wraps around the earth across all longitudes, specify a unique -180, 180 extent
        # if np.any([(ext[0] == -180) & (ext[1] == 180) for ext in list_extent]):
        #     lat_min = min([ext[2] for ext in list_extent])
        #     lat_max = max([ext[3] for ext in list_extent])
        #     list_extent = [[-180, 180, lat_min, lat_max]]

        self.list_extent = list_extent

    def __str__(self):
        """Get simplified representation of SExtent."""
        return str(self.list_extent)

    def __repr__(self):
        """Get simplified representation of SExtent."""
        return str(self.list_extent)

    def __iter__(self):
        """Get list_extent iterator."""
        return self.list_extent.__iter__()

    @property
    def is_global(self):
        """Check if the extent is global."""
        if len(self.list_extent) != 1:
            return False
        if self.list_extent[0] == [-180, 180, -90, 90]:
            return True
        else:
            return False

    def intersects(self, other):
        """Check if SExtent is intersecting the other SExtent.

        Touching extent are considered to not intersect !
        """
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.intersects() expects a SExtent class instance.")
        bl = (self.to_shapely().intersects(other.to_shapely()) and
              not self.to_shapely().touches(other.to_shapely()))
        return bl

    def disjoint(self, other):
        """Check if SExtent does not intersect (and do not touch) the other SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.intersects() expects a SExtent class instance.")
        return self.to_shapely().disjoint(other.to_shapely())

    def within(self, other):
        """Check if SExtent is within another SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.within() expects a SExtent class instance.")
        return self.to_shapely().within(other.to_shapely())

    def contains(self, other):
        """Check if the SExtent contains another SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.contains() expects a SExtent class instance.")
        return self.to_shapely().contains(other.to_shapely())

    def touches(self, other):
        """Check if SExtent external touches another SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.touches() expects a SExtent class instance.")
        return self.to_shapely().touches(other.to_shapely())

    def equals(self, other):
        """Check if SExtent is equals to SExtent."""
        if not isinstance(other, SExtent):
            raise TypeError("SExtent.equals() expects a SExtent class instance.")
        return self.to_shapely().equals(other.to_shapely())

    def to_shapely(self):
        """Return the shapely extent rectangle(s) polygon(s)."""
        return MultiPolygon([Polygon.from_bounds(*bounds_from_extent(ext)) for ext in self.list_extent])

    def plot(self, ax=None, facecolor='orange', edgecolor='black', alpha=0.4, **kwargs):
        """Plot the SLine using Cartopy."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # Retrieve shapely polygon
        geom = self.to_shapely()

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Add extent polygon
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
                          **kwargs)
        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
        return ax


class SPoint(object):
    """Spherical coordinates.

    The ``lon`` and ``lat`` coordinates should be provided in radians.
    """

    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    @property
    def vertices(self):
        """Return SPoint vertices in a ndarray of shape [1,2]."""
        return np.array([self.lon, self.lat])[None, :]

    def cross2cart(self, point):
        """Compute the cross product, and convert to cartesian coordinates."""
        lat1 = self.lat
        lon1 = self.lon
        lat2 = point.lat
        lon2 = point.lon

        ad = np.sin(lat1 - lat2) * np.cos((lon1 - lon2) / 2.0)
        be = np.sin(lat1 + lat2) * np.sin((lon1 - lon2) / 2.0)
        c = np.sin((lon1 + lon2) / 2.0)
        f = np.cos((lon1 + lon2) / 2.0)
        g = np.cos(lat1)
        h = np.cos(lat2)
        i = np.sin(lon2 - lon1)
        res = CCoordinate(np.array([-ad * c + be * f,
                                    ad * f + be * c,
                                    g * h * i]))

        return res

    def to_cart(self):
        """Convert to cartesian."""
        return CCoordinate(np.array([np.cos(self.lat) * np.cos(self.lon),
                                     np.cos(self.lat) * np.sin(self.lon),
                                     np.sin(self.lat)]))

    def distance(self, point):
        """Get distance using Vincenty formula.

        The result must be multiplied by Earth radius to obtain distance in m or km.
        """
        dlambda = self.lon - point.lon
        num = ((np.cos(point.lat) * np.sin(dlambda)) ** 2 +
               (np.cos(self.lat) * np.sin(point.lat) -
                np.sin(self.lat) * np.cos(point.lat) *
                np.cos(dlambda)) ** 2)
        den = (np.sin(self.lat) * np.sin(point.lat) +
               np.cos(self.lat) * np.cos(point.lat) * np.cos(dlambda))

        return np.arctan2(num ** .5, den)

    def hdistance(self, point):
        """Get distance using Haversine formula.

        The result must be multiplied by Earth radius to obtain distance in m or km.
        """
        return 2 * np.arcsin((np.sin((point.lat - self.lat) / 2.0) ** 2.0 +
                              np.cos(point.lat) * np.cos(self.lat) *
                              np.sin((point.lon - self.lon) / 2.0) ** 2.0) ** .5)

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)

    def __eq__(self, other):
        """Check equality."""
        return np.allclose((self.lon, self.lat), (other.lon, other.lat))

    def __str__(self):
        """Get simplified representation of lon/lat arrays in degrees."""
        return str((np.rad2deg(self.lon), np.rad2deg(self.lat)))

    def __repr__(self):
        """Get simplified representation of lon/lat arrays in degrees."""
        return str((np.rad2deg(self.lon), np.rad2deg(self.lat)))

    def __iter__(self):
        """Get iterator over lon/lat pairs."""
        return zip([self.lon, self.lat]).__iter__()

    def plot(self, ax=None, color='blue', alpha=1, **kwargs):
        """Plot the SPoint using Cartopy."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Plot Points
        ax.scatter(x=np.rad2deg(self.vertices[:, 0]),
                   y=np.rad2deg(self.vertices[:, 1]),
                   color=color,
                   alpha=alpha, **kwargs)

        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False

        return ax


class CCoordinate(object):
    """Cartesian coordinates."""

    def __init__(self, cart):
        self.cart = np.array(cart)

    def norm(self):
        """Get Euclidean norm of the vector."""
        return np.sqrt(np.einsum('...i, ...i', self.cart, self.cart))

    def normalize(self):
        """Normalize the vector."""
        self.cart /= np.sqrt(np.einsum('...i, ...i', self.cart, self.cart))

        return self

    def cross(self, point):
        """Get cross product with another vector."""
        return CCoordinate(np.cross(self.cart, point.cart))

    def dot(self, point):
        """Get dot product with another vector."""
        return np.inner(self.cart, point.cart)

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)

    def __eq__(self, other):
        """Check equality."""
        return np.allclose(self.cart, other.cart)

    def __str__(self):
        """Get simplified representation."""
        return str(self.cart)

    def __repr__(self):
        """Get simplified representation."""
        return str(self.cart)

    def __add__(self, other):
        """Add."""
        try:
            return CCoordinate(self.cart + other.cart)
        except AttributeError:
            return CCoordinate(self.cart + np.array(other))

    def __radd__(self, other):
        """Add."""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply."""
        try:
            return CCoordinate(self.cart * other.cart)
        except AttributeError:
            return CCoordinate(self.cart * np.array(other))

    def __rmul__(self, other):
        """Multiply."""
        return self.__mul__(other)

    def to_spherical(self):
        """Convert to Spherical coordinate object."""
        return SPoint(np.arctan2(self.cart[1], self.cart[0]),
                      np.arcsin(self.cart[2]))


class SArc(object):
    """An arc of the great circle between two points.

    A GreatCircle Arc is defined as the shortest tracks between two points.
    An arc is defined as the shortest tracks between two points.
    """

    def __init__(self, start, end):
        self.start, self.end = start, end

    def __hash__(self):
        """Define SArc hash to enable LRU caching of arc.intersection."""
        return hash((self.start.lon, self.start.lat, self.end.lon, self.end.lat))

    def __eq__(self, other):
        """Check equality."""
        if self.start == other.start and self.end == other.end:
            return True
        return False

    def __ne__(self, other):
        """Check not equal comparison."""
        return not self.__eq__(other)

    def __str__(self):
        """Get simplified representation in lat/lon degrees."""
        return str(self.start) + " -> " + str(self.end)

    def __repr__(self):
        """Get simplified representation in lat/lon degrees."""
        return str(self.start) + " -> " + str(self.end)

    @property
    def vertices(self):
        """Get start SPoint and end SPoint vertices array."""
        return self.start.vertices, self.end.vertices

    def to_line(self):
        """Convert to SLine."""
        vertices = np.vstack(self.vertices)
        return SLine(vertices)

    def to_shapely(self):
        """Convert to Shapely MultiLineString."""
        return self.to_line().to_shapely()

    def plot(self, ax=None, edgecolor='black', alpha=0.4, **kwargs):
        """Plot the SArc using Cartopy."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # Retrieve shapely polygon
        geom = self.to_shapely()

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Plot polygon
        ax.add_geometries([geom], crs=ccrs.Geodetic(),
                          facecolor='none',
                          edgecolor=edgecolor, alpha=alpha,
                          **kwargs)

        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()  # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75)
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
        return ax

    def angle(self, other_arc):
        """Get oriented angle between two consecutive arcs.

        Returns:
            Angle in radians. A straight line will be 0. A clockwise path
            will be a negative angle and counter-clockwise will be positive.
        """
        if self.start == other_arc.start:
            a__ = self.start
            b__ = self.end
            c__ = other_arc.end
        elif self.start == other_arc.end:
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
        elif self.end == other_arc.end:
            a__ = self.end
            b__ = self.start
            c__ = other_arc.start
        elif self.end == other_arc.start:
            a__ = self.end
            b__ = self.start
            c__ = other_arc.end
        else:
            raise ValueError("No common point in angle computation.")

        ua_ = a__.cross2cart(b__)
        ub_ = a__.cross2cart(c__)

        val = ua_.dot(ub_) / (ua_.norm() * ub_.norm())
        if abs(val - 1) < EPSILON:
            angle = 0
        elif abs(val + 1) < EPSILON:
            angle = np.pi
        else:
            angle = np.arccos(val)

        n__ = ua_.normalize()
        if n__.dot(c__.to_cart()) > 0:
            return -angle
        else:
            return angle

    def _great_circle_intersections(self, other_arc):
        """Give the two intersections of the greats circles defined by the current arc and *other_arc*.

        From http://williams.best.vwh.net/intersect.htm
        """
        if self.end.lon - self.start.lon > np.pi:
            self = copy.deepcopy(self)
            self.end.lon -= 2 * np.pi
        if other_arc.end.lon - other_arc.start.lon > np.pi:
            other_arc = copy.deepcopy(other_arc)
            other_arc.end.lon -= 2 * np.pi
        if self.end.lon - self.start.lon < -np.pi:
            self = copy.deepcopy(self)
            self.end.lon += 2 * np.pi
        if other_arc.end.lon - other_arc.start.lon < -np.pi:
            other_arc = copy.deepcopy(other_arc)
            other_arc.end.lon += 2 * np.pi

        ea_ = self.start.cross2cart(self.end).normalize()
        eb_ = other_arc.start.cross2cart(other_arc.end).normalize()

        cross = ea_.cross(eb_)
        lat = np.arctan2(cross.cart[2],
                         np.sqrt(cross.cart[0] ** 2 + cross.cart[1] ** 2))
        lon = np.arctan2(cross.cart[1], cross.cart[0])

        return (SPoint(lon, lat),
                SPoint(unwrap_radians(lon + np.pi), -lat))

    def intersects(self, other_arc):
        """Check if the current arc and the *other_arc* intersect.

        An arc is defined as the shortest tracks between two points.
        """
        return bool(self.intersection(other_arc))

    def intersection_point(self, other_arc):
        """Compute the intersection point between two arcs.

        If arc and *other_arc* intersect, it returns the intersection SPoint.
        If arc and *other_arc* does not intersect, it returns None.
        """
        if self == other_arc:
            return None
        great_circles_intersection_spoints = self._great_circle_intersections(other_arc)

        for spoint in great_circles_intersection_spoints:
            a = self.start
            b = self.end
            c = other_arc.start
            d = other_arc.end

            ab_dist = a.hdistance(b)
            cd_dist = c.hdistance(d)
            ap_dist = a.hdistance(spoint)
            bp_dist = b.hdistance(spoint)
            cp_dist = c.hdistance(spoint)
            dp_dist = d.hdistance(spoint)

            if (((spoint in (a, b)) or (abs(ap_dist + bp_dist - ab_dist) < EPSILON)) and
                    ((spoint in (c, d)) or (abs(cp_dist + dp_dist - cd_dist) < EPSILON))):
                return spoint

        return None

    def intersection(self, other_arc):
        """Give the two intersections of the greats circles defined by the current arc and *other_arc*.

        Use _great_circle_intersections instead.
        """
        warnings.warn("'SArc.intersection' is deprecated, use 'intersection_point' instead.",
                      PendingDeprecationWarning)
        return self.intersection_point(other_arc)

    def intersections(self, other_arc):
        """Compute the intersection point between two arcs.

        Use intersection_point instead.
        """
        warnings.warn("'SArc.intersections' is deprecated, use '_great_circle_intersections' instead.",
                      PendingDeprecationWarning)
        return self._great_circle_intersections(other_arc)

    def get_next_intersection(self, arcs, known_intersection_spoint=None):
        """Get the next intersection between the current arc and *arcs*.

        It return a tuple with the intersecting point and the arc within *arcs*
        that intersect the self arc.
        """
        list_intersection_spoint_arc = []
        for arc in arcs:
            spoint = self.intersection_point(arc)
            if (spoint is not None and spoint != arc.end and spoint != self.end):
                list_intersection_spoint_arc.append((spoint, arc))

        def dist(args):
            """Get distance key."""
            return self.start.distance(args[0])

        take_next = False
        for spoint, arc in sorted(list_intersection_spoint_arc, key=dist):
            if known_intersection_spoint is not None:
                if known_intersection_spoint == spoint:
                    take_next = True
                elif take_next:
                    return spoint, arc
            else:
                return spoint, arc

        return None, None

    def midpoint(self, ellips='WGS84'):
        """Return the SArc midpoint coordinate."""
        geod = pyproj.Geod(ellps=ellips)
        lon_start = self.start.lon
        lon_end = self.end.lon
        lat_start = self.start.lat
        lat_end = self.end.lat
        lon_mid, lat_mid = geod.npts(lon_start, lat_start, lon_end, lat_end, npts=1, radians=True)[0]
        return SPoint(lon_mid, lat_mid)

    def segmentize(self, npts=0, max_distance=0, ellips='WGS84'):
        """Segmentize the spherical SArc.

        It returns an SLine.

        npts or max_distance are mutually exclusively. Specify one of them.
        max_distance must be provided in kilometers.
        """
        if npts != 0:
            npts = npts + 2  # + 2 to account for initial and terminus

        geod = pyproj.Geod(ellps=ellips)
        lon_start = self.start.lon
        lon_end = self.end.lon
        lat_start = self.start.lat
        lat_end = self.end.lat

        points = geod.inv_intermediate(lon_start, lat_start, lon_end, lat_end,
                                       del_s=max_distance,
                                       npts=npts,
                                       radians=True,
                                       initial_idx=0, terminus_idx=0)
        lons, lats = (points.lons, points.lats)
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        vertices = np.stack((lons, lats)).T
        return SLine(vertices)


class SLine(object):
    """A spherical line composed of great circle arcs."""

    def __init__(self, vertices):
        """Initialise SLine object.

        Parameters
        ----------
        vertices : np.ndarray
            Array of shape ``[N, 2]`` with ``N`` points describing a line.
            The first column describes longitudes, the second the latitudes.
            Units should be in radians.
        """
        # Check vertices shape is correct
        if not isinstance(vertices, np.ndarray):
            raise TypeError("SLine expects a numpy ndarray.")
        vertices = remove_duplicated_vertices(vertices)
        if len(vertices.shape) != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 2:
            raise ValueError("SLine expects a numpy ndarray with shape (>=2, 2).")

        # Ensure vertices precision
        vertices = vertices.astype(np.float64, copy=False)

        # Define SLine composing arcs
        lon = vertices[:, 0]
        lat = vertices[:, 1]
        list_arcs = []
        for i in range(len(lon) - 1):
            list_arcs.append(SArc(SPoint(lon[i], lat[i]),
                                  SPoint(lon[i + 1], lat[i + 1])))

        self.vertices = vertices
        self.lon = lon
        self.lat = lat
        self._list_arcs = list_arcs

    def __getitem__(self, i):
        """Subset SLine.

        If an integer is provided, it return the corresponding SArc.
        If using slices, it return the SLine subset.
        """
        if isinstance(i, int):
            return self._list_arcs[i]
        if isinstance(i, slice):  # i.e. sline[1:5]
            n_indices = i.stop - i.start
            if n_indices < 2:
                raise ValueError("Keep at least 2 SArc when subsetting SLine.")
            import copy
            i = slice(i.start, i.stop + 1, i.step)
            self = copy.copy(self)
            self.vertices = self.vertices[i, :]
            self.lon = self.lon[i]
            self.lat = self.lat[i]
            self._list_arcs = self._list_arcs[i]
            return self
        else:
            raise TypeError("Either subset the SLine with an integer "
                            "(to get an SArc) or with a slice to retrieve "
                            "a subsetted SLine.")

    def __len__(self):
        """Return the number of SArc(s) composing SLine."""
        return len(self._list_arcs)

    def __iter__(self):
        """Return iterator of SLine composing SArc(s)."""
        return self._list_arcs.__iter__()

    def get_parallel_lines(self, distance=0, ellips="WGS84"):
        """Return 2 SLine(s) at a given distance parallel to the current SLine.

        The distance should be provided in km.
        """
        if distance <= 0:
            raise ValueError("SLine.get_parallel_lines expects a "
                             "distance in km larger than 0.")
        geod = pyproj.Geod(ellps=ellips)

        lon_start = self.vertices[0:-1, 0]
        lat_start = self.vertices[0:-1, 1]
        lon_end = self.vertices[1:, 0]
        lat_end = self.vertices[1:, 1]

        # Retrieve forward azimuths
        az12, _, _ = geod.inv(lon_start, lat_start,
                              lon_end, lat_end,
                              radians=True)

        # Define orthogonal azimuth
        az12_top = az12 - np.pi / 2
        az12_bottom = az12 + np.pi / 2

        # Include last vertex to maintain the same number of vertices
        lon_start = np.append(lon_start, self.vertices[-1, 0])
        lat_start = np.append(lat_start, self.vertices[-1, 1])
        az12_top = np.append(az12_top, az12_top[-1])
        az12_bottom = np.append(az12_bottom, az12_bottom[-1])

        # Retrieve the point coordinate at dist distance and forward azimuth
        dist = np.ones(az12_top.shape) * distance * 1000  # convert km to m
        lon_top, lat_top, _ = geod.fwd(lon_start, lat_start,
                                       az12_top,
                                       dist=dist,
                                       radians=True)
        lon_bottom, lat_bottom, _ = geod.fwd(lon_start, lat_start,
                                             az12_bottom,
                                             dist=dist,
                                             radians=True)
        # Assemble line vertices
        top_vertices = np.stack((lon_top, lat_top)).T
        bottom_vertices = np.stack((lon_bottom, lat_bottom)).T
        return (SLine(top_vertices), SLine(bottom_vertices))

    def buffer(self, distance, ellips="WGS84"):
        """Return a SPolygon buffering on both sides of SLine.

        The distance must be provided in kilometers.
        """
        left_sline, right_sline = self.get_parallel_lines(distance=distance,
                                                          ellips=ellips)
        # Assemble polygon vertices
        polygon_vertices = np.vstack((left_sline.vertices,
                                      right_sline.vertices[::-1, :]))
        polygon_vertices = np.vstack((polygon_vertices,
                                      polygon_vertices[0, :]))
        return SPolygon(polygon_vertices)

    def segmentize(self, npts=0, max_distance=0, ellips='WGS84'):
        """Subdivide each SArc of SLine in n steps."""
        list_vertices = [arc.segmentize(npts=npts,
                                        max_distance=max_distance,
                                        ellips=ellips).vertices[:-1, :] for arc in self._list_arcs[0:-1]]
        list_vertices.append(self._list_arcs[-1].segmentize(npts=npts,
                                                            max_distance=max_distance,
                                                            ellips=ellips).vertices)
        vertices = np.vstack(list_vertices)
        return SLine(vertices)

    def intersects(self, other):
        """Return True if two SLine intersects each other."""
        if not isinstance(other, SLine):
            raise TypeError("SLine.intersects expects an SLine.")
        for arc1 in self._list_arcs:
            for arc2 in other:
                spoint = arc1.intersection_point(arc2)
                if spoint is not None:
                    return True
        return False

    def intersection_points(self, other):
        """Return the intersection points between two SLines."""
        if not isinstance(other, SLine):
            raise TypeError("SLine.intersection_points expects an SLine.")
        list_spoints = []
        for arc1 in self._list_arcs:
            for arc2 in other:
                spoint = arc1.intersection_point(arc2)
                if spoint is not None:
                    list_spoints.append(spoint)
        if len(list_spoints) == 0:
            list_spoints = None
        return list_spoints

    def to_shapely(self):
        """Convert to Shapely MultiLineString."""
        list_vertices = get_antimeridian_safe_line_vertices(np.rad2deg(self.vertices))
        return MultiLineString(list_vertices)

    def _repr_svg_(self):
        """Display SLine in the Ipython terminal."""
        return self.to_shapely()._repr_svg_()

    def plot(self, ax=None, edgecolor='black', alpha=0.4, **kwargs):
        """Plot the SLine using Cartopy."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # Retrieve shapely polygon
        geom = self.to_shapely()

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Plot polygon
        ax.add_geometries([geom], crs=ccrs.Geodetic(),
                          facecolor='none',
                          edgecolor=edgecolor, alpha=alpha,
                          **kwargs)

        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
        return ax


class SPolygon:
    """Spherical Polygon.

    A Spherical Polygon is a collection of points, each of which
    are connected by edges which consist of arcs of great circles.
    The points are ordered clockwise, such that if you walk along
    the surface of the earth along the polygon's edges, the
    'inside' of the polygon will lie to your right.

    Spherical Polygon is initialised with a polygon vertices
    ndarray of shape ``[N, 2]``.
    The first column must contains longitudes, the second column
    must contains latitudes.
    The units must be in radians.
    The inside of the polygon is defined by the vertices
    being defined clockwise around it.

    No check is performed to assess vertices order !
    The Polygon is valid only if is not self-intersecting with itself.

    If the last vertex is equal to the first, the last vertex
    is removed automatically.
    """

    def __init__(self, vertices, check_validity=False):
        """Initialise SphPolygon object.

        Args:
            vertices (np.ndarray): ndarray of shape ``[N, 2]`` with ``N``
                points describing a polygon clockwise.  First column
                describes longitudes, second column describes latitudes.  Units
                should be in radians.
        """
        # Check vertices shape is correct
        if not isinstance(vertices, np.ndarray):
            raise TypeError("SPolygon expects a numpy ndarray.")
        if len(vertices.shape) != 2 or vertices.shape[1] != 2:
            raise ValueError("SPolygon expects an array of shape (n, 2).")
        if vertices.shape[0] < 3:
            raise ValueError("SPolygon expects an array of shape (>=3, 2).")

        # Check last vertices is not equal to first
        vertices = _unclose_vertices(vertices)

        # Remove duplicated vertices
        vertices = remove_duplicated_vertices(vertices, tol=1e-08)

        # Check it has at least 3 vertices
        if vertices.shape[0] < 3:
            raise ValueError("This is likely an error caused by current tol "
                             "argument in remove_duplicated_vertices. "
                             "Please report issue.")
        # Define SExtent
        self.extent = SExtent(*get_list_extent_from_vertices(np.rad2deg(vertices),
                                                             radians=False))

        # Define vertices
        vertices = vertices.astype(np.float64, copy=False)  # important !
        self.vertices = vertices
        self.lon = self.vertices[:, 0]
        self.lat = self.vertices[:, 1]

        # Add geoms for compatibility with SMultiPolygon
        self.geoms = [self]

        # Check for not self-intersection (valid geometry)
        # - This function is quite slow
        if check_validity:
            if not self.is_valid_geometry:
                raise ValueError("The provided vertices cause an invalid "
                                 "self-intersecting polygon.")

    @property
    def is_valid_geometry(self):
        """Check that there are not self intersections."""
        arcs = [edge for edge in self.aedges()]
        list_edges_intersection_spoints = self._get_all_arc_intersection_points(arcs, arcs)
        if len(list_edges_intersection_spoints) == 0:
            return True
        else:
            return False

    def __getitem__(self, i):
        """Get the SPolygon.

        Defined for compatibility with SMultiPolygon. The only valid i value is 0.
        """
        return self.geoms[i]

    def __len__(self):
        """Get the number of SPolygon.

        Defined for compatibility with SMultiPolygon. It returns 1.""
        """
        return len(self.geoms)

    def __iter__(self):
        """Get an iterator returning the SPolygon.

        Defined for compatibility with SMultiPolygon. It returns SPolygon.""
        """
        return self.geoms.__iter__()

    def __str__(self):
        """Get numpy representation of vertices."""
        return str(np.rad2deg(self.vertices))

    def invert(self):
        """Invert the polygon."""
        self.vertices = np.flipud(self.vertices)
        self.lon = self.vertices[:, 0]
        self.lat = self.vertices[:, 1]

    def inverse(self):
        """Return an inverse of the polygon."""
        return SPolygon(np.flipud(self.vertices))

    def aedges(self):
        """Get generator over the edges, in arcs of Coordinates."""
        for (lon_start, lat_start), (lon_stop, lat_stop) in self.edges():
            yield SArc(SPoint(lon_start, lat_start),
                       SPoint(lon_stop, lat_stop))

    def edges(self):
        """Get generator over the edges, in geographical coordinates."""
        for i in range(len(self.lon) - 1):
            yield (self.lon[i], self.lat[i]), (self.lon[i + 1], self.lat[i + 1])
        yield (self.lon[i + 1], self.lat[i + 1]), (self.lon[0], self.lat[0])

    def _area(self):
        """Find the area of the polygon in units of square radii.

        Note that the earth is not exactly spherical.
        The equatorial radius is 6378 km, while the polar radius is 6357 km.
        For more accurate area computations, use the area() function.

        The inside of the polygon is defined by having the vertices enumerated
        clockwise around it.
        A polygon containing the entire planet would have area 4π.

        Uses the algorithm described in [bev1987].
        .. Michael Bevis and Greg Cambareri,
           "Computing the area of a spherical polygon of arbitrary shape",
           in *Mathematical Geology*, May 1987, Volume 19, Issue 4, pp 335-346.

        Note: The article mixes up longitudes and latitudes in equation 3!
        Look at the fortran code appendix for the correct version.
        The units are the square of the radius passed to the constructor.

        For  example, to calculate the area in km² of a polygon near the equator of a
        spherical planet with a radius of 6371 km (similar to Earth):

        >>> radius = 6371 # [km]
        >>> pol = SPolygon(np.deg2rad(np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])))
        >>> print(pol.area()*radius**2 # [km²]
        12363.997753690213
        """
        phi_a = self.lat
        phi_p = self.lat.take(np.arange(len(self.lat)) + 1, mode="wrap")
        phi_b = self.lat.take(np.arange(len(self.lat)) + 2, mode="wrap")
        lam_a = self.lon
        lam_p = self.lon.take(np.arange(len(self.lon)) + 1, mode="wrap")
        lam_b = self.lon.take(np.arange(len(self.lon)) + 2, mode="wrap")

        new_lons_a = np.arctan2(np.sin(lam_a - lam_p) * np.cos(phi_a),
                                np.sin(phi_a) * np.cos(phi_p) -
                                np.cos(phi_a) * np.sin(phi_p) *
                                np.cos(lam_a - lam_p))

        new_lons_b = np.arctan2(np.sin(lam_b - lam_p) * np.cos(phi_b),
                                np.sin(phi_b) * np.cos(phi_p) -
                                np.cos(phi_b) * np.sin(phi_p) *
                                np.cos(lam_b - lam_p))

        alpha = new_lons_a - new_lons_b
        alpha[alpha < 0] += 2 * np.pi

        return (sum(alpha) - (len(self.lon) - 2) * np.pi)

    def area(self, ellps="WGS84"):
        """Calculate the polygon area [in km²] using pyproj.

        ellps argument enable to specify custom ellipsoid for accurate area calculation.
        """
        geod = pyproj.Geod(ellps=ellps)  # "sphere", maybe be more flexibile?
        pyproj_area, _ = geod.polygon_area_perimeter(self.vertices[::-1, 0],
                                                     self.vertices[::-1, 1],
                                                     radians=True)
        return pyproj_area

    def perimeter(self, ellps="WGS84"):
        """Calculate the polygon perimeter [in km] using pyproj.

        ellps argument enable to specify custom ellipsoid for accurate perimeter calculation.
        """
        geod = pyproj.Geod(ellps=ellps)  # "sphere", maybe be more flexibile?
        _, pyproj_perimeter = geod.polygon_area_perimeter(self.vertices[::-1, 0],
                                                          self.vertices[::-1, 1],
                                                          radians=True)
        return pyproj_perimeter

    def equals_exact(self, other, rtol=1e-05, atol=1e-08):
        """Check if self SPolygon is exactly equal to the other SPolygon.

        This method uses exact coordinate equality, which requires coordinates
        to be equal (within specified tolerance) and in the same order.
        This is in contrast with the ``equals`` function which checks for spatial
        (topological) equality.
        """
        if not isinstance(other, (SPolygon, SMultiPolygon, type(None))):
            raise NotImplementedError
        if isinstance(other, SMultiPolygon) or isinstance(other, type(None)):
            return False
        if (self.vertices.shape == other.vertices.shape and
                np.allclose(self.vertices, other.vertices, rtol=rtol, atol=atol)):
            return True
        else:
            return False

    def _is_inside(self, other):
        """Check if the polygon is entirely inside (not touching) the other.

        This method must be used only if no intersection between self and other !
        """
        # Check first if extent is within
        if not self.extent.within(other.extent):
            return False

        # --------------------------------------------------------------------.
        # Define antipodal points of first two vertices
        anti_lon_0 = self.lon[0] + np.pi
        if anti_lon_0 > np.pi:
            anti_lon_0 -= np.pi * 2

        anti_lon_1 = self.lon[1] + np.pi
        if anti_lon_1 > np.pi:
            anti_lon_1 -= np.pi * 2

        anti_lat_0 = -self.lat[0]
        anti_lat_1 = -self.lat[1]

        # --------------------------------------------------------------------.
        # Define ????
        arc1 = SArc(SPoint(self.lon[1],
                           self.lat[1]),
                    SPoint(anti_lon_0,
                           anti_lat_0))

        arc2 = SArc(SPoint(anti_lon_0,
                           anti_lat_0),
                    SPoint(anti_lon_1,
                           anti_lat_1))

        arc3 = SArc(SPoint(anti_lon_1,
                           anti_lat_1),
                    SPoint(self.lon[0],
                           self.lat[0]))

        # --------------------------------------------------------------------.
        # Do what ???
        other_arcs = [edge for edge in other.aedges()]
        for arc in [arc1, arc2, arc3]:
            inter, other_arc = arc.get_next_intersection(other_arcs)
            if inter is not None:
                sarc = SArc(arc.start, inter)
                earc = SArc(inter, other_arc.end)
                return sarc.angle(earc) < 0
        return other._area() > (2 * np.pi)

    @staticmethod
    def _get_all_arc_intersection_points(arcs1, arcs2):
        """Get the list of SArc(s) at intersection points.

        It returns a list of format [(SPoint, SArc1, SArc2), ...]
        """
        list_edges_intersection_spoints = []
        for arc1 in arcs1:
            for arc2 in arcs2:
                spoint = arc1.intersection_point(arc2)
                if (spoint is not None and spoint != arc1.end and spoint != arc2.end):
                    list_edges_intersection_spoints.append((spoint, arc1, arc2))
        return list_edges_intersection_spoints

    def _bool_oper(self, other, sign=1):
        """Perform a boolean operation on this and *other* polygons.

        By default, or when sign is 1, the union is performed.
        If sign is -1, the intersection of the polygons is returned.

        The algorithm works this way:
            1. Find an intersection between the two polygons.
            2. If none can be found, then the two polygons are either not
               overlapping, or one is entirely included in the other.
            3. Otherwise, follow the edges of a polygon until another
               intersection is encountered, at which point you start following
               the edges of the other polygon, and so on until you come
               back to the first intersection.

        In which direction to follow the edges of the polygons depends if
        being interested in the union or the intersection of the two polygons.

        The routine can deal with multiple intersections, but can return
        unexpected results when the union is performed between 2 SPolygons
        that have more than a single intersection.
        """
        # Iterate over other SMultiPolygon
        if isinstance(other, SMultiPolygon):
            if sign == -1:    # intersection
                return SMultiPolygon([self.intersection(p) for p in other])
            else:  # union
                # First identify which polygons intersects
                list_polygons = self.geoms + other.geoms
                return _cascade_polygon_union(list_polygons)

        # Logical speed ups
        if isinstance(other, type(None)):
            return None

        # If wishing the intersection and the extents do not intersects
        if sign == -1 and not self.extent.intersects(other.extent):
            return None

        # If vertices are equal, return one polygon
        if self.equals_exact(other):
            return self

        # --------------------------------------------------------------------.
        # Get list of SArc(s) for each polygon
        arcs1 = [edge for edge in self.aedges()]
        arcs2 = [edge for edge in other.aedges()]

        # --------------------------------------------------------------------.
        # Get list of intersection points and asssociated intersecting arcs
        # - If arc1==arc2 (same direction) --> No intersection point !
        # - It excludes intersection points which are equal to both the end
        #   coordinate of the arcs.
        # - It includes intersection points if arc1 and arc2 are
        #   touching/overlapping but have different directions.
        # TODO: This can still cause problems in the edge case where
        #       two polygons are just touching !!!
        #       The output of the intersection could result in a line ...
        #       The occurence of this is limited by the use the
        #       self.extent.intersects(other.extent) check above
        list_edges_intersection_spoints = self._get_all_arc_intersection_points(arcs1, arcs2)

        # --------------------------------------------------------------------.
        # If no intersection points:
        # - Find out if one poly is included in the other (or viceversa).
        # - Or if the polygons are disjointm return None
        n_intersection_points = len(list_edges_intersection_spoints)
        if n_intersection_points == 0:
            polys = ['dummy', self, other]
            if self._is_inside(other):  # within / equals (topologically)
                return polys[-sign]
            if other._is_inside(self):  # contain / equals (topologically)
                return polys[sign]
            if sign == 1:              # if searching for union
                return SMultiPolygon([self, other])
            # if sign=-1 (searching for intersection)
            return None  # disjoint

        # --------------------------------------------------------------------.
        # Define function to reorder the list of SArc(s)
        def reorder_arcs(start_arc, arcs):
            """Return a list of SArc starting with start_arc."""
            idx = arcs.index(start_arc)
            return arcs[idx:] + arcs[:idx]

        # --------------------------------------------------------------------.
        # Starting from an intersection point, follow the edges of a polygon.
        # Restart the loop if not all intersection points have been encountered
        list_intersection_spoints = [tpl[0] for tpl in list_edges_intersection_spoints]
        original_list_intersection_spoints = list_intersection_spoints.copy()
        original_arcs1 = arcs1.copy()
        original_arcs2 = arcs2.copy()

        # Initialize
        RE_INITIALIZE = True
        list_vertices = []
        while len(list_intersection_spoints) > 0:

            # (Re)initialize
            if RE_INITIALIZE:
                idx = original_list_intersection_spoints.index(list_intersection_spoints[0])
                intersection_spoint, edge1, edge2 = list_edges_intersection_spoints[idx]
                RE_INITIALIZE = False
                FLAG_POLYGON_IDENTIFIED = False
                arcs1 = original_arcs1
                arcs2 = original_arcs2
                nodes = []

            # Reorder arcs so that intersecting arc is at first position
            arcs1 = reorder_arcs(edge1, arcs1)
            arcs2 = reorder_arcs(edge2, arcs2)

            # "Close" the polygon with the first arc
            c_arcs1 = arcs1 + [edge1]
            c_arcs2 = arcs2 + [edge2]

            # Get the first SArc from intersection point to original SArc end
            arc1 = SArc(intersection_spoint, edge1.end)
            arc2 = SArc(intersection_spoint, edge2.end)

            # Swap arcs depending on intersection/union
            if np.sign(arc1.angle(arc2)) != sign:
                arcs1, arcs2 = arcs2, arcs1
                c_arcs1, c_arcs2 = c_arcs2, c_arcs1

            # Append start coordinate to nodes list
            nodes.append(intersection_spoint)

            # Update the list of intersection points to still encounter
            try:
                list_intersection_spoints.remove(intersection_spoint)
            except Exception:  # TODO ValueError? AttributeError
                pass

            # Travel along a list of arcs till next intersection
            for edge1 in c_arcs1:
                intersection_spoint, edge2 = edge1.get_next_intersection(c_arcs2, intersection_spoint)
                # When reaching another intersection point, exit the for loop
                if intersection_spoint is not None:
                    break
                # Otherwise add the edges as vertices of the intersecting/union polygon
                elif len(nodes) > 0 and edge1.end not in [nodes[-1], nodes[0]]:
                    nodes.append(edge1.end)

            # If returned to the starting point, the polygon has been identified.
            # - Remove last node if equal to the first
            if intersection_spoint is None:
                if len(nodes) > 2 and nodes[-1] == nodes[0]:
                    nodes = nodes[:-1]
                    FLAG_POLYGON_IDENTIFIED = True
            elif intersection_spoint == nodes[0]:
                FLAG_POLYGON_IDENTIFIED = True

            # Add identified polygon vertices to the list of polygons
            # - Ensure longitudes are within [-pi, pi]
            if FLAG_POLYGON_IDENTIFIED:
                vertices = np.array([(unwrap_radians(node.lon), node.lat) for node in nodes])
                list_vertices.append(vertices)
                RE_INITIALIZE = True

        # Remove duplicate vertices
        list_vertices = [remove_duplicated_vertices(v, tol=1e-08) for v in list_vertices]

        # Check more than two nodes have been found (otherwise line ...)
        list_vertices = [v for v in list_vertices if len(v) > 2]
        if len(list_vertices) == 0:
            return None

        # Retrieve list of SPolygon
        list_SPolygons = [SPolygon(v) for v in list_vertices]

        # If performing union, the first list element might contain the exterior,
        #  while the others the interiors (which are currently discarded)
        if sign == 1 and len(list_SPolygons) > 1:
            print("Union of polygons whose boundaries intersects in more than 4 points might be misleading.")
            list_SPolygons = [list_SPolygons[0]]

        # Return the SPolygon/SMultiPolygons
        return SMultiPolygon(list_SPolygons)

    def union(self, other):
        """Return the union of this and `other` polygon.

        If the two polygons do not overlap (they have nothing in common),
        a MultiPolygon is returned.
        """
        return self._bool_oper(other, 1)

    def intersection(self, other):
        """Return the intersection of this and `other` polygon.

        NB! If the two polygons do not intersect (they have nothing in common)
        None is returned.
        """
        return self._bool_oper(other, -1)

    def within(self, other):
        """Check if the polygon is entirely inside the other."""
        # Check input instance
        if not isinstance(other, (SPolygon, SMultiPolygon, type(None))):
            raise NotImplementedError

        # Iterate over other SMultiPolygon
        if isinstance(other, SMultiPolygon):
            return np.any([self.within(p) for p in other])

        # Logical speed ups
        if isinstance(other, type(None)):
            return False

        # Compute intersection
        intersection = self.intersection(other)

        # Check same area
        if not isinstance(intersection, type(None)):
            return np.abs(intersection._area() - self._area()) < EPSILON
        else:
            return False

    def contains(self, other):
        """Check if the polygon contains entirely the other polygon."""
        if isinstance(other, SMultiPolygon):
            return np.all([p.within(self) for p in other])
        elif isinstance(other, SPolygon):
            return other.within(self)
        else:
            raise NotImplementedError

    def equals(self, other):
        """Test spatial topological equality between two SPolygon.

        If A is within B and B is within A, A and B are considered equal.
        """
        if not isinstance(other, (SPolygon, SMultiPolygon, type(None))):
            raise NotImplementedError
        if isinstance(other, SMultiPolygon) or isinstance(other, type(None)):
            return False
        if self.equals_exact(other):
            return True
        if self.within(other) and other.within(self):
            return True
        else:
            return False

    def intersects(self, other):
        """Return True if intersect, within, contains, or equals other."""
        intersection = self.intersection(other)
        if isinstance(intersection, type(None)):
            return False
        if intersection:
            return True
        else:
            return False

    def disjoint(self, other):
        """Return True if the two polygons does not intersect."""
        return not self.intersects(other)

    def overlap_fraction(self, other):
        """Get the fraction of the current polygon covered by the *other* polygon."""
        intersect_poly = self.intersection(other)
        if intersect_poly is None:
            return 0
        else:
            return intersect_poly._area() / self._area()

    def overlap_rate(self, other):
        """Get the fraction of *other" polygon covered by the current polygon."""
        intersect_poly = self.intersection(other)
        if intersect_poly is None:
            return 0
        else:
            return intersect_poly._area() / other._area()

    def intersection_points(self, other):
        """Get intersection points between 2 polygons."""
        sline = self.to_line()
        if isinstance(other, SLine):
            other_slines = [other]
        if isinstance(other, SPolygon):
            other_slines = [other.to_line()]
        if isinstance(other, SMultiPolygon):
            other_slines = [o.to_line() for o in other]
        list_spoints = [sline.intersection_points(o_sline) for o_sline in other_slines]
        list_spoints = [item for sublist in list_spoints if sublist is not None for item in sublist]
        # list_spoints = [p for p in list_spoints if p is not None]
        return list_spoints

    def to_line(self):
        """Convert SPolygon to SLine."""
        return SLine(_close_vertices(self.vertices))

    def buffer(self, distance, ellips="WGS84"):
        """Return an buffered SPolygon.

        If distance [in km] is positive, it enlarge the SPolygon.
        If distance [in km] is negative, it decrease the size of the SPolygon.
        """
        flag_sign = np.sign(distance)
        distance = np.abs(distance)
        left_sline, right_sline = self.to_line().get_parallel_lines(distance=distance, ellips=ellips)
        if flag_sign == -1:
            return SPolygon(right_sline.vertices)
        else:
            return SPolygon(left_sline.vertices)

    def segmentize(self, npts=0, max_distance=0, ellips='WGS84'):
        """Subdivide each SArc in n steps."""
        sline = self.to_line()
        list_vertices = [arc.segmentize(npts=npts,
                                        max_distance=max_distance,
                                        ellips=ellips).vertices[:-1, :] for arc in sline._list_arcs[0:-1]]
        list_vertices.append(sline._list_arcs[-1].segmentize(npts=npts,
                                                             max_distance=max_distance,
                                                             ellips=ellips).vertices)
        vertices = np.vstack(list_vertices)
        return SPolygon(vertices)

    def to_shapely(self):
        """Convert to Shapely Polygon."""
        return Polygon(np.rad2deg(self.vertices)[::-1])

    def _repr_svg_(self):
        """Display the SPolygon in the Ipython terminal."""
        return self.to_shapely()._repr_svg_()

    def plot(self, ax=None, facecolor=None, edgecolor='black', alpha=0.4, **kwargs):
        """Plot the SPolygon using Cartopy."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # Retrieve shapely polygon
        geom = self.to_shapely()

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Plot polygon
        ax.add_geometries([geom], crs=ccrs.Geodetic(),
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          alpha=alpha,
                          **kwargs)

        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
        return ax


class SMultiPolygon():
    """A collection of one or more SPolygons.

    The collection of SPolygon(s) must not overlap !
    If component polygons overlap the collection is `invalid` and
    some operations on it may fail.
    """

    def __new__(cls, polygons, check_validity=False):
        """Create SPolygon or SMultiPolygon object.

        If polygons is a list with only a single SPolygon, it returns a SPolygon.
        None value present in the list are removed.
        """
        # Check providing a list of SPolygon only (and eventually None)
        if not isinstance(polygons, list):
            raise TypeError("Not providing a list of SPolygons to SMultiPolygon.")
        if not np.all([isinstance(p, (SPolygon, type(None))) for p in polygons]):
            raise ValueError("SMultiPolygon accepts only a list of SPolygon.")

        # Remove possible None
        idx_polygons = np.where([not isinstance(p, type(None)) for p in polygons])[0]

        # If only None, return None
        # - This is important when _bool_oper returns None (i.e. not intersecting)
        if len(idx_polygons) == 0:
            return None

        # If a single SPolygon, return SPolygon
        polygons = [polygons[i] for i in idx_polygons]
        if len(polygons) == 1:
            return polygons[0]

        # Else create SMultiPolygon
        return super().__new__(cls)

    def __init__(self, polygons, check_validity=False):
        self.geoms = polygons
        # Check that polygons does not overlap
        # - This function is quite slow
        if check_validity:
            if not self.is_valid_geometry:
                raise ValueError("The SPolygon(s) composing SMultiPolygon must not overlap.")

    @property
    def is_valid_geometry(self):
        """Check that there are not self intersections."""
        # First check for non self-intersection in each single SPolygon
        if not np.all([p.is_valid_geometry for p in self.geoms]):
            return False
        # Then, check the SPolygons does not overlap
        out = self.union(self)
        if len(out) == len(self):
            return True
        else:
            False

    def __getitem__(self, i):
        """Get a specific SPolygon composing SMultiPolygon."""
        return self.geoms[i]

    def __len__(self):
        """Get the number of SPolygon composing SMultiPolygon."""
        return len(self.geoms)

    def __iter__(self):
        """Get an iterator returning the SPolygons composing SMultiPolygon."""
        return self.geoms.__iter__()

    def _area(self):
        """Compute the area of the polygon in units of square radii."""
        return np.sum([p._area() for p in self.geoms])

    def area(self):
        """Compute the polygon area [in km²] using pyproj."""
        return np.sum([p.area() for p in self.geoms])

    def union(self, other):
        """Return the union of this and `other` polygon."""
        list_polygons = self.geoms + other.geoms
        union_p = list_polygons[0]
        for p in list_polygons[1:]:
            union_p = p.union(union_p)  # ensure call to SPolygon.union
        return union_p

    def intersection(self, other):
        """Return the intersection of this and `other` polygon."""
        p_inter = SMultiPolygon([p.intersection(other) for p in self.geoms])
        # Tentative union of intersections if a MultiPolygon
        if isinstance(p_inter, SMultiPolygon):
            p_inter = p_inter.geoms[0].union(p_inter.geoms[1:])
        return p_inter

    def overlap_rate(self, other):
        """Get how much the current polygon overlaps the *other* polygon."""
        intersect_polys = ([p.intersection(other) for p in self.geoms])
        list_polygons = [p for p in intersect_polys if (p is not None)]
        if len(list_polygons) == 0:
            return 0
        intersect_area = np.sum([p.area() for p in list_polygons])
        return intersect_area / other.area()

    def intersects(self, other):
        """Return True if intersect, within, contains, or equals other."""
        return np.any([p.intersects(other) for p in self.geoms])

    def disjoint(self, other):
        """Return True if the two polygons does not intersect."""
        return np.all([p.disjoint(other) for p in self.geoms])

    def within(self, other):
        """Check if the polygon is entirely inside the other."""
        return np.all([p.within(other) for p in self.geoms])

    def contains(self, other):
        """Check if the polygon contains entirely the other polygon."""
        return np.any([p.contains(other) for p in self.geoms])

    def equals_exact(self, other, rtol=1e-05, atol=1e-08):
        """Check if SMultiPolygon is exactly equal to the other SMultiPolygon.

        This method uses exact geometry and coordinate equality,
        which requires geometries and coordinates to be equal
        (within specified tolerance) and in the same order.
        This is in contrast with the ``equals`` function which checks for
        spatial (topological) equality.
        """
        if not isinstance(other, (SPolygon, SMultiPolygon, type(None))):
            raise NotImplementedError
        if isinstance(other, SPolygon) or isinstance(other, type(None)):
            return False
        if len(self.geoms) != len(other.geoms):
            return False
        return np.all([s.equals_exact(o, rtol=rtol, atol=atol) for s, o in zip(self.geoms, other.geoms)])

    def equals(self, other):
        """Test spatial topological equality between two SMultiPolygon.

        If A is within B and B is within A, A and B are considered equal.
        The geometries within SMultiPolygon must not be necessary in the same order.
        """
        if not isinstance(other, (SPolygon, SMultiPolygon, type(None))):
            raise NotImplementedError
        if isinstance(other, SPolygon) or isinstance(other, type(None)):
            return False
        if len(self.geoms) != len(other.geoms):
            return False
        if self.equals_exact(other):
            return True
        if (np.all([np.any([o.equals(s) for s in self.geoms]) for i, o in enumerate(other.geoms)]) and
                np.all([np.any([o.equals(s) for o in other.geoms]) for i, s in enumerate(self.geoms)])):
            return True
        else:
            return False

    def intersection_points(self, other):
        """Get intersection points between 2 polygons."""
        list_spoints = [p.intersection_points(other) for p in self.geoms]
        list_spoints = [item for sublist in list_spoints if sublist is not None for item in sublist]
        return list_spoints

    def to_shapely(self):
        """Convert to Shapely MultiPolygon."""
        return MultiPolygon([p.to_shapely() for p in self.geoms])

    def _repr_svg_(self):
        """Display SPolygons in the Ipython terminal."""
        return self.to_shapely()._repr_svg_()

    def plot(self, ax=None, facecolor=None, edgecolor='black', alpha=0.4, **kwargs):
        """Plot the SPolygon using Cartopy."""
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # Retrieve shapely polygon
        bbox_geom = self.to_shapely()

        # Create figure if ax not provided
        ax_not_provided = False
        if ax is None:
            ax_not_provided = True
            proj_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

        # Add swath polygon
        ax.add_geometries([bbox_geom],
                          crs=ccrs.Geodetic(),
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          alpha=alpha,
                          **kwargs)

        # Beautify plot by default
        if ax_not_provided:
            ax.stock_img()
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
        return ax

# ----------------------------------------------------------------------------.
# Backward Compatibilites
# --> TO BE CHECKED... NOT SURE IS THE CORRECT APPROACH


class SCoordinate(SPoint):
    """Introduce SCoordinate for backcompatibility reasons."""

    def __init_subclass__(self):
        """Get a dummy docstring for flake8 hook isort sake."""
        warnings.warn("SCoordinate is deprecated. Use SPoint instead.",
                      PendingDeprecationWarning, 2)


class Arc(SArc):
    """Introduce Arc for backcompatibility reasons."""

    def __init_subclass__(self):
        """Get a dummy docstring for flake8 hook isort sake."""
        warnings.warn("Arc is deprecated. Use SArc instead.",
                      PendingDeprecationWarning, 2)


class SphPolygon(SPolygon):  # TODO: radius can not be passed anymore
    """Introduce SphPolygon for backcompatibility reasons."""

    def __init_subclass__(self):
        """Get a dummy docstring for flake8 hook isort sake."""
        warnings.warn("SphPolygon is deprecated. Use SPolygon instead.",
                      PendingDeprecationWarning, 2)

    def area(self):
        """Redefine area for backcompatibility reasons."""
        return SPolygon._area(self)
