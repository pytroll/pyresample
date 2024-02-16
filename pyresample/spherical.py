#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 - 2021 Pyresample developers
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
"""Some generalized spherical functions.

base type is a numpy array of size (n, 2) (2 for lon and lats)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 0.0000001


def _unwrap_radians(val, mod=np.pi):
    """Put *val* between -*mod* and *mod*."""
    return (val + mod) % (2 * mod) - mod


def _xyz_to_vertices(x, y, z):
    """Create vertices array from x,y,z values or vectors.

    If x, y, z are scalar arrays, it creates a 1x3 np.array.
    If x, y, z are np.array with shape nx1, it creates a nx3 np.array.
    """
    if x.ndim == 0:
        vertices = np.array([x, y, z])
    else:
        vertices = np.vstack([x, y, z]).T
    return vertices


def _ensure_is_array(arr):
    """Ensure that a possible np.value input is converted to np.array."""
    if arr.ndim == 0:
        arr = np.asarray([arr])
    return arr


def _vincenty_matrix(lon, lat, lon_ref, lat_ref):
    """Compute a distance matrix using Vincenty formula.

    The lon/lat inputs must be provided in radians !
    The output must be multiplied by the Earth radius to obtain the distance in m or km.
    The returned distance matrix has shape (n x n_ref).
    """
    lon = _ensure_is_array(lon)
    lat = _ensure_is_array(lat)
    lon_ref = _ensure_is_array(lon_ref)
    lat_ref = _ensure_is_array(lat_ref)
    lon = lon[:, np.newaxis]
    lat = lat[:, np.newaxis]
    diff_lon = lon - lon_ref
    num = ((np.cos(lat_ref) * np.sin(diff_lon)) ** 2 +
           (np.cos(lat) * np.sin(lat_ref) -
            np.sin(lat) * np.cos(lat_ref) * np.cos(diff_lon)) ** 2)
    den = (np.sin(lat) * np.sin(lat_ref) +
           np.cos(lat) * np.cos(lat_ref) * np.cos(diff_lon))
    dist = np.arctan2(num ** .5, den)
    return dist


def _haversine_matrix(lon, lat, lon_ref, lat_ref):
    """Compute a distance matrix using haversine formula.

    The lon/lat inputs must be provided in radians !
    The output must be multiplied by the Earth radius to obtain the distance in m or km.
    The returned distance matrix has shape (n x n_ref).
    """
    lon = _ensure_is_array(lon)
    lat = _ensure_is_array(lat)
    lon_ref = _ensure_is_array(lon_ref)
    lat_ref = _ensure_is_array(lat_ref)
    lon = lon[:, np.newaxis]
    lat = lat[:, np.newaxis]
    diff_lon = lon - lon_ref  # n x n_ref matrix
    diff_lat = lat - lat_ref  # n x n_ref matrix
    a = np.sin(diff_lat / 2.0) ** 2.0 + np.cos(lat) * np.cos(lat_ref) * np.sin(diff_lon / 2.0) ** 2.0
    dist = 2.0 * np.arcsin(a ** .5)  # equivalent of; 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return dist


def _check_lon_validity(lon):
    """Check longitude validity."""
    if np.any(np.isinf(lon)):
        raise ValueError("Longitude values can not contain inf values.")


def _check_lat_validity(lat):
    """Check latitude validity."""
    if np.any(np.isinf(lat)):
        raise ValueError("Latitude values can not contain inf values.")
    if np.any(np.logical_or(lat > np.pi / 2, lat < -np.pi / 2)):
        raise ValueError("Latitude values must range between [-pi/2, pi/2].")


def _check_lon_lat(lon, lat):
    """Check and format lon/lat values/arrays."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    _check_lon_validity(lon)
    _check_lat_validity(lat)
    return lon, lat


class SCoordinate(object):
    """Spherical coordinates.

    The ``lon`` and ``lat`` coordinates must be provided in radians.

    """

    def __init__(self, lon, lat):
        lon, lat = _check_lon_lat(lon, lat)
        self.lon = _unwrap_radians(lon)
        self.lat = lat

    @property
    def vertices(self):
        """Return point(s) radians vertices in a ndarray of shape [n,2]."""
        # Single values
        if self.lon.ndim == 0:
            vertices = np.array([self.lon, self.lat])[np.newaxis, :]
        # Array values
        else:
            vertices = np.vstack((self.lon, self.lat)).T
        return vertices

    @property
    def vertices_in_degrees(self):
        """Return point(s) degrees vertices in a ndarray of shape [n,2]."""
        return np.rad2deg(self.vertices)

    def cross2cart(self, point):
        """Compute the cross product, and convert to cartesian coordinates.

        Note:
        - the cross product of the same point gives a zero vector.
        - the cross product between points lying at the equator gives a zero vector.
        - the cross product between points lying at the poles.
        """
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
        x = -ad * c + be * f
        y = ad * f + be * c
        z = g * h * i
        vertices = _xyz_to_vertices(x, y, z)
        res = CCoordinate(vertices)
        return res

    def to_cart(self):
        """Convert to cartesian."""
        x = np.cos(self.lat) * np.cos(self.lon)
        y = np.cos(self.lat) * np.sin(self.lon)
        z = np.sin(self.lat)
        vertices = _xyz_to_vertices(x, y, z)
        return CCoordinate(vertices)

    def distance(self, point):
        """Get distance using Vincenty formula.

        The result must be multiplied by Earth radius to obtain distance in m or km.
        """
        lat = self.lat
        lon = self.lon
        lon_ref = point.lon
        lat_ref = point.lat
        dist = _vincenty_matrix(lon, lat, lon_ref, lat_ref)
        if dist.size == 1:  # single point case
            dist = dist.item()
        return dist

    def hdistance(self, point):
        """Get distance using Haversine formula.

        The result must be multiplied by Earth radius to obtain distance in m or km.
        """
        lat = self.lat
        lon = self.lon
        lon_ref = point.lon
        lat_ref = point.lat
        dist = _haversine_matrix(lon, lat, lon_ref, lat_ref)
        if dist.size == 1:  # single point case
            dist = dist.item()
        return dist

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)

    def __eq__(self, other):
        """Check equality."""
        return np.allclose((self.lon, self.lat), (other.lon, other.lat))

    def __str__(self):
        """Get simplified representation of lon/lat arrays in degrees."""
        return str((float(np.rad2deg(self.lon)), float(np.rad2deg(self.lat))))

    def __repr__(self):
        """Get simplified representation of lon/lat arrays in degrees."""
        return str((float(np.rad2deg(self.lon)), float(np.rad2deg(self.lat))))

    def __iter__(self):
        """Get iterator over lon/lat pairs."""
        return zip([self.lon, self.lat]).__iter__()

    def plot(self, ax=None,
             projection_crs=None,
             add_coastlines=True,
             add_gridlines=True,
             add_background=True,
             **plot_kwargs):
        """Plot the point(s) using Cartopy.

        Assume vertices to be in radians.
        """
        import matplotlib.pyplot as plt
        try:
            import cartopy.crs as ccrs
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Install cartopy to plot spherical geometries. For example, 'pip install cartopy'.") from err

        # Create figure if ax not provided
        if ax is None:
            if projection_crs is None:
                projection_crs = ccrs.PlateCarree()
            fig, ax = plt.subplots(subplot_kw=dict(projection=projection_crs))

        # Plot Points
        vertices = self.vertices_in_degrees
        ax.scatter(x=vertices[:, 0],
                   y=vertices[:, 1],
                   **plot_kwargs)

        # Beautify plots
        if add_background:
            ax.stock_img()
        if add_coastlines:
            ax.coastlines()
        if add_gridlines():
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

    def normalize(self, inplace=False):
        """Normalize the vector.

        If self.cart == [0,0,0], norm=0, and cart becomes [nan, nan, nan]:
        Note that self.cart == [0,0,0] can occurs when computing:
        - the cross product of the same point.
        - the cross product between points lying at the equator.
        - the cross product between points lying at the poles.
        """
        norm = self.norm()
        norm = norm[..., np.newaxis]  # enable vectorization
        if inplace:
            self.cart /= norm
            return None
        cart = self.cart / norm
        return CCoordinate(cart)

    def cross(self, point):
        """Get cross product with another vector.

        The cross product of the same vector gives a zero vector.
        """
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
        """Convert to SPoint/SMultiPoint object."""
        # TODO: this in future should point to SPoint or SMultiPoint
        lon = np.arctan2(self.cart[..., 1], self.cart[..., 0])
        lat = np.arcsin(self.cart[..., 2])
        return SCoordinate(lon, lat)


class Arc(object):
    """An arc of the great circle between two points."""

    def __init__(self, start, end):
        self.start, self.end = start, end

    def __eq__(self, other):
        """Check equality."""
        if self.start == other.start and self.end == other.end:
            return 1
        return 0

    def __ne__(self, other):
        """Check not equal comparison."""
        return not self.__eq__(other)

    def __str__(self):
        """Get simplified representation."""
        return str(self.start) + " -> " + str(self.end)

    def __repr__(self):
        """Get simplified representation."""
        return str(self.start) + " -> " + str(self.end)

    def angle(self, other_arc):
        """Oriented angle between two arcs.

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

    def intersections(self, other_arc):
        """Give the two intersections of the greats circles defined by the current arc and *other_arc*.

        From http://williams.best.vwh.net/intersect.htm
        """
        end_lon = self.end.lon
        other_end_lon = other_arc.end.lon

        if self.end.lon - self.start.lon > np.pi:
            end_lon -= 2 * np.pi
        if other_arc.end.lon - other_arc.start.lon > np.pi:
            other_end_lon -= 2 * np.pi
        if self.end.lon - self.start.lon < -np.pi:
            end_lon += 2 * np.pi
        if other_arc.end.lon - other_arc.start.lon < -np.pi:
            other_end_lon += 2 * np.pi

        end_point = SCoordinate(end_lon, self.end.lat)
        other_end_point = SCoordinate(other_end_lon, other_arc.end.lat)

        ea_ = self.start.cross2cart(end_point).normalize()
        eb_ = other_arc.start.cross2cart(other_end_point).normalize()

        cross = ea_.cross(eb_)
        lat = np.arctan2(cross.cart[2],
                         np.sqrt(cross.cart[0] ** 2 + cross.cart[1] ** 2))
        lon = np.arctan2(cross.cart[1], cross.cart[0])

        return (SCoordinate(lon, lat),
                SCoordinate(_unwrap_radians(lon + np.pi), -lat))

    def intersects(self, other_arc):
        """Check if the current arc and the *other_arc* intersect.

        An arc is defined as the shortest tracks between two points.
        """
        return bool(self.intersection(other_arc))

    def intersection(self, other_arc):
        """Return where, if the current arc and the *other_arc* intersect.

        None is returned if there is not intersection. An arc is defined
        as the shortest tracks between two points.
        """
        if self == other_arc:
            return None

        for i in self.intersections(other_arc):
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
            d__ = other_arc.end

            ab_ = a__.hdistance(b__)
            cd_ = c__.hdistance(d__)

            if (((i in (a__, b__)) or
                (abs(a__.hdistance(i) + b__.hdistance(i) - ab_) < EPSILON)) and
                ((i in (c__, d__)) or
                 (abs(c__.hdistance(i) + d__.hdistance(i) - cd_) < EPSILON))):
                return i
        return None

    def get_next_intersection(self, arcs, known_inter=None):
        """Get the next intersection between the current arc and *arcs*."""
        res = []
        for arc in arcs:
            inter = self.intersection(arc)
            if (inter is not None and
                    inter != arc.end and
                    inter != self.end):
                res.append((inter, arc))

        def dist(args):
            """Get distance key."""
            return self.start.distance(args[0])

        take_next = False
        for inter, arc in sorted(res, key=dist):
            if known_inter is not None:
                if known_inter == inter:
                    take_next = True
                elif take_next:
                    return inter, arc
            else:
                return inter, arc

        return None, None


class SphPolygon:
    """Spherical polygon.

    Represents a polygon on a spherical geoid.  Initialise with
    an ndarray of shape ``[N, 2]`` where the first column contains longitudes
    and the second column contains latitudes.  The units should be in radians.
    The inside of the polygon is defined by the vertices being defined clockwise
    around it.

    The optional second argument ``radius`` indicates the radius of the
    spherical geoid on which calculations occur.

    """

    def __init__(self, vertices, radius=1):
        """Initialise SphPolygon object.

        Args:
            vertices (np.ndarray): ndarray of shape ``[N, 2]`` with ``N``
                points describing a polygon clockwise.  First column
                describes longitudes, second column describes latitudes.  Units
                should be in radians.
            radius (optional, number): Radius of spherical planet.
        """
        self.vertices = vertices.astype(np.float64, copy=False)
        self.lon = _unwrap_radians(self.vertices[:, 0])
        self.lat = self.vertices[:, 1]
        self.radius = radius
        self.cvertices = np.array([np.cos(self.lat) * np.cos(self.lon),
                                   np.cos(self.lat) * np.sin(self.lon),
                                   np.sin(self.lat)]).T * radius
        self.x__ = self.cvertices[:, 0]
        self.y__ = self.cvertices[:, 1]
        self.z__ = self.cvertices[:, 2]

    def invert(self):
        """Invert the polygon."""
        self.vertices = np.flipud(self.vertices)
        self.cvertices = np.flipud(self.cvertices)
        self.lon = self.vertices[:, 0]
        self.lat = self.vertices[:, 1]
        self.x__ = self.cvertices[:, 0]
        self.y__ = self.cvertices[:, 1]
        self.z__ = self.cvertices[:, 2]

    def inverse(self):
        """Return an inverse of the polygon."""
        return SphPolygon(np.flipud(self.vertices), radius=self.radius)

    def aedges(self):
        """Get generator over the edges, in arcs of Coordinates."""
        for (lon_start, lat_start), (lon_stop, lat_stop) in self.edges():
            yield Arc(SCoordinate(lon_start, lat_start),
                      SCoordinate(lon_stop, lat_stop))

    def edges(self):
        """Get generator over the edges, in geographical coordinates."""
        for i in range(len(self.lon) - 1):
            yield (self.lon[i], self.lat[i]), (self.lon[i + 1], self.lat[i + 1])
        yield (self.lon[i + 1], self.lat[i + 1]), (self.lon[0], self.lat[0])

    def area(self):
        """Find the area of a polygon.

        The inside of the polygon is defined by having the vertices enumerated
        clockwise around it.

        Uses the algorithm described in [bev1987]_.

        .. [bev1987] , Michael Bevis and Greg Cambareri,
           "Computing the area of a spherical polygon of arbitrary shape",
           in *Mathematical Geology*, May 1987, Volume 19, Issue 4, pp 335-346.

        Note: The article mixes up longitudes and latitudes in equation 3! Look
        at the fortran code appendix for the correct version.

        The units are the square of the radius passed to the constructor.  For
        example, to calculate the area in km² of a polygon near the equator of a
        spherical planet with a radius of 6371 km (similar to Earth):

        >>> pol = SphPolygon(np.deg2rad(np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])),
                             radius=6371)
        >>> print(pol.area())
        12363.997753690213

        If `SphPolygon` was constructed without passing any units, the result
        has units of square radii (i.e., the polygon containing the entire
        planet would have area 4π).
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

        return (sum(alpha) - (len(self.lon) - 2) * np.pi) * self.radius ** 2

    def _bool_oper(self, other, sign=1):
        """Perform a boolean operation on this and *other* polygons.abs.

        By default, or when sign is 1, the union is perfomed. If sign is -1,
        the intersection of the polygons is returned.

        The algorithm works this way: Find an intersection between the two
        polygons. If none can be found, then the two polygons are either not
        overlapping, or one is entirely included in the other. Otherwise,
        follow the edges of a polygon until another intersection is
        encountered, at which point you start following the edges of the other
        polygon, and so on until you come back to the first intersection. In
        which direction to follow the edges of the polygons depends if you are
        interested in the union or the intersection of the two polygons.
        """
        arcs1 = [edge for edge in self.aedges()]
        arcs2 = [edge for edge in other.aedges()]

        # find the first intersection, to start from.
        for edge1 in arcs1:
            inter, edge2 = edge1.get_next_intersection(arcs2)
            if inter is not None and inter != edge1.end and inter != edge2.end:
                break

        # if no intersection is found, find out if the one poly is included in
        # the other.
        if inter is None:
            polys = [0, self, other]
            if self._is_inside(other):
                return polys[-sign]
            if other._is_inside(self):
                return polys[sign]

            return None

        nodes = self._find_intersection_nodes(inter, edge1, edge2, arcs1, arcs2, sign)
        return SphPolygon(np.array([(node.lon, node.lat) for node in nodes]), radius=self.radius)

    @staticmethod
    def _find_intersection_nodes(inter, edge1, edge2, arcs1, arcs2, sign):
        def rotate_arcs(start_arc, arcs):
            idx = arcs.index(start_arc)
            return arcs[idx:] + arcs[:idx]

        # starting from the intersection, follow the edges of one of the polygons
        nodes = []
        while True:
            arcs1 = rotate_arcs(edge1, arcs1)
            arcs2 = rotate_arcs(edge2, arcs2)

            narcs1 = arcs1 + [edge1]
            narcs2 = arcs2 + [edge2]

            arc1 = Arc(inter, edge1.end)
            arc2 = Arc(inter, edge2.end)

            if np.sign(arc1.angle(arc2)) != sign:
                arcs1, arcs2 = arcs2, arcs1
                narcs1, narcs2 = narcs2, narcs1

            nodes.append(inter)

            for edge1 in narcs1:
                inter, edge2 = edge1.get_next_intersection(narcs2, inter)
                if inter is not None:
                    break
                elif len(nodes) > 0 and edge1.end not in [nodes[-1], nodes[0]]:
                    nodes.append(edge1.end)

            if inter is None and len(nodes) > 2 and nodes[-1] == nodes[0]:
                nodes = nodes[:-1]
                break
            if inter == nodes[0]:
                break
        return nodes

    def union(self, other):
        """Return the union of this and `other` polygon.

        NB! If the two polygons do not overlap (they have nothing in common) None is returned.
        """
        return self._bool_oper(other, 1)

    def intersection(self, other):
        """Return the intersection of this and `other` polygon."""
        return self._bool_oper(other, -1)

    def _is_inside(self, other):
        """Check if the polygon is entirely inside the other.

        Should be used with :meth:`inter` first to check if the is a
        known intersection.
        """
        anti_lon_0 = self.lon[0] + np.pi
        if anti_lon_0 > np.pi:
            anti_lon_0 -= np.pi * 2

        anti_lon_1 = self.lon[1] + np.pi
        if anti_lon_1 > np.pi:
            anti_lon_1 -= np.pi * 2

        arc1 = Arc(SCoordinate(self.lon[1],
                               self.lat[1]),
                   SCoordinate(anti_lon_0,
                               -self.lat[0]))

        arc2 = Arc(SCoordinate(anti_lon_0,
                               -self.lat[0]),
                   SCoordinate(anti_lon_1,
                               -self.lat[1]))

        arc3 = Arc(SCoordinate(anti_lon_1,
                               -self.lat[1]),
                   SCoordinate(self.lon[0],
                               self.lat[0]))

        other_arcs = [edge for edge in other.aedges()]
        for arc in [arc1, arc2, arc3]:
            inter, other_arc = arc.get_next_intersection(other_arcs)
            if inter is not None:
                sarc = Arc(arc.start, inter)
                earc = Arc(inter, other_arc.end)
                return sarc.angle(earc) < 0
        return other.area() > (2 * np.pi * other.radius ** 2)

    def __str__(self):
        """Get numpy representation of vertices."""
        return str(np.rad2deg(self.vertices))
