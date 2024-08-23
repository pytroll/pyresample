#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2023 Pyresample developers
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
"""Classes for geometry operations."""
from __future__ import annotations

import hashlib
import math
import warnings
from collections import OrderedDict
from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pyproj
import yaml
from pyproj import Geod, Proj
from pyproj.aoi import AreaOfUse

from pyresample import CHUNK_SIZE
from pyresample._spatial_mp import Cartesian, Cartesian_MP, Proj_MP
from pyresample.area_config import create_area_def
from pyresample.boundary import SimpleBoundary
from pyresample.utils import load_cf_area
from pyresample.utils.proj4 import (
    get_geodetic_crs_with_no_datum_shift,
    get_geostationary_height,
    ignore_pyproj_proj_warnings,
    proj4_dict_to_str,
    proj4_radius_parameters,
)

from . import _formatting_html

try:
    from xarray import DataArray
except ImportError:
    DataArray = np.ndarray

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import odc.geo as odc_geo
except ModuleNotFoundError:
    odc_geo = None

from pyproj import CRS
from pyproj.enums import TransformDirection

logger = getLogger(__name__)

if TYPE_CHECKING:
    # defined in typeshed to hide private C-level type
    from hashlib import _Hash


class DimensionError(ValueError):
    """Wrap ValueError."""


class IncompatibleAreas(ValueError):
    """Error when the areas to combine are not compatible."""


class InvalidArea(ValueError):
    """Error to be raised when an area is invalid for a given purpose."""


class BaseDefinition:
    """Base class for geometry definitions.

    .. versionchanged:: 1.8.0

        `BaseDefinition` no longer checks the validity of the provided
        longitude and latitude coordinates to improve performance. Longitude
        arrays are expected to be between -180 and 180 degrees, latitude -90
        to 90 degrees. Use :func:`~pyresample.utils.check_and_wrap` to preprocess
        your arrays.
    """

    def __init__(self, lons=None, lats=None, nprocs=1):
        """Initialize BaseDefinition."""
        if type(lons) is not type(lats):
            raise TypeError('lons and lats must be of same type')
        elif lons is not None:
            if not isinstance(lons, (np.ndarray, DataArray)):
                lons = np.asanyarray(lons)
                lats = np.asanyarray(lats)
            if lons.shape != lats.shape:
                raise ValueError('lons and lats must have same shape')

        self.nprocs = nprocs
        self.lats = lats
        self.lons = lons
        self.ndim = None
        self.cartesian_coords = None
        self.hash = None

    def __getitem__(self, key):
        """Slice a 2D geographic definition."""
        y_slice, x_slice = key
        return self.__class__(
            lons=self.lons[y_slice, x_slice],
            lats=self.lats[y_slice, x_slice],
            nprocs=self.nprocs
        )

    def __hash__(self):
        """Compute the hash of this object."""
        if self.hash is None:
            self.hash = int(self.update_hash().hexdigest(), 16)
        return self.hash

    def update_hash(self, existing_hash: Optional[_Hash] = None) -> _Hash:
        """Update the hash."""
        if existing_hash is None:
            existing_hash = hashlib.sha1()  # nosec: B324
        existing_hash.update(get_array_hashable(self.lons))
        existing_hash.update(get_array_hashable(self.lats))
        try:
            if self.lons.mask is not False:
                existing_hash.update(get_array_hashable(self.lons.mask))
        except AttributeError:
            pass
        return existing_hash

    def __eq__(self, other):
        """Test for approximate equality."""
        if self is other:
            return True
        if not isinstance(other, BaseDefinition):
            return False
        self_lons, self_lats = self._extract_lonlat_subarrays(self)
        other_lons, other_lats = self._extract_lonlat_subarrays(other)
        if self_lons is other_lons and self_lats is other_lats:
            return True

        arrs_to_comp = (self_lons, self_lats, other_lons, other_lats)
        all_dask_arrays = da is not None and all(isinstance(x, da.Array) for x in arrs_to_comp)
        if all_dask_arrays:
            # Optimization: We assume that if two dask arrays have the same task name
            # they are equivalent. This allows for using geometry objects in dict keys
            # without computing the dask arrays underneath.
            return self_lons.name == other_lons.name and self_lats.name == other_lats.name

        try:
            lons_close = np.allclose(self_lons, other_lons, atol=1e-6, rtol=5e-9, equal_nan=True)
            if not lons_close:
                return False
            lats_close = np.allclose(self_lats, other_lats, atol=1e-6, rtol=5e-9, equal_nan=True)
            return lats_close
        except ValueError:
            return False

    @staticmethod
    def _extract_lonlat_subarrays(
            geom_obj: BaseDefinition
    ) -> tuple[npt.ArrayLike | da.Array, npt.ArrayLike | da.Array]:
        if geom_obj.lons is None or geom_obj.lats is None:
            lons, lats = geom_obj.get_lonlats()
        else:
            lons = geom_obj.lons
            lats = geom_obj.lats

        if isinstance(lons, DataArray) and np.ndarray is not DataArray:
            lons = lons.data
            lats = lats.data
        return lons, lats

    def __ne__(self, other):
        """Test for approximate equality."""
        return not self.__eq__(other)

    def get_area_extent_for_subset(self, row_LR, col_LR, row_UL, col_UL):
        """Calculate extent for a subdomain of this area.

        Rows are counted from upper left to lower left and columns are
        counted from upper left to upper right.

        Args:
            row_LR (int): row of the lower right pixel
            col_LR (int): col of the lower right pixel
            row_UL (int): row of the upper left pixel
            col_UL (int): col of the upper left pixel

        Returns:
            area_extent (tuple):
                Area extent (LL_x, LL_y, UR_x, UR_y) of the subset

        Author:
            Ulrich Hamann
        """
        (a, b) = self.get_proj_coords(data_slice=(row_LR, col_LR))
        a = a - 0.5 * self.pixel_size_x
        b = b - 0.5 * self.pixel_size_y
        (c, d) = self.get_proj_coords(data_slice=(row_UL, col_UL))
        c = c + 0.5 * self.pixel_size_x
        d = d + 0.5 * self.pixel_size_y

        return a, b, c, d

    def get_lonlat(self, row, col):
        """Retrieve lon and lat of single pixel.

        Parameters
        ----------
        row : int
        col : int

        Returns
        -------
        (lon, lat) : tuple of floats
        """
        if self.ndim != 2:
            raise DimensionError(('operation undefined '
                                  'for %sD geometry ') % self.ndim)
        elif self.lons is None or self.lats is None:
            raise ValueError('lon/lat values are not defined')
        return self.lons[row, col], self.lats[row, col]

    def get_lonlats(self, data_slice=None, chunks=None, **kwargs):
        """Get longitude and latitude arrays representing this geometry.

        Returns
        -------
        (lon, lat) : tuple of numpy arrays
            If `chunks` is provided then the arrays will be dask arrays
            with the provided chunk size. If `chunks` is not provided then
            the returned arrays are the same as the internal data types
            of this geometry object (numpy or dask).
        """
        lons = self.lons
        lats = self.lats
        if lons is None or lats is None:
            raise ValueError('lon/lat values are not defined')
        elif DataArray is not np.ndarray and isinstance(lons, DataArray):
            # lons/lats are xarray DataArray objects, use numpy/dask array underneath
            lons = lons.data
            lats = lats.data

        if chunks is not None:
            import dask.array as da
            if isinstance(lons, da.Array):
                # rechunk to this specific chunk size
                lons = lons.rechunk(chunks)
                lats = lats.rechunk(chunks)
            elif not isinstance(lons, da.Array):
                # convert numpy array to dask array
                lons = da.from_array(np.asanyarray(lons), chunks=chunks)
                lats = da.from_array(np.asanyarray(lats), chunks=chunks)
        if data_slice is not None:
            lons, lats = lons[data_slice], lats[data_slice]
        return lons, lats

    def get_lonlats_dask(self, chunks=None):
        """Get the lon lats as a single dask array."""
        warnings.warn("'get_lonlats_dask' is deprecated, please use "
                      "'get_lonlats' with the 'chunks' keyword argument specified.", DeprecationWarning, stacklevel=2)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_lonlats(chunks=chunks)

    def get_boundary_lonlats(self):
        """Return Boundary objects."""
        s1_lon, s1_lat = self.get_lonlats(data_slice=(0, slice(None)))
        s2_lon, s2_lat = self.get_lonlats(data_slice=(slice(None), -1))
        s3_lon, s3_lat = self.get_lonlats(data_slice=(-1, slice(None, None, -1)))
        s4_lon, s4_lat = self.get_lonlats(data_slice=(slice(None, None, -1), 0))
        return (SimpleBoundary(s1_lon.squeeze(), s2_lon.squeeze(), s3_lon.squeeze(), s4_lon.squeeze()),
                SimpleBoundary(s1_lat.squeeze(), s2_lat.squeeze(), s3_lat.squeeze(), s4_lat.squeeze()))

    def get_bbox_lonlats(self, vertices_per_side: Optional[int] = None, force_clockwise: bool = True,
                         frequency: Optional[int] = None) -> tuple:
        """Return the bounding box lons and lats sides.

        Args:
            vertices_per_side:
                The number of points to provide for each side. By default (None)
                the full width and height will be provided.
            frequency:
                Deprecated, use vertices_per_side
            force_clockwise:
                Perform minimal checks and reordering of coordinates to ensure
                that the returned coordinates follow a clockwise direction.
                This is important for compatibility with
                :class:`pyresample.spherical.SphPolygon` where operations depend
                on knowing the inside versus the outside of a polygon. These
                operations assume that coordinates are clockwise.
                Default is True.

        Returns:
            Two lists of four elements each. The first list is longitude
            coordinates, the second latitude. Each element is a numpy array
            representing a specific side of the geometry. The order of the
            arrays is first row (index 0), last column, last row, and first
            column. The arrays are sliced (ordered) in a way to ensure that the
            coordinates follow a clockwise path. In the usual case this results
            in the coordinates starting in the north-west corner. In the case
            where the data is oriented with the first pixel (row 0, column 0)
            in the south-east corner, the coordinates will start in that
            corner. Other orientations that are detected to follow a
            counter-clockwise path will be reordered to provide a
            clockwise path in order to be compatible with other parts of
            pyresample (ex. :class:`pyresample.spherical.SphPolygon`).

        """
        if frequency is not None:
            warnings.warn("The `frequency` argument is pending deprecation, use `vertices_per_side` instead",
                          PendingDeprecationWarning, stacklevel=2)
        vertices_per_side = vertices_per_side or frequency
        lon_sides, lat_sides = self._get_geographic_sides(vertices_per_side=vertices_per_side)
        if force_clockwise and not self._corner_is_clockwise(
                lon_sides[0][-2], lat_sides[0][-2],
                lon_sides[0][-1], lat_sides[0][-1],
                lon_sides[1][1], lat_sides[1][1]):
            # going counter-clockwise
            lon_sides, lat_sides = self._reverse_boundaries(lon_sides, lat_sides)
        return lon_sides, lat_sides

    def _get_sides(self, coord_fun, vertices_per_side) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return the boundary sides."""
        top_slice, right_slice, bottom_slice, left_slice = self._get_bbox_slices(vertices_per_side)
        top_dim1, top_dim2 = coord_fun(data_slice=top_slice)
        right_dim1, right_dim2 = coord_fun(data_slice=right_slice)
        bottom_dim1, bottom_dim2 = coord_fun(data_slice=bottom_slice)
        left_dim1, left_dim2 = coord_fun(data_slice=left_slice)
        sides_dim1, sides_dim2 = zip(*[(top_dim1.squeeze(), top_dim2.squeeze()),
                                       (right_dim1.squeeze(), right_dim2.squeeze()),
                                       (bottom_dim1.squeeze(), bottom_dim2.squeeze()),
                                       (left_dim1.squeeze(), left_dim2.squeeze())])
        if hasattr(sides_dim1[0], 'compute') and da is not None:
            sides_dim1, sides_dim2 = da.compute(sides_dim1, sides_dim2)
        return self._filter_sides_nans(sides_dim1, sides_dim2)

    def _filter_sides_nans(
            self,
            dim1_sides: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            dim2_sides: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Remove nan and inf values present in each side."""
        new_dim1_sides = []
        new_dim2_sides = []
        for dim1_side, dim2_side in zip(dim1_sides, dim2_sides):
            # FIXME: ~(~np.isfinite(dim1_side) | ~np.isfinite(dim1_side))
            is_valid_mask = ~(np.isnan(dim1_side) | np.isnan(dim2_side))
            if not is_valid_mask.any():
                raise ValueError("Can't compute boundary coordinates. At least one side is completely invalid.")
            new_dim1_sides.append(dim1_side[is_valid_mask])
            new_dim2_sides.append(dim2_side[is_valid_mask])
        return new_dim1_sides, new_dim2_sides

    def _get_bbox_slices(self, vertices_per_side):
        # FIXME: This currently replicate values if heigh/width < row_num/col_num !
        height, width = self.shape
        if vertices_per_side is None:
            row_num = height
            col_num = width
        else:
            row_num = vertices_per_side
            col_num = vertices_per_side
        s1_slice = (0, np.linspace(0, width - 1, col_num, dtype=int))
        s2_slice = (np.linspace(0, height - 1, row_num, dtype=int), -1)
        s3_slice = (-1, np.linspace(width - 1, 0, col_num, dtype=int))
        s4_slice = (np.linspace(height - 1, 0, row_num, dtype=int), 0)
        return s1_slice, s2_slice, s3_slice, s4_slice

    @staticmethod
    def _reverse_boundaries(sides_lons: list, sides_lats: list) -> tuple:
        """Reverse the order of the lists and the arrays in those lists.

        Given lists of 4 numpy arrays, this will reverse the order of the
        arrays in that list and the elements of each of those arrays. This
        has the end result when the coordinates are counter-clockwise of
        reversing the coordinates to make them clockwise.

        """
        lons = [lon[::-1] for lon in sides_lons[::-1]]
        lats = [lat[::-1] for lat in sides_lats[::-1]]
        return lons, lats

    @staticmethod
    def _corner_is_clockwise(lon1, lat1, corner_lon, corner_lat, lon2, lat2):
        """Determine if coordinates follow a clockwise path.

        This uses :class:`pyresample.spherical.Arc` to determine the angle
        between the first line segment (Arc) from (lon1, lat1) to
        (corner_lon, corner_lat) and the second line segment from
        (corner_lon, corner_lat) to (lon2, lat2). A straight line would
        produce an angle of 0, a clockwise path would have a negative angle,
        and a counter-clockwise path would have a positive angle.

        """
        from pyresample.spherical import Arc, SCoordinate
        point1 = SCoordinate(math.radians(lon1), math.radians(lat1))
        point2 = SCoordinate(math.radians(corner_lon), math.radians(corner_lat))
        point3 = SCoordinate(math.radians(lon2), math.radians(lat2))
        arc1 = Arc(point1, point2)
        arc2 = Arc(point2, point3)
        angle = arc1.angle(arc2)
        is_clockwise = -np.pi < angle < 0
        return is_clockwise

    def get_edge_lonlats(self, vertices_per_side=None, frequency=None):
        """Get the concatenated boundary of the current swath."""
        if frequency is not None:
            warnings.warn("The `frequency` argument is pending deprecation, use `vertices_per_side` instead",
                          PendingDeprecationWarning, stacklevel=2)
        vertices_per_side = vertices_per_side or frequency
        lons, lats = self.get_bbox_lonlats(vertices_per_side=vertices_per_side, force_clockwise=False)
        blons = np.ma.concatenate(lons)
        blats = np.ma.concatenate(lats)
        return blons, blats

    def boundary(self, vertices_per_side=None, force_clockwise=False, frequency=None):
        """Retrieve the AreaBoundary object.

        Parameters
        ----------
        vertices_per_side:
             (formerly `frequency`) The number of points to provide for each side. By default (None)
            the full width and height will be provided.
        force_clockwise:
            Perform minimal checks and reordering of coordinates to ensure
            that the returned coordinates follow a clockwise direction.
            This is important for compatibility with
            :class:`pyresample.spherical.SphPolygon` where operations depend
            on knowing the inside versus the outside of a polygon. These
            operations assume that coordinates are clockwise.
            Default is False.
        """
        from pyresample.boundary import AreaBoundary
        if frequency is not None:
            warnings.warn("The `frequency` argument is pending deprecation, use `vertices_per_side` instead",
                          PendingDeprecationWarning, stacklevel=2)
        vertices_per_side = vertices_per_side or frequency
        # FIXME:
        # - Here return SphericalBoundary ensuring correct vertices ordering
        # - Deprecate get_bbox_lonlats and usage of force_clockwise
        lon_sides, lat_sides = self.get_bbox_lonlats(vertices_per_side=vertices_per_side,
                                                     force_clockwise=force_clockwise)
        return AreaBoundary.from_lonlat_sides(lon_sides, lat_sides)

    def get_cartesian_coords(self, nprocs=None, data_slice=None, cache=False):
        """Retrieve cartesian coordinates of geometry definition.

        Parameters
        ----------
        nprocs : int, optional
            Number of processor cores to be used.
            Defaults to the nprocs set when instantiating object
        data_slice : slice object, optional
            Calculate only cartesian coordnates for the defined slice
        cache : bool, optional
            Store result the result. Requires data_slice to be None

        Returns
        -------
        cartesian_coords : numpy array
        """
        if cache:
            warnings.warn("'cache' keyword argument will be removed in the "
                          "future and data will not be cached.", PendingDeprecationWarning, stacklevel=2)

        if self.cartesian_coords is None:
            # Coordinates are not cached
            if nprocs is None:
                nprocs = self.nprocs

            if data_slice is None:
                # Use full slice
                data_slice = slice(None)

            lons, lats = self.get_lonlats(nprocs=nprocs, data_slice=data_slice)

            if nprocs > 1:
                cartesian = Cartesian_MP(nprocs)
            else:
                cartesian = Cartesian()

            cartesian_coords = cartesian.transform_lonlats(np.ravel(lons), np.ravel(lats))
            if isinstance(lons, np.ndarray) and lons.ndim > 1:
                # Reshape to correct shape
                cartesian_coords = cartesian_coords.reshape(lons.shape[0], lons.shape[1], 3)

            if cache and data_slice is None:
                self.cartesian_coords = cartesian_coords
        else:
            # Coordinates are cached
            if data_slice is None:
                cartesian_coords = self.cartesian_coords
            else:
                cartesian_coords = self.cartesian_coords[data_slice]

        return cartesian_coords

    @property
    def corners(self):
        """Return the corners centroids of the current area."""
        from pyresample.spherical_geometry import Coordinate
        return [Coordinate(*self.get_lonlat(0, 0)),
                Coordinate(*self.get_lonlat(0, -1)),
                Coordinate(*self.get_lonlat(-1, -1)),
                Coordinate(*self.get_lonlat(-1, 0))]

    def __contains__(self, point):
        """Check if a point is inside the 4 corners of the current area.

        This uses great circle arcs as area boundaries.
        """
        from pyresample.spherical_geometry import Coordinate, point_inside
        corners = self.corners

        if isinstance(point, tuple):
            return point_inside(Coordinate(*point), corners)
        else:
            return point_inside(point, corners)

    def overlaps(self, other):
        """Test if the current area overlaps the *other* area.

        This is based solely on the corners of areas, assuming the
        boundaries to be great circles.

        Parameters
        ----------
        other : object
            Instance of subclass of BaseDefinition

        Returns
        -------
        overlaps : bool
        """
        from pyresample.spherical_geometry import Arc

        self_corners = self.corners

        other_corners = other.corners

        for i in self_corners:
            if i in other:
                return True
        for i in other_corners:
            if i in self:
                return True

        self_arc1 = Arc(self_corners[0], self_corners[1])
        self_arc2 = Arc(self_corners[1], self_corners[2])
        self_arc3 = Arc(self_corners[2], self_corners[3])
        self_arc4 = Arc(self_corners[3], self_corners[0])

        other_arc1 = Arc(other_corners[0], other_corners[1])
        other_arc2 = Arc(other_corners[1], other_corners[2])
        other_arc3 = Arc(other_corners[2], other_corners[3])
        other_arc4 = Arc(other_corners[3], other_corners[0])

        for i in (self_arc1, self_arc2, self_arc3, self_arc4):
            for j in (other_arc1, other_arc2, other_arc3, other_arc4):
                if i.intersects(j):
                    return True
        return False

    def get_area(self):
        """Get the area of the convex area defined by the corners of the curren area."""
        from pyresample.spherical_geometry import get_polygon_area

        return get_polygon_area(self.corners)

    def intersection(self, other):
        """Return the corners of the intersection polygon of the current area with *other*.

        Parameters
        ----------
        other : object
            Instance of subclass of BaseDefinition

        Returns
        -------
        (corner1, corner2, corner3, corner4) : tuple of points
        """
        from pyresample.spherical_geometry import intersection_polygon
        return intersection_polygon(self.corners, other.corners)

    def overlap_rate(self, other):
        """Get how much the current area overlaps an *other* area.

        Parameters
        ----------
        other : object
            Instance of subclass of BaseDefinition

        Returns
        -------
        overlap_rate : float
        """
        from pyresample.spherical_geometry import get_polygon_area
        other_area = other.get_area()
        inter_area = get_polygon_area(self.intersection(other))
        return inter_area / other_area

    def get_area_slices(self, area_to_cover):
        """Compute the slice to read based on an `area_to_cover`."""
        raise NotImplementedError

    @property
    def is_geostationary(self):
        """Whether this geometry is in a geostationary satellite projection or not."""
        return False

    def _get_geographic_sides(self, vertices_per_side: Optional[int] = None) -> tuple:
        """Return the geographic boundary sides of the current area.

        Args:
            vertices_per_side:
                The number of points to provide for each side.
                By default (None) the full width and height will be provided.
                If the area object is an AreaDefinition with any corner out of the Earth disk
                (i.e. full disc geostationary area, Robinson projection, polar projections, ...)
                by default only 50 points are selected.
        """
        # FIXME: Add logic for out-of-earth disk projections
        if self.is_geostationary:
            return self._get_geostationary_boundary_sides(vertices_per_side=vertices_per_side, coordinates="geographic")
        sides_lons, sides_lats = self._get_sides(coord_fun=self.get_lonlats, vertices_per_side=vertices_per_side)
        return sides_lons, sides_lats

    def _get_geostationary_boundary_sides(self, vertices_per_side, coordinates):
        class_name = self.__class__.__name__
        raise NotImplementedError(f"'_get_geostationary_boundary_sides' is not implemented for {class_name}")


class CoordinateDefinition(BaseDefinition):
    """Base class for geometry definitions defined by lons and lats only."""

    def __init__(self, lons, lats, nprocs=1):
        """Initialize CoordinateDefinition."""
        if not isinstance(lons, (np.ndarray, DataArray)):
            lons = np.asanyarray(lons)
            lats = np.asanyarray(lats)
        super().__init__(lons, lats, nprocs)
        if lons.shape == lats.shape and lons.dtype == lats.dtype:
            self.shape = lons.shape
            self.size = lons.size
            self.ndim = lons.ndim
            self.dtype = lons.dtype
        else:
            raise ValueError(('%s must be created with either '
                              'lon/lats of the same shape with same dtype') %
                             self.__class__.__name__)

    def concatenate(self, other):
        """Concatenate coordinate definitions."""
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to concatenate %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        klass = _get_highest_level_class(self, other)
        lons = np.concatenate((self.lons, other.lons))
        lats = np.concatenate((self.lats, other.lats))
        nprocs = min(self.nprocs, other.nprocs)
        return klass(lons, lats, nprocs=nprocs)

    def append(self, other):
        """Append another coordinate definition to existing one."""
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to append %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        self.lons = np.concatenate((self.lons, other.lons))
        self.lats = np.concatenate((self.lats, other.lats))
        self.shape = self.lons.shape
        self.size = self.lons.size

    def __str__(self):
        """Return string representation of the coordinate definition."""
        # Rely on numpy's object printing
        return ('Shape: %s\nLons: %s\nLats: %s') % (str(self.shape),
                                                    str(self.lons),
                                                    str(self.lats))

    __repr__ = __str__

    def geocentric_resolution(self, ellps='WGS84', radius=None, nadir_factor=2):
        """Calculate maximum geocentric pixel resolution.

        If `lons` is a :class:`xarray.DataArray` object with a `resolution`
        attribute, this will be used instead of loading the longitude and
        latitude data. In this case the resolution attribute is assumed to
        mean the nadir resolution of a swath and will be multiplied by the
        `nadir_factor` to adjust for increases in the spatial resolution
        towards the limb of the swath.

        Args:
            ellps (str): PROJ Ellipsoid for the Cartographic projection
                used as the target geocentric coordinate reference system.
                Default: 'WGS84'. Ignored if `radius` is provided.
            radius (float): Spherical radius of the Earth to use instead of
                the definitions in `ellps`.
            nadir_factor (int): Number to multiply the nadir resolution
                attribute by to reflect pixel size on the limb of the swath.

        Returns: Estimated maximum pixel size in meters on a geocentric
            coordinate system (X, Y, Z) representing the Earth.

        Raises: RuntimeError if a simple search for valid longitude/latitude
            data points found no valid data points.

        """
        if hasattr(self.lons, 'attrs') and \
                self.lons.attrs.get('resolution') is not None:
            return self.lons.attrs['resolution'] * nadir_factor
        if self.ndim == 1:
            raise RuntimeError("Can't confidently determine geocentric "
                               "resolution for 1D swath.")
        rows = self.shape[0]
        start_row = rows // 2  # middle row
        src = CRS('+proj=latlong +datum=WGS84')
        if radius:
            dst = CRS("+proj=cart +a={} +b={}".format(radius, radius))
        else:
            dst = CRS("+proj=cart +ellps={}".format(ellps))
        # simply take the first two columns of the middle of the swath
        lons = self.lons[start_row: start_row + 1, :2]
        lats = self.lats[start_row: start_row + 1, :2]
        if hasattr(lons.data, 'compute'):
            # dask arrays, compute them together
            import dask.array as da
            lons, lats = da.compute(lons, lats)
        if hasattr(lons, 'values'):
            # convert xarray to numpy array
            lons = lons.values
            lats = lats.values
        lons = lons.ravel()
        lats = lats.ravel()
        alt = np.zeros_like(lons)

        transformer = pyproj.Transformer.from_crs(src, dst)
        xyz = np.stack(transformer.transform(lons, lats, alt), axis=1)
        dist = np.linalg.norm(xyz[1] - xyz[0])
        dist = dist[np.isfinite(dist)]
        if not dist.size:
            raise RuntimeError("Could not calculate geocentric resolution")
        return dist[0]


class GridDefinition(CoordinateDefinition):
    """Grid defined by lons and lats.

    Parameters
    ----------
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.

    Attributes
    ----------
    shape : tuple
        Grid shape as (rows, cols)
    size : int
        Number of elements in grid
    lons : object
        Grid lons
    lats : object
        Grid lats
    cartesian_coords : object
        Grid cartesian coordinates
    """

    def __init__(self, lons, lats, nprocs=1):
        """Initialize GridDefinition."""
        super().__init__(lons, lats, nprocs)
        if lons.shape != lats.shape:
            raise ValueError('lon and lat grid must have same shape')
        elif lons.ndim != 2:
            raise ValueError('2 dimensional lon lat grid expected')


def get_array_hashable(arr):
    """Compute a hashable form of the array `arr`.

    Works with numpy arrays, dask.array.Array, and xarray.DataArray.
    """
    # look for precomputed value
    if isinstance(arr, DataArray) and np.ndarray is not DataArray:
        return arr.attrs.get('hash', get_array_hashable(arr.data))
    else:
        try:
            return arr.name.encode('utf-8')  # dask array
        except AttributeError:
            return np.ascontiguousarray(arr).view(np.uint8)  # np array


class SwathDefinition(CoordinateDefinition):
    """Swath defined by lons and lats.

    Parameters
    ----------
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.
    crs: pyproj.CRS,
       The CRS to use. longlat on WGS84 by default.

    Attributes
    ----------
    shape : tuple
        Swath shape
    size : int
        Number of elements in swath
    ndims : int
        Swath dimensions
    lons : object
        Swath lons
    lats : object
        Swath lats
    cartesian_coords : object
        Swath cartesian coordinates
    """

    def __init__(self, lons, lats, nprocs=1, crs=None):
        """Initialize SwathDefinition."""
        if not isinstance(lons, (np.ndarray, DataArray)):
            lons = np.asanyarray(lons)
            lats = np.asanyarray(lats)
        super().__init__(lons, lats, nprocs)
        if lons.shape != lats.shape:
            raise ValueError('lon and lat arrays must have same shape')
        elif lons.ndim > 2:
            raise ValueError('Only 1 and 2 dimensional swaths are allowed')
        self.crs = crs or CRS(proj="longlat", ellps="WGS84")

    def copy(self):
        """Copy the current swath."""
        return SwathDefinition(self.lons, self.lats)

    @staticmethod
    def _do_transform(src, dst, lons, lats, alt):
        """Run pyproj.transform and stack the results."""
        transformer = pyproj.Transformer.from_crs(src, dst)
        x, y, z = transformer.transform(lons, lats, alt)
        return np.dstack((x, y, z))

    def aggregate(self, **dims):
        """Aggregate the current swath definition by averaging.

        For example, averaging over 2x2 windows:
        `sd.aggregate(x=2, y=2)`
        """
        import dask.array as da
        import pyproj

        geocent = pyproj.CRS(proj='geocent')
        latlong = pyproj.CRS(proj='latlong')
        res = da.map_blocks(self._do_transform, latlong, geocent,
                            self.lons.data, self.lats.data,
                            da.zeros_like(self.lons.data), new_axis=[2],
                            meta=np.array((), dtype=self.lons.dtype),
                            dtype=self.lons.dtype,
                            chunks=(self.lons.chunks[0], self.lons.chunks[1], 3))
        res = DataArray(res, dims=['y', 'x', 'coord'], coords=self.lons.coords)
        res = res.coarsen(**dims).mean()
        lonlatalt = da.map_blocks(self._do_transform, geocent, latlong,
                                  res[:, :, 0].data, res[:, :, 1].data,
                                  res[:, :, 2].data, new_axis=[2],
                                  meta=np.array((), dtype=res.dtype),
                                  dtype=res.dtype,
                                  chunks=res.data.chunks)
        lons = DataArray(lonlatalt[:, :, 0], dims=self.lons.dims,
                         coords=res.coords, attrs=self.lons.attrs.copy())
        lats = DataArray(lonlatalt[:, :, 1], dims=self.lons.dims,
                         coords=res.coords, attrs=self.lons.attrs.copy())
        try:
            resolution = lons.attrs['resolution'] * ((dims.get('x', 1) + dims.get('y', 1)) / 2)
            lons.attrs['resolution'] = resolution
            lats.attrs['resolution'] = resolution
        except KeyError:
            pass
        return SwathDefinition(lons, lats)

    def __hash__(self):
        """Compute the hash of this object."""
        if self.hash is None:
            self.hash = int(self.update_hash().hexdigest(), 16)
        return self.hash

    def _repr_html_(self):
        """Html representation."""
        return _formatting_html.area_repr(self)

    def _compute_omerc_parameters(self, ellipsoid):
        """Compute the oblique mercator projection bouding box parameters."""
        lines, cols = self.lons.shape
        lon1, lon2 = np.asanyarray(self.lons[[0, -1], int(cols / 2)])
        lat1, lat, lat2 = np.asanyarray(
            self.lats[[0, int(lines / 2), -1], int(cols / 2)])
        if any(np.isnan((lon1, lon2, lat1, lat, lat2))):
            thelons = self.lons[:, int(cols / 2)]
            thelons = thelons.where(thelons.notnull(), drop=True)
            thelats = self.lats[:, int(cols / 2)]
            thelats = thelats.where(thelats.notnull(), drop=True)
            lon1, lon2 = np.asanyarray(thelons[[0, -1]])
            lines = len(thelats)
            lat1, lat, lat2 = np.asanyarray(thelats[[0, int(lines / 2), -1]])

        proj_dict2points = {'proj': 'omerc', 'lat_0': lat, 'ellps': ellipsoid,
                            'lat_1': lat1, 'lon_1': lon1,
                            'lat_2': lat2, 'lon_2': lon2,
                            'no_rot': True
                            }

        # We need to compute alpha-based omerc for geotiff support
        lonc, lat0 = Proj(**proj_dict2points)(0, 0, inverse=True)
        az1, az2, _ = Geod(**proj_dict2points).inv(lonc, lat0, lon2, lat2)
        azimuth = az1
        az1, az2, _ = Geod(**proj_dict2points).inv(lonc, lat0, lon1, lat1)
        if abs(az1 - azimuth) > 1:
            if abs(az2 - azimuth) > 1:
                logger.warning("Can't find appropriate azimuth.")
            else:
                azimuth += az2
                azimuth /= 2
        else:
            azimuth += az1
            azimuth /= 2
        if abs(azimuth) > 90:
            azimuth = 180 + azimuth

        prj_params = {'proj': 'omerc', 'alpha': float(azimuth), 'lat_0': float(lat0), 'lonc': float(lonc),
                      'gamma': 0,
                      'ellps': ellipsoid}

        return prj_params

    def _compute_generic_parameters(self, projection, ellipsoid):
        """Compute the projection bb parameters for most projections."""
        lines, cols = self.lons.shape
        lat_0 = self.lats[int(lines / 2), int(cols / 2)]
        lon_0 = self.lons[int(lines / 2), int(cols / 2)]
        return {'proj': projection, 'ellps': ellipsoid,
                'lat_0': lat_0, 'lon_0': lon_0}

    def compute_bb_proj_params(self, proj_dict):
        """Compute BB projection parameters."""
        projection = proj_dict['proj']
        if projection == 'omerc':
            ellipsoid = proj_dict.get('ellps', 'sphere')
            return self._compute_omerc_parameters(ellipsoid)
        else:
            ellipsoid = proj_dict.get('ellps', 'WGS84')
            new_proj = self._compute_generic_parameters(projection, ellipsoid)
            new_proj.update(proj_dict)
            return new_proj

    def _compute_uniform_shape(self, resolution=None):
        """Compute the height and width of a domain to have uniform resolution across dimensions."""
        g = Geod(ellps='WGS84')

        def notnull(arr):
            try:
                return arr.where(arr.notnull(), drop=True)
            except AttributeError:
                return arr[np.isfinite(arr)]
        leftlons = self.lons[:, 0]
        rightlons = self.lons[:, -1]
        middlelons = self.lons[:, int(self.lons.shape[1] / 2)]
        leftlats = self.lats[:, 0]
        rightlats = self.lats[:, -1]
        middlelats = self.lats[:, int(self.lats.shape[1] / 2)]
        try:
            import dask.array as da
        except ImportError:
            pass
        else:
            leftlons, rightlons, middlelons, leftlats, rightlats, middlelats = da.compute(leftlons, rightlons,
                                                                                          middlelons, leftlats,
                                                                                          rightlats, middlelats)
        leftlons = notnull(leftlons)
        rightlons = notnull(rightlons)
        middlelons = notnull(middlelons)
        leftlats = notnull(leftlats)
        rightlats = notnull(rightlats)
        middlelats = notnull(middlelats)

        az1, az2, width1 = g.inv(leftlons[0], leftlats[0], rightlons[0], rightlats[0])
        az1, az2, width2 = g.inv(leftlons[-1], leftlats[-1], rightlons[-1], rightlats[-1])
        az1, az2, height = g.inv(middlelons[0], middlelats[0], middlelons[-1], middlelats[-1])
        width = min(width1, width2)
        if resolution is None:
            vresolution = height * 1.0 / self.lons.shape[0]
            hresolution = width * 1.0 / self.lons.shape[1]
            resolution = (vresolution, hresolution)
        if isinstance(resolution, (tuple, list)):
            resolution = min(*resolution)
        width = int(width * 1.1 / resolution)
        height = int(height * 1.1 / resolution)
        return height, width

    def compute_optimal_bb_area(self, proj_dict=None, resolution=None):
        """Compute the "best" bounding box area for this swath with `proj_dict`.

        By default, the projection is Oblique Mercator (`omerc` in proj.4), in
        which case the right projection angle `alpha` is computed from the
        swath centerline. For other projections, only the appropriate center of
        projection and area extents are computed.

        The height and width are computed so that the resolution is
        approximately the same across dimensions.
        """
        if proj_dict is None:
            proj_dict = {}
        projection = proj_dict.setdefault('proj', 'omerc')
        area_id = projection + '_otf'
        description = 'On-the-fly ' + projection + ' area'
        height, width = self._compute_uniform_shape(resolution)
        proj_dict = self.compute_bb_proj_params(proj_dict)

        area = DynamicAreaDefinition(area_id, description, proj_dict)
        lons, lats = self.get_edge_lonlats()
        return area.freeze((lons, lats), shape=(height, width))


class DynamicAreaDefinition(object):
    """An AreaDefintion containing just a subset of the needed parameters.

    The purpose of this class is to be able to adapt the area extent and shape
    of the area to a given set of longitudes and latitudes, such that e.g.
    polar satellite granules can be resampled optimally to a given projection.

    Note that if the provided projection is geographic (lon/lat degrees) and
    the provided longitude and latitude data crosses the anti-meridian
    (-180/180), the resulting area will be the smallest possible in order to
    contain that data and avoid a large area spanning from -180 to 180
    longitude. This means the resulting AreaDefinition will have a right-most
    X extent greater than 180 degrees. This does not apply to data crossing
    the north or south pole as there is no "smallest" area in this case.

    Attributes:
        area_id:
            The name of the area.
        description:
            The description of the area.
        projection:
            The dictionary or string or CRS object of projection parameters.
            Doesn't have to be complete. If not complete, ``proj_info`` must
            be provided to ``freeze`` to "fill in" any missing parameters.
        width:
            x dimension in number of pixels, aka number of grid columns
        height:
            y dimension in number of pixels, aka number of grid rows
        shape:
            Corresponding array shape as (height, width)
        area_extent:
            The area extent of the area.
        resolution:
            Resolution of the resulting area as (pixel_size_x, pixel_size_y)
            or a scalar if pixel_size_x == pixel_size_y.
        optimize_projection:
            Whether the projection parameters have to be optimized.

    """

    def __init__(self, area_id=None, description=None, projection=None,
                 width=None, height=None, area_extent=None,
                 resolution=None, optimize_projection=False):
        """Initialize the DynamicAreaDefinition."""
        self.area_id = area_id
        self.description = description
        self.width = width
        self.height = height
        self.shape = (self.height, self.width)
        self.area_extent = area_extent
        self.optimize_projection = optimize_projection
        if isinstance(resolution, (int, float)):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self._projection = projection

        # check if non-dict projections are valid
        # dicts may be updated later
        if not isinstance(self._projection, dict):
            CRS(projection)

    def _get_proj_dict(self):
        projection = self._projection
        try:
            crs = CRS(projection)
        except RuntimeError:
            # could be incomplete dictionary
            return projection
        return crs.to_dict()

    @property
    def pixel_size_x(self):
        """Pixel width in projection units."""
        if self.resolution is None:
            return None
        return self.resolution[0]

    @property
    def pixel_size_y(self):
        """Pixel height in projection units."""
        if self.resolution is None:
            return None
        return self.resolution[1]

    def compute_domain(
            self,
            corners: Sequence,
            resolution: Optional[Union[float, tuple[float, float]]] = None,
            shape: Optional[tuple[int, int]] = None,
            projection: Optional[Union[CRS, dict, str, int]] = None
    ):
        """Compute shape and area_extent from corners and [shape or resolution] info.

        Args:
            corners:
                4-element sequence representing the outer corners of the
                region. Note that corners represents the center of pixels,
                while area_extent represents the edge of pixels. The four
                values are (xmin_corner, ymin_corner, xmax_corner, ymax_corner).
                If the x corners are ``None`` then the full extent (area of use)
                of the projection will be used. When needed, area of use is taken
                from the PROJ library or in the case of a geographic lon/lat
                projection -180/180 is used. A RuntimeError is raised if the
                area of use is needed (when x corners are ``None``) and area
                of use can't be determined.
            resolution:
                Spatial resolution in projection units (typically meters or
                degrees). If not specified then shape must be provided.
                If a scalar then it is treated as the x and y resolution. If
                a tuple then x resolution is the first element, y is the
                second.
            shape:
                Number of pixels in the area as a 2-element tuple. The first
                is number of rows, the second number of columns.
            projection:
                PROJ.4 definition string, dictionary, integer EPSG code, or
                pyproj CRS object.

        Note that ``shape`` is (rows, columns) and ``resolution`` is
        (x_size, y_size); the dimensions are flipped.

        """
        if resolution is not None and shape is not None:
            raise ValueError("Both resolution and shape can't be provided.")
        elif resolution is None and shape is None:
            raise ValueError("Either resolution or shape must be provided.")
        if resolution is not None and isinstance(resolution, (int, float)):
            resolution = (resolution, resolution)
        if projection is None:
            projection = self._projection

        corners = self._update_corners_for_full_extent(corners, shape, resolution, projection)
        if shape:
            height, width = shape
            x_resolution = (corners[2] - corners[0]) * 1.0 / (width - 1)
            y_resolution = (corners[3] - corners[1]) * 1.0 / (height - 1)
            area_extent = (corners[0] - x_resolution / 2,
                           corners[1] - y_resolution / 2,
                           corners[2] + x_resolution / 2,
                           corners[3] + y_resolution / 2)
        elif resolution:
            x_resolution, y_resolution = resolution
            half_x = x_resolution / 2
            half_y = y_resolution / 2
            # align extents with pixel resolution
            area_extent = (
                math.floor((corners[0] - half_x) / x_resolution) * x_resolution,
                math.floor((corners[1] - half_y) / y_resolution) * y_resolution,
                math.ceil((corners[2] + half_x) / x_resolution) * x_resolution,
                math.ceil((corners[3] + half_y) / y_resolution) * y_resolution,
            )
            width = int(round((area_extent[2] - area_extent[0]) / x_resolution))
            height = int(round((area_extent[3] - area_extent[1]) / y_resolution))

        return area_extent, width, height

    def _update_corners_for_full_extent(self, corners, shape, resolution, projection):
        corners = list(corners)
        if corners[0] is not None:
            return corners
        aou = self._get_crs_area_of_use(projection)
        if shape is not None:
            width = shape[1]
            x_resolution = (aou.east - aou.west) / width
            corners[0] = aou.west + x_resolution / 2.0
            corners[2] = aou.east - x_resolution / 2.0
        else:
            x_resolution = resolution[0]
            corners[0] = aou.west + x_resolution / 2.0
            corners[2] = aou.east - x_resolution / 2.0
        return corners

    def _get_crs_area_of_use(self, projection):
        crs = CRS(projection)
        aou = crs.area_of_use
        if aou is None:
            if crs.is_geographic:
                return AreaOfUse(west=-180.0, south=-90.0, east=180.0, north=90.0)
            raise RuntimeError("Projection has no defined area of use")
        return aou

    def freeze(self, lonslats=None, resolution=None, shape=None, proj_info=None,
               antimeridian_mode=None):
        """Create an AreaDefinition from this area with help of some extra info.

        Parameters
        ----------
        lonlats : SwathDefinition or tuple
          The geographical coordinates to contain in the resulting area.
          A tuple should be ``(lons, lats)``.
        resolution:
          the resolution of the resulting area.
        shape:
          the shape of the resulting area.
        proj_info:
          complementing parameters to the projection info.
        antimeridian_mode:
            How to handle lon/lat data crossing the anti-meridian of the
            projection. This currently only affects lon/lat geographic
            projections and data cases not covering the north or south pole.
            The possible options are:

            * "modify_extents": Set the X bounds to the edges of the data, but
                add 360 to the right-most bound. This has the effect of making
                the area coordinates continuous from the left side to the
                right side. However, this means that some coordinates will be
                outside the coordinate space of the projection. Although most
                PROJ and pyresample functionality can handle this there may be
                some edge cases.
            * "modify_crs": Change the prime meridian of the projection
                from 0 degrees longitude to 180 degrees longitude. This has
                the effect of putting the data on a continuous coordinate
                system. However, this means that comparing data resampled to
                this resulting area and an area not over the anti-meridian
                would be more difficult.
            * "global_extents": Ignore the bounds of the data and use -180/180
                degrees as the west and east bounds of the data. This will
                generate a large output area, but with the benefit of keeping
                the data on the original projection. Note that some resampling
                methods may produce artifacts when resampling on the edge of
                the area (the anti-meridian).

        Shape parameters are ignored if the instance is created
        with the `optimize_projection` flag set to True.
        """
        with ignore_pyproj_proj_warnings():
            proj_dict = self._get_proj_dict()
        projection = self._projection
        if proj_info is not None:
            # this is now our complete projection information
            proj_dict.update(proj_info)
            projection = proj_dict

        if self.optimize_projection:
            return lonslats.compute_optimal_bb_area(proj_dict, resolution=resolution or self.resolution)
        if resolution is None:
            resolution = self.resolution
        if shape is None:
            shape = self.shape
        height, width = shape
        shape = None if None in shape else shape
        area_extent = self.area_extent
        if not area_extent or not width or not height:
            projection, corners = self._compute_bound_centers(proj_dict, lonslats, antimeridian_mode=antimeridian_mode)
            area_extent, width, height = self.compute_domain(corners, resolution, shape, projection)
        return AreaDefinition(self.area_id, self.description, '',
                              projection, width, height,
                              area_extent)

    def _compute_bound_centers(self, proj_dict, lonslats, antimeridian_mode):
        from pyresample.utils.proj4 import DaskFriendlyTransformer

        lons, lats = self._extract_lons_lats(lonslats)
        crs = CRS(proj_dict)
        transformer = DaskFriendlyTransformer.from_crs(CRS(4326), crs, always_xy=True)
        xarr, yarr = transformer.transform(lons, lats)
        xarr[xarr > 9e29] = np.nan
        yarr[yarr > 9e29] = np.nan
        xmin = np.nanmin(xarr)
        xmax = np.nanmax(xarr)
        ymin = np.nanmin(yarr)
        ymax = np.nanmax(yarr)
        if hasattr(lons, "compute"):
            xmin, xmax, ymin, ymax = da.compute(xmin, xmax, ymin, ymax)
        x_passes_antimeridian = (xmax - xmin) > 355
        epsilon = 0.1
        y_is_pole = (ymax >= 90 - epsilon) or (ymin <= -90 + epsilon)
        if crs.is_geographic and x_passes_antimeridian and not y_is_pole:
            # cross anti-meridian of projection
            xmin, xmax = self._compute_new_x_corners_for_antimeridian(xarr, antimeridian_mode)
            if antimeridian_mode == "modify_crs":
                proj_dict.update({"pm": 180.0})
        return proj_dict, (xmin, ymin, xmax, ymax)

    @staticmethod
    def _extract_lons_lats(lonslats):
        try:
            lons, lats = lonslats
        except (TypeError, ValueError):
            lons, lats = lonslats.get_lonlats()
        return lons, lats

    def _compute_new_x_corners_for_antimeridian(self, xarr, antimeridian_mode):
        if antimeridian_mode == "global_extents":
            xmin, xmax = (None, None)
        else:
            wrapped_array = xarr % 360
            xmin = np.nanmin(wrapped_array)
            xmax = np.nanmax(wrapped_array)
            if hasattr(wrapped_array, "compute"):
                xmin, xmax = da.compute(xmin, xmax)
            if antimeridian_mode == "modify_crs":
                xmin -= 180
                xmax -= 180
        return xmin, xmax


def _invproj(data_x, data_y, proj_wkt):
    """Perform inverse projection."""
    # XXX: does pyproj copy arrays? What can we do so it doesn't?
    crs = CRS.from_wkt(proj_wkt)
    gcrs = get_geodetic_crs_with_no_datum_shift(crs)
    transformer = pyproj.Transformer.from_crs(gcrs, crs, always_xy=True)
    lon, lat = transformer.transform(data_x, data_y, direction=TransformDirection.INVERSE)
    return np.stack([lon.astype(data_x.dtype), lat.astype(data_y.dtype)])


def _generate_2d_coords(pixel_size_x, pixel_size_y, pixel_upper_left_x, pixel_upper_left_y,
                        chunks, dtype, block_info=None):
    start_y_idx = block_info[None]["array-location"][1][0]
    end_y_idx = block_info[None]["array-location"][1][1]
    start_x_idx = block_info[None]["array-location"][2][0]
    end_x_idx = block_info[None]["array-location"][2][1]
    dtype = block_info[None]["dtype"]
    x, y = _generate_1d_proj_vectors((start_x_idx, end_x_idx),
                                     (start_y_idx, end_y_idx),
                                     (pixel_size_x, pixel_size_y),
                                     (pixel_upper_left_x, pixel_upper_left_y),
                                     dtype)
    x_2d, y_2d = np.meshgrid(x, y)
    res = np.stack([x_2d, y_2d])
    return res


def _generate_1d_proj_vectors(col_range, row_range,
                              pixel_size_xy, offset_xy,
                              dtype, chunks=None):
    x_kwargs, y_kwargs, arange = _get_vector_arange_args(dtype, chunks)
    x = arange(*col_range, **x_kwargs) * pixel_size_xy[0] + offset_xy[0]
    y = arange(*row_range, **y_kwargs) * -pixel_size_xy[1] + offset_xy[1]
    return x, y


def _get_vector_arange_args(dtype, chunks):
    x_kwargs = {}
    y_kwargs = {}

    y_chunks, x_chunks = _chunks_to_yx_chunks(chunks)
    if x_chunks is not None or y_chunks is not None:
        # use dask functions instead of numpy
        from dask.array import arange
        x_kwargs = {'chunks': x_chunks}
        y_kwargs = {'chunks': y_chunks}
    else:
        arange = np.arange
    x_kwargs['dtype'] = dtype
    y_kwargs['dtype'] = dtype
    return x_kwargs, y_kwargs, arange


def _chunks_to_yx_chunks(chunks):
    if chunks is not None and not isinstance(chunks, int):
        y_chunks = chunks[0]
        x_chunks = chunks[1]
    else:
        y_chunks = x_chunks = chunks
    return y_chunks, x_chunks


class _ProjectionDefinition(BaseDefinition):
    """Base class for definitions based on CRS and area extents."""

    @property
    def x_size(self):
        """Return area width."""
        warnings.warn("'x_size' is deprecated, use 'width' instead.", PendingDeprecationWarning, stacklevel=2)
        return self.width

    @property
    def y_size(self):
        """Return area height."""
        warnings.warn("'y_size' is deprecated, use 'height' instead.", PendingDeprecationWarning, stacklevel=2)
        return self.height

    @property
    def crs(self):
        """Wrap the `crs` property in a helper property.

        The :class:`pyproj.crs.CRS` object is not thread-safe. To avoid
        accidentally passing it between threads, we only create it when it
        is requested (the `self.crs` property). The alternative of storing it
        as a normal instance attribute could cause issues between threads.

        """
        return CRS.from_wkt(self.crs_wkt)

    @property
    def proj_dict(self):
        """Return the PROJ projection dictionary.

        This is no longer the preferred way of describing CRS information.
        Switch to the `crs` or `crs_wkt` properties for the most flexibility.
        """
        return self.crs.to_dict()

    @property
    def size(self):
        """Return size of the definition."""
        return self.height * self.width

    @property
    def shape(self):
        """Return shape of the definition."""
        return (self.height, self.width)


def masked_ints(func):
    """Return masked integer arrays when returning array indices."""
    @wraps(func)
    def wrapper(self, xm, ym):
        is_scalar = np.isscalar(xm) and np.isscalar(ym)

        x__, y__ = func(self, xm, ym)
        epsilon = 0.02  # arbitrary buffer for floating point precision
        x_mask = ((x__ < -0.5 - epsilon) | (x__ > self.width - 0.5 + epsilon))
        y_mask = ((y__ < -0.5 - epsilon) | (y__ > self.height - 0.5 + epsilon))
        x__ = np.clip(x__, 0, self.width - 1)
        y__ = np.clip(y__, 0, self.height - 1)
        x__ = np.round(x__).astype(int)
        y__ = np.round(y__).astype(int)

        x_masked = np.ma.masked_array(x__, mask=x_mask, copy=False)
        y_masked = np.ma.masked_array(y__, mask=y_mask, copy=False)
        if is_scalar:
            if x_masked.all() is np.ma.masked or y_masked.all() is np.ma.masked:
                raise ValueError('Point outside area:( %f %f)' % (x__, y__))
            return x__.item(), y__.item()

        else:
            return x_masked, y_masked
    return wrapper


def preserve_scalars(func):
    """Preserve scalars through the coordinate conversion functions."""
    @wraps(func)
    def wrapper(self, xm, ym):
        x__, y__ = func(self, xm, ym)
        try:
            return x__.item(), y__.item()
        except ValueError:
            return x__, y__

    return wrapper


def daskify_2in_2out(func):
    """Daskify the coordinate conversion functions."""
    @wraps(func)
    def wrapper(self, coord1, coord2):
        if da is None or not (isinstance(coord1, da.Array) or isinstance(coord2, da.Array)):
            return func(self, coord1, coord2)
        newfunc = partial(func, self)
        dims = '(' + ', '.join('i_' + str(i) for i in range(coord1.ndim)) + ')'
        signature = dims + ', ' + dims + '->' + dims + ', ' + dims
        return da.apply_gufunc(newfunc, signature, coord1, coord2, output_dtypes=(float, float))

    return wrapper


class AreaDefinition(_ProjectionDefinition):
    """Holds definition of an area.

    Parameters
    ----------
    area_id : str
        Identifier for the area
    description : str
        Human-readable description of the area
    proj_id : str
        ID of projection
    projection: dict or str or pyproj.crs.CRS
        Dictionary of PROJ parameters or string of PROJ or WKT parameters.
        Can also be a :class:`pyproj.crs.CRS` object.
    width : int
        x dimension in number of pixels, aka number of grid columns
    height : int
        y dimension in number of pixels, aka number of grid rows
    area_extent : list
        Area extent as a list (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    nprocs : int, optional
        Number of processor cores to be used for certain calculations

    Attributes
    ----------
    area_id : str
        Identifier for the area
    description : str
        Human-readable description of the area
    proj_id : str
        ID of projection
    projection : dict or str
        Dictionary or string with Proj.4 parameters
    width : int
        x dimension in number of pixels, aka number of grid columns
    height : int
        y dimension in number of pixels, aka number of grid rows
    size : int
        Number of points in grid
    area_extent_ll : tuple
        Area extent in lons lats as a tuple (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
    pixel_size_x : float
        Pixel width in projection units
    pixel_size_y : float
        Pixel height in projection units
    upper_left_extent : tuple
        Coordinates (x, y) of upper left corner of upper left pixel in projection units
    pixel_upper_left : tuple
        Coordinates (x, y) of center of upper left pixel in projection units
    pixel_offset_x : float
        x offset between projection center and upper left corner of upper
        left pixel in units of pixels.
    pixel_offset_y : float
        y offset between projection center and upper left corner of upper
        left pixel in units of pixels..
    crs : pyproj.crs.CRS
        Coordinate reference system object similar to the PROJ parameters in
        `proj_dict` and `proj_str`. This is the preferred attribute to use
        when working with the `pyproj` library. Note, however, that this
        object is not thread-safe and should not be passed between threads.
    crs_wkt : str
        WellKnownText version of the CRS object. This is the preferred
        way of describing CRS information as a string.
    cartesian_coords : object
        Grid cartesian coordinates
    """

    def __init__(self, area_id, description, proj_id, projection, width, height,
                 area_extent, nprocs=1, lons=None, lats=None,
                 dtype=np.float64):
        """Initialize AreaDefinition."""
        super().__init__(lons, lats, nprocs)
        self.area_id = area_id
        self.description = description
        self.proj_id = proj_id
        self.width = int(width)
        self.height = int(height)
        self.crop_offset = (0, 0)
        if lons is not None:
            if lons.shape != self.shape:
                raise ValueError('Shape of lon lat grid must match '
                                 'area definition')
        self.ndim = 2
        self.pixel_size_x = (area_extent[2] - area_extent[0]) / float(width)
        self.pixel_size_y = (area_extent[3] - area_extent[1]) / float(height)
        self._area_extent = tuple(area_extent)
        self.crs_wkt = CRS(projection).to_wkt()

        # Calculate area_extent in lon lat
        proj = Proj(projection)
        corner_lons, corner_lats = proj((area_extent[0], area_extent[2]),
                                        (area_extent[1], area_extent[3]),
                                        inverse=True)
        self.area_extent_ll = (corner_lons[0], corner_lats[0],
                               corner_lons[1], corner_lats[1])

        # Calculate projection coordinates of extent of upper left pixel
        self.upper_left_extent = (float(area_extent[0]), float(area_extent[3]))
        self.pixel_upper_left = (float(area_extent[0]) + float(self.pixel_size_x) / 2,
                                 float(area_extent[3]) - float(self.pixel_size_y) / 2)

        # Pixel_offset defines the distance to projection center from origin
        # (UL) of image in units of pixels.
        self.pixel_offset_x = -self.area_extent[0] / self.pixel_size_x
        self.pixel_offset_y = self.area_extent[3] / self.pixel_size_y

        self._projection_x_coords = None
        self._projection_y_coords = None

        self.dtype = dtype

    @property
    def is_geostationary(self):
        """Whether this area is in a geostationary satellite projection or not."""
        coord_operation = self.crs.coordinate_operation
        if coord_operation is None:
            return False
        return 'geostationary' in coord_operation.method_name.lower()

    def _get_geostationary_boundary_sides(self, vertices_per_side=None, coordinates="geographic"):
        """Retrieve the boundary sides list for geostationary projections with out-of-Earth disk coordinates.

        The boundary sides right (1) and side left (3) are set to length 2.
        """
        # FIXME:
        # - If vertices_per_side is too small, there is the risk to loose boundary side points
        #   at the intersection corners between the CRS bounds polygon and the area
        #   extent polygon (which could exclude relevant regions of the geos area).
        # - After fixing this, evaluate nb_points required for FULL DISC and CONUS area !
        # Define default frequency
        if vertices_per_side is None:
            vertices_per_side = 50
        # Ensure at least 4 points are used
        if vertices_per_side < 4:
            vertices_per_side = 4
        # Ensure an even number of vertices for side creation
        if (vertices_per_side % 2) != 0:
            vertices_per_side = vertices_per_side + 1
        # Retrieve coordinates (x,y) or (lon, lat)
        if coordinates == "geographic":
            x, y = get_geostationary_bounding_box_in_lonlats(self, nb_points=vertices_per_side)
        else:
            x, y = get_geostationary_bounding_box_in_proj_coords(self, nb_points=vertices_per_side)
        # Ensure that a portion of the area is within the Earth disk.
        if x.shape[0] < 4:
            raise ValueError("The geostationary projection area is entirely out of the Earth disk.")
        # Retrieve dummy sides for GEO
        # FIXME:
        # - _get_geostationary_bounding_box_* does not guarantee to return nb_points and even points!
        # - if odd nb_points, above can go out of index
        # --> sides_x = self._get_dummy_sides(x, vertices_per_side=vertices_per_side)
        # --> sides_y = self._get_dummy_sides(y, vertices_per_side=vertices_per_side)
        side02_step = int(vertices_per_side / 2) - 1
        sides_x = [
            x[slice(0, side02_step + 1)],
            x[slice(side02_step, side02_step + 1 + 1)],
            x[slice(side02_step + 1, side02_step * 2 + 1 + 1)],
            np.append(x[side02_step * 2 + 1], x[0])
        ]
        sides_y = [
            y[slice(0, side02_step + 1)],
            y[slice(side02_step, side02_step + 1 + 1)],
            y[slice(side02_step + 1, side02_step * 2 + 1 + 1)],
            np.append(y[side02_step * 2 + 1], y[0])
        ]
        return sides_x, sides_y

    def get_edge_bbox_in_projection_coordinates(self, vertices_per_side: Optional[int] = None,
                                                frequency: Optional[int] = None):
        """Return the bounding box in projection coordinates."""
        if frequency is not None:
            warnings.warn("The `frequency` argument is pending deprecation, use `vertices_per_side` instead",
                          PendingDeprecationWarning, stacklevel=2)
        vertices_per_side = vertices_per_side or frequency
        x, y = self._get_sides(self.get_proj_coords, vertices_per_side=vertices_per_side)
        return np.hstack(x), np.hstack(y)

    @property
    def area_extent(self):
        """Tuple of this area's extent (xmin, ymin, xmax, ymax)."""
        return self._area_extent

    def copy(self, **override_kwargs):
        """Make a copy of the current area.

        This replaces the current values with anything in *override_kwargs*.
        """
        kwargs = {'area_id': self.area_id,
                  'description': self.description,
                  'proj_id': self.proj_id,
                  'projection': self.crs_wkt,
                  'width': self.width,
                  'height': self.height,
                  'area_extent': self.area_extent,
                  }
        kwargs.update(override_kwargs)
        return AreaDefinition(**kwargs)

    def aggregate(self, **dims):
        """Return an aggregated version of the area."""
        width = int(self.width / dims.get('x', 1))
        height = int(self.height / dims.get('y', 1))
        return self.copy(height=height, width=width)

    @property
    def resolution(self):
        """Return area resolution in X and Y direction."""
        return self.pixel_size_x, self.pixel_size_y

    @property
    def name(self):
        """Return area name."""
        warnings.warn("'name' is deprecated, use 'description' instead.", PendingDeprecationWarning, stacklevel=2)
        return self.description

    @classmethod
    def from_epsg(cls, code, resolution):
        """Create an AreaDefinition object from an epsg code (string or int) and a resolution."""
        if CRS is None:
            raise NotImplementedError
        crs = CRS('EPSG:' + str(code))
        bounds = crs.area_of_use.bounds
        proj = Proj(crs)
        left1, low1 = proj(bounds[0], bounds[1])
        right1, up1 = proj(bounds[2], bounds[3])
        left2, up2 = proj(bounds[0], bounds[3])
        right2, low2 = proj(bounds[2], bounds[1])
        left = min(left1, left2)
        right = max(right1, right2)
        up = max(up1, up2)
        low = min(low1, low2)
        area_extent = (left, low, right, up)
        return create_area_def(crs.name, crs, area_extent=area_extent, resolution=resolution)

    @classmethod
    def from_extent(cls, area_id, projection, shape, area_extent, units=None, **kwargs):
        """Create an AreaDefinition object from area_extent and shape.

        Parameters
        ----------
        area_id : str
            ID of area
        projection : dict or str
            Projection parameters as a proj4_dict or proj4_string
        shape : list
            Number of pixels in the y and x direction (height, width)
        area_extent : list
            Area extent as a list (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        units : str, optional
            Units that provided arguments should be interpreted as. This can be
            one of 'deg', 'degrees', 'meters', 'metres', and any
            parameter supported by the
            `cs2cs -lu <https://proj4.org/apps/cs2cs.html#cmdoption-cs2cs-lu>`_
            command. Units are determined in the following priority:

            1. units expressed with each variable through a DataArray's attrs attribute.
            2. units passed to ``units``
            3. units used in ``projection``
            4. meters

        description : str, optional
            Description/name of area. Defaults to area_id
        proj_id : str, optional
            ID of projection
        nprocs : int, optional
            Number of processor cores to be used
        lons : numpy array, optional
            Grid lons
        lats : numpy array, optional
            Grid lats

        Returns
        -------
        AreaDefinition : AreaDefinition
        """
        return create_area_def(area_id, projection, shape=shape, area_extent=area_extent, units=units, **kwargs)

    @classmethod
    def from_circle(cls, area_id, projection, center, radius, shape=None, resolution=None, units=None, **kwargs):
        """Create an AreaDefinition from center, radius, and shape or from center, radius, and resolution.

        Parameters
        ----------
        area_id : str
            ID of area
        projection : dict or str
            Projection parameters as a proj4_dict or proj4_string
        center : list
            Center of projection (x, y)
        radius : list or float
            Length from the center to the edges of the projection (dx, dy)
        shape : list, optional
            Number of pixels in the y and x direction (height, width)
        resolution : list or float, optional
            Size of pixels: (dx, dy)
        units : str, optional
            Units that provided arguments should be interpreted as. This can be
            one of 'deg', 'degrees', 'meters', 'metres', and any
            parameter supported by the
            `cs2cs -lu <https://proj4.org/apps/cs2cs.html#cmdoption-cs2cs-lu>`_
            command. Units are determined in the following priority:

            1. units expressed with each variable through a DataArray's attrs attribute.
            2. units passed to ``units``
            3. units used in ``projection``
            4. meters

        description : str, optional
            Description/name of area. Defaults to area_id
        proj_id : str, optional
            ID of projection
        nprocs : int, optional
            Number of processor cores to be used
        lons : numpy array, optional
            Grid lons
        lats : numpy array, optional
            Grid lats
        optimize_projection:
            Whether the projection parameters have to be optimized for a DynamicAreaDefinition.

        Returns
        -------
        AreaDefinition or DynamicAreaDefinition : AreaDefinition or DynamicAreaDefinition
            If shape or resolution are provided, an AreaDefinition object is returned.
            Else a DynamicAreaDefinition object is returned

        Notes
        -----
        * ``resolution`` and ``radius`` can be specified with one value if dx == dy
        """
        return create_area_def(area_id, projection, shape=shape, center=center, radius=radius,
                               resolution=resolution, units=units, **kwargs)

    @classmethod
    def from_area_of_interest(cls, area_id, projection, shape, center, resolution, units=None, **kwargs):
        """Create an AreaDefinition from center, resolution, and shape.

        Parameters
        ----------
        area_id : str
            ID of area
        projection : dict or str
            Projection parameters as a proj4_dict or proj4_string
        shape : list
            Number of pixels in the y and x direction (height, width)
        center : list
            Center of projection (x, y)
        resolution : list or float
            Size of pixels: (dx, dy). Can be specified with one value if dx == dy
        units : str, optional
            Units that provided arguments should be interpreted as. This can be
            one of 'deg', 'degrees', 'meters', 'metres', and any
            parameter supported by the
            `cs2cs -lu <https://proj4.org/apps/cs2cs.html#cmdoption-cs2cs-lu>`_
            command. Units are determined in the following priority:

            1. units expressed with each variable through a DataArray's attrs attribute.
            2. units passed to ``units``
            3. units used in ``projection``
            4. meters

        description : str, optional
            Description/name of area. Defaults to area_id
        proj_id : str, optional
            ID of projection
        nprocs : int, optional
            Number of processor cores to be used
        lons : numpy array, optional
            Grid lons
        lats : numpy array, optional
            Grid lats

        Returns
        -------
        AreaDefinition : AreaDefinition
        """
        return create_area_def(area_id, projection, shape=shape, center=center,
                               resolution=resolution, units=units, **kwargs)

    @classmethod
    def from_ul_corner(cls, area_id, projection, shape, upper_left_extent, resolution, units=None, **kwargs):
        """Create an AreaDefinition object from upper_left_extent, resolution, and shape.

        Parameters
        ----------
        area_id : str
            ID of area
        projection : dict or str
            Projection parameters as a proj4_dict or proj4_string
        shape : list
            Number of pixels in the y and x direction (height, width)
        upper_left_extent : list
            Upper left corner of upper left pixel (x, y)
        resolution : list or float
            Size of pixels in **meters**: (dx, dy). Can be specified with one value if dx == dy
        units : str, optional
            Units that provided arguments should be interpreted as. This can be
            one of 'deg', 'degrees', 'meters', 'metres', and any
            parameter supported by the
            `cs2cs -lu <https://proj4.org/apps/cs2cs.html#cmdoption-cs2cs-lu>`_
            command. Units are determined in the following priority:

            1. units expressed with each variable through a DataArray's attrs attribute.
            2. units passed to ``units``
            3. units used in ``projection``
            4. meters

        description : str, optional
            Description/name of area. Defaults to area_id
        proj_id : str, optional
            ID of projection
        nprocs : int, optional
            Number of processor cores to be used
        lons : numpy array, optional
            Grid lons
        lats : numpy array, optional
            Grid lats

        Returns
        -------
        AreaDefinition : AreaDefinition
        """
        return create_area_def(area_id, projection, shape=shape, upper_left_extent=upper_left_extent,
                               resolution=resolution, units=units, **kwargs)

    @classmethod
    def from_cf(cls, cf_file, variable=None, y=None, x=None):
        """Create an AreaDefinition object from a netCDF/CF file.

        Parameters
        ----------
        nc_file : string or object
            path to a netCDF/CF file, or opened xarray.Dataset object
        variable : string, optional
            name of the variable to load the AreaDefinition from
            If variable is None the file will be searched for valid CF
            area definitions
        y : string, optional
            name of the variable to use as 'y' axis of the CF area definition
            If y is None an appropriate 'y' axis will be deduced from the CF file
        x : string, optional
            name of the variable to use as 'x' axis of the CF area definition
            If x is None an appropriate 'x' axis will be deduced from the CF file

        Returns
        -------
        AreaDefinition : AreaDefinition

        """
        return load_cf_area(cf_file, variable=variable, y=y, x=x)[0]

    def __hash__(self):
        """Compute the hash of this object."""
        if self.hash is None:
            self.hash = int(self.update_hash().hexdigest(), 16)
        return self.hash

    @property
    def proj_str(self):
        """Return PROJ projection string.

        This is no longer the preferred way of describing CRS information.
        Switch to the `crs` or `crs_wkt` properties for the most flexibility.

        """
        proj_dict = self.proj_dict.copy()
        if 'towgs84' in proj_dict and isinstance(proj_dict['towgs84'], list):
            # pyproj 2+ creates a list in the dictionary
            # but the string should be comma-separated
            if all(x == 0 for x in proj_dict['towgs84']):
                # all 0s in towgs84 are technically equal to not having them
                # specified, but PROJ considers them different
                proj_dict.pop('towgs84')
            else:
                proj_dict['towgs84'] = ','.join(str(x) for x in proj_dict['towgs84'])
        return proj4_dict_to_str(proj_dict, sort=True)

    def __str__(self):
        """Return string representation of the AreaDefinition."""
        # We need a sorted dictionary for a unique hash of str(self)
        with ignore_pyproj_proj_warnings():
            proj_dict = self.proj_dict
        proj_param_str = ', '.join(["'%s': '%s'" % (str(k), str(proj_dict[k])) for k in sorted(proj_dict.keys())])
        proj_str = '{' + proj_param_str + '}'
        if not self.proj_id:
            third_line = ""
        else:
            third_line = "Projection ID: {0}\n".format(self.proj_id)
        return ('Area ID: {0}\nDescription: {1}\n{2}'
                'Projection: {3}\nNumber of columns: {4}\nNumber of rows: {5}\n'
                'Area extent: {6}').format(self.area_id, self.description, third_line,
                                           proj_str, self.width, self.height,
                                           tuple(round(x, 4) for x in self.area_extent))

    __repr__ = __str__

    def _repr_html_(self):
        return _formatting_html.area_repr(self)

    def to_cartopy_crs(self):
        """Convert projection to cartopy CRS object."""
        import cartopy.crs as ccrs
        if not issubclass(ccrs.Projection, CRS):
            raise ImportError("Pyresample only supports converting to cartopy "
                              "0.20.0+ CRS objects. Either update cartopy or "
                              "downgrade to an older version of Pyresample "
                              "(<1.22.0) that supports older versions of "
                              "cartopy.")

        # cartopy 0.20+ are subclasses of Pyproj CRS class
        bounds = (self.area_extent[0],
                  self.area_extent[2],
                  self.area_extent[1],
                  self.area_extent[3])
        from pyresample.utils.cartopy import Projection
        crs = Projection(self.crs, bounds=bounds)
        return crs

    def _cartopy_proj_params(self):
        if self.crs.to_epsg() is not None:
            return "EPSG:{}".format(self.crs.to_epsg())
        return self.crs.to_proj4()

    def to_odc_geobox(self):
        """Convert AreaDefinition to ODC GeoBox.

        See: https://odc-geo.readthedocs.io/en/latest/
        """
        if odc_geo is None:
            raise ModuleNotFoundError("Please install 'odc-geo' to use this method.")

        return odc_geo.geobox.GeoBox.from_bbox(bbox=self.area_extent, crs=self.crs,
                                               resolution=odc_geo.Resolution(x=self.pixel_size_x, y=-self.pixel_size_y),
                                               tight=True)

    def create_areas_def(self):
        """Generate YAML formatted representation of this area.

        Deprecated.  Use :meth:`dump` instead.
        """
        warnings.warn("'create_areas_def' is deprecated. Please use `dump` instead, which also "
                      "supports writing directly to a file.", DeprecationWarning, stacklevel=2)

        return self.dump()

    def dump(self, filename=None):
        """Generate YAML formatted representation of this area.

        For the opposite (i.e. to get an AreaDefinition from a YAML-formatted
        representation), see :func:`~pyresample.area_config.load_area_from_string`.

        Args:
            filename (str or pathlib.Path or file-like object): Yaml file location to dump the area to.

        Returns:
            If file is None returns yaml str
        """
        if self.crs.to_epsg() is not None:
            proj_dict = {'EPSG': self.crs.to_epsg()}
        else:
            with ignore_pyproj_proj_warnings():
                proj_dict = self.crs.to_dict()

        res = OrderedDict(description=self.description,
                          projection=OrderedDict(proj_dict),
                          shape=OrderedDict([('height', self.height), ('width', self.width)]))
        units = res['projection'].pop('units', None)
        extent = OrderedDict([('lower_left_xy', _numpy_values_to_native(self.area_extent[:2])),
                              ('upper_right_xy', _numpy_values_to_native(self.area_extent[2:]))])
        if units is not None:
            extent['units'] = units
        res['area_extent'] = extent

        yml_str = ordered_dump(OrderedDict([(self.area_id, res)]), default_flow_style=None)

        if filename is not None:
            if hasattr(filename, 'write'):
                filename.write(yml_str)
            elif isinstance(filename, (str, Path)):
                with open(filename, 'a') as fh:
                    fh.write(yml_str)
        else:
            return yml_str

    def create_areas_def_legacy(self):
        """Create area definition in legacy format."""
        warnings.warn("Pyresample's legacy areas file format is deprecated. "
                      "Use the 'YAML' format instead.", stacklevel=2)
        proj_dict = self.proj_dict
        proj_str = ','.join(["%s=%s" % (str(k), str(proj_dict[k]))
                             for k in sorted(proj_dict.keys())])

        fmt = "REGION: {name} {{\n"
        fmt += "\tNAME:\t{name}\n"
        fmt += "\tPCS_ID:\t{area_id}\n"
        fmt += "\tPCS_DEF:\t{proj_str}\n"
        fmt += "\tXSIZE:\t{x_size}\n"
        fmt += "\tYSIZE:\t{y_size}\n"
        fmt += "\tAREA_EXTENT: {area_extent}\n}};\n"
        area_def_str = fmt.format(name=self.description, area_id=self.area_id,
                                  proj_str=proj_str, x_size=self.width,
                                  y_size=self.height,
                                  area_extent=self.area_extent)
        return area_def_str

    def __eq__(self, other):
        """Test for equality."""
        try:
            return ((np.allclose(self.area_extent, other.area_extent)) and
                    (self.crs == other.crs) and
                    (self.shape == other.shape))
        except AttributeError:
            return super().__eq__(other)

    def __ne__(self, other):
        """Test for equality."""
        return not self.__eq__(other)

    def update_hash(self, existing_hash: Optional[_Hash] = None) -> _Hash:
        """Update a hash, or return a new one if needed."""
        if existing_hash is None:
            existing_hash = hashlib.sha1()  # nosec: B324
        existing_hash.update(self.crs_wkt.encode('utf-8'))
        existing_hash.update(np.array(self.shape))
        existing_hash.update(np.array(self.area_extent))
        return existing_hash

    @daskify_2in_2out
    def get_array_coordinates_from_lonlat(self, lon, lat):
        """Retrieve the array coordinates (float) for a given lon/lat.

        If lon,lat is a tuple of sequences of longitudes and latitudes, a tuple
        of arrays is returned.

        Args:
            lon (array_like): point or sequence of longitudes
            lat (array_like): point or sequence of latitudes

        Returns:
            floats or arrays of floats: the array coordinates (cols/rows)
        """
        xm_, ym_ = self.get_projection_coordinates_from_lonlat(lon, lat)
        return self.get_array_coordinates_from_projection_coordinates(xm_, ym_)

    @preserve_scalars
    @daskify_2in_2out
    def get_array_coordinates_from_projection_coordinates(self, xm, ym):
        """Find the floating-point grid cell index for a specified projection coordinate.

        If xm, ym is a tuple of sequences of projection coordinates, a tuple
        of arrays are returned.

        Args:
            xm (array_like): point or sequence of x-coordinates in
                                 meters (map projection)
            ym (array_like): point or sequence of y-coordinates in
                                 meters (map projection)

        Returns:
            floats or arrays of floats : the array coordinates (cols/rows)
        """
        xm = np.asanyarray(xm)
        ym = np.asanyarray(ym)

        upl_x, upl_y, xscale, yscale = self._get_corner_and_scale()

        x__ = (xm - upl_x) / xscale
        y__ = (ym - upl_y) / yscale

        return x__, y__

    def _get_corner_and_scale(self):
        """Unpack pixel sizes and upper left pixel."""
        xscale = self.pixel_size_x
        # because rows direction is the opposite of y's
        yscale = -self.pixel_size_y
        upl_x, upl_y = self.pixel_upper_left
        return upl_x, upl_y, xscale, yscale

    @masked_ints
    def get_array_indices_from_lonlat(self, lon, lat):
        """Find the closest integer grid cell index for a given lon/lat.

        If lon,lat is a point, a ValueError is raised if it is outside the area
        domain. If lon,lat is a tuple of sequences of longitudes and latitudes,
        a tuple of masked arrays are returned. The masked values are the actual
        row and col indexing the grid cell if the area had been big enough, or
        the numpy default (999999) if invalid.

        Args:
            lon (array_like) : point or sequence of longitudes
            lat (array_like): point or sequence of latitudes

        Returns:
            ints or masked arrays of ints : the array indices (cols/rows)

        Raises:
            ValueError: if the return point is outside the area domain
        """
        return self.get_array_coordinates_from_lonlat(lon, lat)

    @masked_ints
    def get_array_indices_from_projection_coordinates(self, xm, ym):
        """Find the closest integer grid cell index for a specified projection coordinate.

        If xm, ym is a point, a ValueError is raised if it is outside the area
        domain. If xm, ym is a tuple of sequences of projection coordinates, a
        tuple of masked arrays are returned.

        Args:
            xm (array_like): point or sequence of x-coordinates in
                                 meters (map projection)
            ym (array_like): point or sequence of y-coordinates in
                                 meters (map projection)

        Returns:
            ints or masked arrays of ints : the array indices (cols/rows)

        Raises:
            ValueError: if the return point is outside the area domain
        """
        return self.get_array_coordinates_from_projection_coordinates(xm, ym)

    @daskify_2in_2out
    def get_projection_coordinates_from_lonlat(self, lon, lat):
        """Get the projection coordinate from longitudes and latitudes.

        If lon,lat is a tuple of sequences of longitudes and latitudes, a tuple
        of arrays is returned.

        Args:
            lon (array_like): point or sequence of longitudes
            lat (array_like): point or sequence of latitudes

        Returns:
            floats or arrays of floats: the projection coordinates x, y in meters
        """
        p = Proj(self.crs)
        return p(lon, lat)

    @daskify_2in_2out
    def get_projection_coordinates_from_array_coordinates(self, cols, rows):
        """Get the projection coordinate from the array coordinates.

        If cols, rows is a tuple of sequences of array coordinates, a tuple
        of arrays is returned.

        Args:
            cols (array_like): the column coordinates
            rows (array_like): the row coordinates

        Returns:
            floats or arrays of floats: the projection coordinates x, y in meters
        """
        cols = np.asanyarray(cols)
        rows = np.asanyarray(rows)

        upl_x, upl_y, xscale, yscale = self._get_corner_and_scale()

        x__ = cols * xscale + upl_x
        y__ = rows * yscale + upl_y

        return x__, y__

    @daskify_2in_2out
    def get_lonlat_from_array_coordinates(self, cols, rows):
        """Get the longitude and latitude from (floating) column and row indices.

        If cols, rows is a tuple of sequences of array coordinates, a tuple
        of arrays is returned.

        Args:
            cols (array_like): the column coordinates
            rows (array_like): the row coordinates

        Returns:
            floats or arrays of floats: the longitude, latitude in degrees

        """
        x__, y__ = self.get_projection_coordinates_from_array_coordinates(cols, rows)
        return self.get_lonlat_from_projection_coordinates(x__, y__)

    @daskify_2in_2out
    def get_lonlat_from_projection_coordinates(self, xm, ym):
        """Get the lonlat from projection coordinates.

        If xm, ym is a tuple of sequences of projection coordinates, a tuple
        of arrays is returned.

        Args:
            xm (array_like): the x projection coordinates in meters
            ym (array_like): the y projection coordinates in meters

        Returns:
            floats or arrays of floats: the longitude, latitude in degrees

        """
        p = Proj(self.crs)
        return p(xm, ym, inverse=True)

    def colrow2lonlat(self, cols, rows):
        """Return lons and lats for the given image columns and rows.

        Both scalars and arrays are supported. To be used with scarse
        data points instead of slices (see get_lonlats).
        """
        p = Proj(self.crs)
        x = self.projection_x_coords
        y = self.projection_y_coords
        return p(x[cols], y[rows], inverse=True)

    def lonlat2colrow(self, lons, lats):
        """Return image columns and rows for the given lons and lats.

        Both scalars and arrays are supported.  Same as
        get_xy_from_lonlat, renamed for convenience.
        """
        warnings.warn("'lonlat2colrow' is deprecated, please use "
                      "'get_array_indices_from_lonlat' instead.", DeprecationWarning, stacklevel=2)

        return self.get_array_indices_from_lonlat(lons, lats)

    def get_xy_from_lonlat(self, lon, lat):
        """Retrieve closest x and y coordinates.

        Retrieve the closest x and y coordinates (column, row indices) for the
        specified geolocation (lon,lat) if inside area. If lon,lat is a point a
        ValueError is raised if the return point is outside the area domain. If
        lon,lat is a tuple of sequences of longitudes and latitudes, a tuple of
        masked arrays are returned.

        Args:
            lon : point or sequence (list or array) of longitudes
            lat : point or sequence (list or array) of latitudes

        Returns:
            (x, y) : tuple of points/arrays
        """
        warnings.warn("'get_xy_from_lonlat' is deprecated, please use "
                      "'get_array_indices_from_lonlat' instead.", DeprecationWarning,
                      stacklevel=2)

        return self.get_array_indices_from_lonlat(lon, lat)

    def get_xy_from_proj_coords(self, xm, ym):
        """Find closest grid cell index for a specified projection coordinate.

        If xm, ym is a tuple of sequences of projection coordinates, a tuple
        of masked arrays are returned.

        Args:
            xm (list or array): point or sequence of x-coordinates in
                                 meters (map projection)
            ym (list or array): point or sequence of y-coordinates in
                                 meters (map projection)

        Returns:
            x, y : column and row grid cell indexes as 2 scalars or arrays

        Raises:
            ValueError: if the return point is outside the area domain
        """
        warnings.warn("'get_xy_from_proj_coords' is deprecated, please use "
                      "'get_array_indices_from_projection_coordinates' instead.", DeprecationWarning,
                      stacklevel=2)

        return self.get_array_indices_from_projection_coordinates(self, xm, ym)

    def get_lonlat(self, row, col):
        """Retrieve lon and lat values of single point in area grid.

        Parameters
        ----------
        row : int
        col : int

        Returns
        -------
        (lon, lat) : tuple of floats
        """
        lon, lat = self.get_lonlats(nprocs=None, data_slice=(row, col))
        return lon.item(), lat.item()

    def get_proj_vectors_dask(self, chunks=None, dtype=None):
        """Get projection vectors."""
        warnings.warn("'get_proj_vectors_dask' is deprecated, please use "
                      "'get_proj_vectors' with the 'chunks' keyword argument specified.", DeprecationWarning,
                      stacklevel=2)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_proj_vectors(dtype=dtype, chunks=chunks)

    def _get_proj_vectors(self, dtype=None, chunks=None):
        """Get 1D projection coordinates."""
        if dtype is None:
            dtype = self.dtype
        x, y = _generate_1d_proj_vectors((0, self.width),
                                         (0, self.height),
                                         (self.pixel_size_x, self.pixel_size_y),
                                         (self.pixel_upper_left[0], self.pixel_upper_left[1]),
                                         dtype, chunks=chunks)
        return x, y

    def get_proj_vectors(self, dtype=None, chunks=None):
        """Calculate 1D projection coordinates for the X and Y dimension.

        Parameters
        ----------
        dtype : numpy.dtype
            Numpy data type for the returned arrays
        chunks : int or tuple
            Return dask arrays with the chunk size specified. If this is a
            tuple then the first element is the Y array's chunk size and the
            second is the X array's chunk size.

        Returns
        -------
        tuple: (X, Y) where X and Y are 1-dimensional numpy arrays

        The data type of the returned arrays can be controlled with the
        `dtype` keyword argument. If `chunks` is provided then dask arrays
        are returned instead.
        """
        return self._get_proj_vectors(dtype=dtype, chunks=chunks)

    def get_proj_coords_dask(self, chunks=None, dtype=None):
        """Get projection coordinates."""
        warnings.warn("'get_proj_coords_dask' is deprecated, please use "
                      "'get_proj_coords' with the 'chunks' keyword argument specified.", DeprecationWarning,
                      stacklevel=2)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_proj_coords(chunks=chunks, dtype=dtype)

    def get_proj_coords(self, data_slice=None, dtype=None, chunks=None):
        """Get projection coordinates of grid.

        Parameters
        ----------
        data_slice : slice object, optional
            Calculate only coordinates for specified slice
        dtype : numpy.dtype, optional
            Data type of the returned arrays
        chunks: int or tuple, optional
            Create dask arrays and use this chunk size

        Returns
        -------
        (target_x, target_y) : tuple of numpy arrays
            Grids of area x- and y-coordinates in projection units

        .. versionchanged:: 1.11.0

            Removed 'cache' keyword argument and add 'chunks' for creating
            dask arrays.
        """
        if dtype is None:
            dtype = self.dtype
        y_slice, x_slice = self._get_yx_data_slice(data_slice)
        if chunks is not None:
            target_x, target_y = self._proj_coords_dask(chunks, dtype)
            if y_slice is not None:
                target_x = target_x[y_slice, x_slice]
                target_y = target_y[y_slice, x_slice]
            return target_x, target_y

        target_x, target_y = self._get_proj_vectors(dtype=dtype, chunks=chunks)
        if y_slice is not None:
            target_y = target_y[y_slice]
        if x_slice is not None:
            target_x = target_x[x_slice]

        target_x, target_y = np.meshgrid(target_x, target_y)
        return target_x, target_y

    @staticmethod
    def _get_yx_data_slice(data_slice):
        if data_slice is not None and isinstance(data_slice, slice):
            return data_slice, slice(None, None, None)
        elif data_slice is not None:
            return data_slice[0], data_slice[1]
        return None, None

    def _proj_coords_dask(self, chunks, dtype):
        """Generate 2D x and y coordinate arrays.

        This is a separate function because it allows dask to optimize and
        separate the individual 2D chunks of coordinates. Using the basic
        numpy form of these calculations produces an unnecessary
        relationship between the "arange" 1D projection vectors and every
        2D coordinate chunk. This makes it difficult for dask to schedule
        2D chunks in an optimal way.

        """
        y_chunks, x_chunks = _chunks_to_yx_chunks(chunks)
        norm_y_chunks, norm_x_chunks = da.core.normalize_chunks((y_chunks, x_chunks), self.shape, dtype=dtype)
        # We must provide `chunks` and `dtype` as passed arguments to ensure
        # the returned array has a unique dask name
        # See: https://github.com/dask/dask/issues/8450
        res = da.map_blocks(_generate_2d_coords,
                            self.pixel_size_x, self.pixel_size_y,
                            self.pixel_upper_left[0], self.pixel_upper_left[1],
                            chunks, dtype,
                            chunks=((2,), norm_y_chunks, norm_x_chunks),
                            meta=np.array((), dtype=dtype),
                            dtype=dtype,
                            )
        target_x, target_y = res[0], res[1]
        return target_x, target_y

    @property
    def projection_x_coords(self):
        """Return projection X coordinates."""
        return self.get_proj_vectors()[0]

    @property
    def projection_y_coords(self):
        """Return projection Y coordinates."""
        return self.get_proj_vectors()[1]

    @property
    def outer_boundary_corners(self):
        """Return the lon,lat of the outer edges of the corner points."""
        from pyresample.spherical_geometry import Coordinate
        proj = Proj(self.crs)
        corner_lons, corner_lats = proj((self.area_extent[0], self.area_extent[2],
                                         self.area_extent[2], self.area_extent[0]),
                                        (self.area_extent[3], self.area_extent[3],
                                         self.area_extent[1], self.area_extent[1]),
                                        inverse=True)
        return [Coordinate(corner_lons[0], corner_lats[0]),
                Coordinate(corner_lons[1], corner_lats[1]),
                Coordinate(corner_lons[2], corner_lats[2]),
                Coordinate(corner_lons[3], corner_lats[3])]

    def get_lonlats_dask(self, chunks=None, dtype=None):
        """Get longitudes and latitudes."""
        warnings.warn("'get_lonlats_dask' is deprecated, please use "
                      "'get_lonlats' with the 'chunks' keyword argument specified.", DeprecationWarning, stacklevel=2)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_lonlats(chunks=chunks, dtype=dtype)

    def get_lonlats(self, nprocs=None, data_slice=None, cache=False, dtype=None, chunks=None):
        """Return lon and lat arrays of area.

        Note that this historically this method always returns
        longitude/latitudes on the geodetic (unprojected) model of the Earth
        used by the Coordinate Reference System (CRS) for this area. However,
        this is not true for shifted datums. For example, a projection
        including a PROJ.4 parameter like ``+pm=180`` to shift
        longitudes/latitudes 180 degrees, will return degrees on the ``+pm=0``
        equivalent of the geodetic CRS.

        Parameters
        ----------
        nprocs : int, optional
            Number of processor cores to be used.
            Defaults to the nprocs set when instantiating object
        data_slice : slice object, optional
            Calculate only coordinates for specified slice
        cache : bool, optional
            Store the result internally for later reuse. Requires data_slice
            to be None.
        dtype : numpy.dtype, optional
            Data type of the returned arrays
        chunks: int or tuple, optional
            Create dask arrays and use this chunk size

        Returns
        -------
        (lons, lats) : tuple of numpy arrays
            Grids of area lons and and lats
        """
        if cache:
            warnings.warn("'cache' keyword argument will be removed in the "
                          "future and data will not be cached.", PendingDeprecationWarning, stacklevel=2)
        if dtype is None:
            dtype = self.dtype

        if self.lons is not None:
            # Data is cache already
            lons = self.lons
            lats = self.lats
            if data_slice is not None:
                lons = lons[data_slice]
                lats = lats[data_slice]
            return lons, lats

        # Get X/Y coordinates for the whole area
        target_x, target_y = self.get_proj_coords(data_slice=data_slice, chunks=chunks, dtype=dtype)
        if nprocs is None and not hasattr(target_x, 'chunks'):
            nprocs = self.nprocs
        if nprocs is not None and hasattr(target_x, 'chunks'):
            # we let 'get_proj_coords' decide if dask arrays should be made
            # but if the user provided nprocs then this doesn't make sense
            raise ValueError("Can't specify 'nprocs' and 'chunks' at the same time")

        if hasattr(target_x, 'chunks'):
            # we are using dask arrays, map blocks to th
            from dask.array import map_blocks
            res = map_blocks(_invproj, target_x, target_y,
                             chunks=(2,) + target_x.chunks,
                             meta=np.array((), dtype=target_x.dtype),
                             dtype=target_x.dtype,
                             new_axis=[0], proj_wkt=self.crs_wkt)
            lons, lats = res[0], res[1]
            return lons, lats

        proj_kwargs = {}
        if nprocs > 1:
            target_proj = Proj_MP(self.crs)
            proj_kwargs["nprocs"] = nprocs
            proj_kwargs["inverse"] = True
        else:
            gcrs = get_geodetic_crs_with_no_datum_shift(self.crs)
            target_trans = pyproj.Transformer.from_crs(gcrs, self.crs, always_xy=True)
            target_proj = target_trans.transform
            proj_kwargs["direction"] = TransformDirection.INVERSE

        # Get corresponding longitude and latitude values
        lons, lats = target_proj(target_x, target_y, **proj_kwargs)
        lons = np.asanyarray(lons, dtype=dtype)
        lats = np.asanyarray(lats, dtype=dtype)

        if cache and data_slice is None:
            # Cache the result if requested
            self.lons = lons
            self.lats = lats

        return lons, lats

    @property
    def proj4_string(self):
        """Return projection definition as Proj.4 string."""
        warnings.warn("'proj4_string' is deprecated, please use 'proj_str' "
                      "instead.", DeprecationWarning, stacklevel=2)
        return proj4_dict_to_str(self.proj_dict)

    def get_area_slices(self, area_to_cover, shape_divisible_by=None):
        """Compute the slice to read based on an `area_to_cover`."""
        from .future.geometry._subset import get_area_slices
        return get_area_slices(self, area_to_cover, shape_divisible_by)

    def crop_around(self, other_area):
        """Crop this area around `other_area`."""
        xslice, yslice = self.get_area_slices(other_area)
        return self[yslice, xslice]

    def __getitem__(self, key):
        """Apply slices to the area_extent and size of the area."""
        yslice, xslice = key
        # Get actual values, replace Nones
        yindices = yslice.indices(self.height)
        total_rows = int((yindices[1] - yindices[0]) / yindices[2])
        ystopactual = yindices[1] - (yindices[1] - 1) % yindices[2]
        xindices = xslice.indices(self.width)
        total_cols = int((xindices[1] - xindices[0]) / xindices[2])
        xstopactual = xindices[1] - (xindices[1] - 1) % xindices[2]
        yslice = slice(yindices[0], ystopactual, yindices[2])
        xslice = slice(xindices[0], xstopactual, xindices[2])

        new_area_extent = ((self.pixel_upper_left[0] + (xslice.start - 0.5) * self.pixel_size_x),
                           (self.pixel_upper_left[1] - (yslice.stop - 0.5) * self.pixel_size_y),
                           (self.pixel_upper_left[0] + (xslice.stop - 0.5) * self.pixel_size_x),
                           (self.pixel_upper_left[1] - (yslice.start - 0.5) * self.pixel_size_y))

        new_area = AreaDefinition(self.area_id, self.description,
                                  self.proj_id, self.crs,
                                  total_cols,
                                  total_rows,
                                  new_area_extent)
        new_area.crop_offset = (self.crop_offset[0] + yslice.start,
                                self.crop_offset[1] + xslice.start)
        return new_area

    def geocentric_resolution(self, ellps='WGS84', radius=None):
        """Find best estimate for overall geocentric resolution.

        This method is extremely important to the results of KDTree-based
        resamplers like the nearest neighbor resampling. This is used to
        determine how far the KDTree should be queried for valid pixels
        before giving up (`radius_of_influence`). This method attempts to
        make a best guess at what geocentric resolution (the units used by
        the KDTree) represents the majority of an area.

        To do this this method will:

        1. Create a vertical mid-line and a horizontal mid-line.
        2. Convert these coordinates to geocentric coordinates.
        3. Compute the distance between points along these lines.
        4. Take the histogram of each set of distances and find the
           bin with the most points.
        5. Take the average of the edges of that bin.
        6. Return the maximum of the vertical and horizontal bin
           edge averages.

        """
        def _safe_bin_edges(arr):
            try:
                return np.histogram_bin_edges(arr, bins=10)[:2]
            except ValueError:
                # numpy 2.1.0+ produces a ValueError if it can't fill
                # all bins due to a small data range
                # we just arbitrarily use the first 2 elements as all elements
                # should be within floating point precision for our use case
                return arr[:2]
        from pyproj.transformer import Transformer
        rows, cols = self.shape
        mid_row = rows // 2
        mid_col = cols // 2
        x, y = self.get_proj_vectors()
        mid_col_x = np.repeat(x[mid_col], y.size)
        mid_row_y = np.repeat(y[mid_row], x.size)
        src = Proj(self.crs)
        if radius:
            dst = Proj("+proj=cart +a={} +b={}".format(radius, radius))
        else:
            dst = Proj("+proj=cart +ellps={}".format(ellps))
        # need some altitude, go with the surface (0)
        alt_x = np.zeros(x.size)
        alt_y = np.zeros(y.size)
        transformer = Transformer.from_crs(src.crs, dst.crs, always_xy=True)
        # convert our midlines to (X, Y, Z) geocentric coordinates
        hor_xyz = np.stack(transformer.transform(x, mid_row_y, alt_x), axis=1)
        vert_xyz = np.stack(transformer.transform(mid_col_x, y, alt_y), axis=1)
        # Find the distance in meters along our midlines
        hor_dist = np.linalg.norm(np.diff(hor_xyz, axis=0), axis=1)
        vert_dist = np.linalg.norm(np.diff(vert_xyz, axis=0), axis=1)
        # Get rid of any NaNs or infinite values
        hor_dist = hor_dist[np.isfinite(hor_dist)]
        vert_dist = vert_dist[np.isfinite(vert_dist)]
        # use the average of the largest histogram bin to avoid
        # outliers and really large values.
        # Very useful near edge of disk geostationary areas.
        hor_res = vert_res = 0
        if hor_dist.size:
            hor_res = np.mean(_safe_bin_edges(hor_dist))
        if vert_dist.size:
            vert_res = np.mean(_safe_bin_edges(vert_dist))
        # Use the maximum distance between the two midlines instead of
        # binning both of them together. If we binned them together then
        # we are highly dependent on the shape of the area (more rows in
        # the area would almost always mean that we resulted in the vertical
        # midline's distance).
        res = max(hor_res, vert_res)
        if not res:
            raise RuntimeError("Could not calculate geocentric resolution")
        # return np.max(np.concatenate(vert_dist, hor_dist))  # alternative to histogram
        return res

    def _get_projection_sides(self, vertices_per_side: Optional[int] = None) -> tuple:
        """Return the projection boundary sides of the current area.

        Args:
            vertices_per_side:
                The number of points to provide for each side.
                By default (None) the full width and height will be provided.
                If the area object is an AreaDefinition with any corner out of the Earth disk
                (i.e. full disc geostationary area, Robinson projection, polar projections, ...)
                by default only 50 points are selected.

        Returns:
            The output structure is a tuple of two lists of four elements each.
            The first list contains the projection x coordinates.
            The second list contains the projection y coordinates.
            Each list element is a numpy array representing a specific side of the geometry.
            The order of the sides are [top", "right", "bottom", "left"]
        """
        # FIXME: Add logic for out-of-earth-disk
        if self.is_geostationary:
            return self._get_geostationary_boundary_sides(vertices_per_side=vertices_per_side,
                                                          coordinates="projection")
        sides_x, sides_y = self._get_sides(coord_fun=self.get_proj_coords,
                                           vertices_per_side=vertices_per_side)
        return sides_x, sides_y


def get_geostationary_angle_extent(geos_area):
    """Get the max earth (vs space) viewing angles in x and y."""
    # get some projection parameters
    a, b = proj4_radius_parameters(geos_area.crs)
    h = get_geostationary_height(geos_area.crs)
    req = a / 1000.0
    rp = b / 1000.0
    h = h / 1000.0 + req

    # compute some constants
    aeq = 1 - req ** 2 / (h ** 2)
    ap_ = 1 - rp ** 2 / (h ** 2)

    x_angle = np.arccos(np.sqrt(aeq))
    y_angle = np.arccos(np.sqrt(ap_))
    return x_angle, y_angle


def get_geostationary_bounding_box_in_proj_coords(geos_area, nb_points=50):
    """Get the bbox in geos projection coordinates of the valid pixels inside `geos_area`.

    Args:
      geos_area: Geostationary area definition to get the bounding box for.
      nb_points: Number of points on the polygon.

    """
    x, y = get_full_geostationary_bounding_box_in_proj_coords(geos_area, nb_points)
    ll_x, ll_y, ur_x, ur_y = geos_area.area_extent

    from shapely.geometry import Polygon
    geo_bbox = Polygon(np.vstack((x, y)).T)
    area_bbox = Polygon(((ll_x, ll_y), (ll_x, ur_y), (ur_x, ur_y), (ur_x, ll_y)))
    intersection = area_bbox.intersection(geo_bbox)
    try:
        x, y = intersection.boundary.xy
    except NotImplementedError:
        return np.array([]), np.array([])
    return np.asanyarray(x[:-1]), np.asanyarray(y[:-1])


def get_full_geostationary_bounding_box_in_proj_coords(geos_area, nb_points=50):
    """Get the valid boundary geos projection coordinates of the full disk.

    Args:
      geos_area: Geostationary area definition to get the bounding box for.
      nb_points: Number of points on the polygon
    """
    x_max_angle, y_max_angle = get_geostationary_angle_extent(geos_area)
    h = get_geostationary_height(geos_area.crs)

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    points_around = np.linspace(-np.pi, np.pi, nb_points, endpoint=False)
    x = np.cos(points_around) * (x_max_angle - 0.0001)
    y = -np.sin(points_around) * (y_max_angle - 0.0001)
    x *= h
    y *= h
    return x, y


def get_geostationary_bounding_box_in_lonlats(geos_area, nb_points=50):
    """Get the bbox in lon/lats of the valid pixels inside `geos_area`.

    Args:
      geos_area: Geostationary area definition to get the bounding box for.
      nb_points: Number of points on the polygon
    """
    x, y = get_geostationary_bounding_box_in_proj_coords(geos_area, nb_points)
    lons, lats = Proj(geos_area.crs)(x, y, inverse=True)
    return lons, lats


def get_geostationary_bounding_box(geos_area, nb_points=50):
    """Get the bbox in lon/lats of the valid pixels inside `geos_area`.

    Args:
      geos_area: Geostationary area definition to get the bounding box for.
      nb_points: Number of points on the polygon

    """
    warnings.warn("'get_geostationary_bounding_box' is deprecated. Please use "
                  "'get_geostationary_bounding_box_in_lonlats' instead.",
                  DeprecationWarning, stacklevel=2)
    return get_geostationary_bounding_box_in_lonlats(geos_area, nb_points)


def combine_area_extents_vertical(area1, area2):
    """Combine the area extents of areas 1 and 2."""
    if (area1.area_extent[0] == area2.area_extent[0] and area1.area_extent[2] == area2.area_extent[2]):
        current_extent = list(area1.area_extent)
        if np.isclose(area1.area_extent[1], area2.area_extent[3]):
            current_extent[1] = area2.area_extent[1]
        elif np.isclose(area1.area_extent[3], area2.area_extent[1]):
            current_extent[3] = area2.area_extent[3]
        else:
            raise IncompatibleAreas(
                "Can't concatenate non-contiguous area definitions: "
                "{0} and {1}".format(area1, area2))
    else:
        raise IncompatibleAreas(
            "Can't concatenate area definitions with "
            "incompatible area extents: "
            "{0} and {1}".format(area1, area2))
    return current_extent


def concatenate_area_defs(area1, area2, axis=0):
    """Append *area2* to *area1* and return the results."""
    crs_is_equal = area1.crs == area2.crs
    if axis == 0:
        same_size = area1.width == area2.width
    else:
        raise NotImplementedError('Only vertical contatenation is supported.')
    if not crs_is_equal or not same_size:
        raise IncompatibleAreas("Can't concatenate area definitions with "
                                "different projections: "
                                "{0} and {1}".format(area1, area2))

    if axis == 0:
        area_extent = combine_area_extents_vertical(area1, area2)
        x_size = int(area1.width)
        y_size = int(area1.height + area2.height)
    else:
        raise NotImplementedError('Only vertical contatenation is supported.')
    return AreaDefinition(area1.area_id, area1.description, area1.proj_id,
                          area1.crs, x_size, y_size,
                          area_extent)


class StackedAreaDefinition(_ProjectionDefinition):
    """Definition based on muliple vertically stacked AreaDefinitions."""

    def __init__(self, *definitions, **kwargs):
        """Initialize StackedAreaDefinition based on *definitions*.

        *kwargs* used here are `nprocs` and `dtype` (see AreaDefinition).
        """
        nprocs = kwargs.get('nprocs', 1)
        super().__init__(nprocs=nprocs)
        self.dtype = kwargs.get('dtype', np.float64)
        self.defs = []
        self.crs_wkt = None
        for definition in definitions:
            self.append(definition)

    @property
    def width(self):
        """Return width of the area definition."""
        return self.defs[0].width

    @property
    def height(self):
        """Return height of the area definition."""
        return sum(definition.height for definition in self.defs)

    def append(self, definition):
        """Append another definition to the area."""
        if isinstance(definition, StackedAreaDefinition):
            for area in definition.defs:
                self.append(area)
            return
        if definition.height == 0:
            return
        if not self.defs:
            self.crs_wkt = definition.crs_wkt
        elif self.crs != definition.crs:
            raise NotImplementedError('Cannot append areas:'
                                      ' CRS mismatch')
        try:
            self.defs[-1] = concatenate_area_defs(self.defs[-1], definition)
        except (IncompatibleAreas, IndexError):
            self.defs.append(definition)

    def get_lonlats(self, nprocs=None, data_slice=None, cache=False, dtype=None, chunks=None):
        """Return lon and lat arrays of the area."""
        if chunks is not None:
            from dask.array import vstack
        else:
            vstack = np.vstack

        llons = []
        llats = []
        try:
            row_slice, col_slice = data_slice
        except TypeError:
            row_slice = slice(0, self.height)
            col_slice = slice(0, self.width)
        offset = 0
        for def_idx, areadef in enumerate(self.defs):
            # compute appropriate chunks for the current AreaDefinition
            chunks_for_areadef = self._get_chunks_for_areadef_in_stacked_areadef(chunks, def_idx)

            local_row_slice = slice(max(row_slice.start - offset, 0),
                                    min(max(row_slice.stop - offset, 0), areadef.height),
                                    row_slice.step)
            lons, lats = areadef.get_lonlats(nprocs=nprocs, data_slice=(local_row_slice, col_slice),
                                             cache=cache, dtype=dtype, chunks=chunks_for_areadef)

            llons.append(lons)
            llats.append(lats)
            offset += lons.shape[0]

        self.lons = vstack(llons)
        self.lats = vstack(llats)

        return self.lons, self.lats

    def _get_chunks_for_areadef_in_stacked_areadef(self, chunks_stacked_areadef, areadef_idx):
        """Get the chunks for an AreaDefinition stored based on another chunked StackedAreaDefinition."""
        if isinstance(chunks_stacked_areadef, tuple) and isinstance(chunks_stacked_areadef[0], int):
            # defined chunk is just an integer, so use that for all areadefs
            chunks_for_areadef = chunks_stacked_areadef
        elif isinstance(chunks_stacked_areadef, tuple) and len(chunks_stacked_areadef[0]) == len(self.defs):
            # amount of chunks defined matches the amout of areadefs,
            # so assign each chunk to each areadef following the array order
            chunks_for_areadef = (chunks_stacked_areadef[0][areadef_idx], chunks_stacked_areadef[1])
        elif isinstance(chunks_stacked_areadef, tuple) and len(chunks_stacked_areadef[0]) != len(self.defs):
            # too many/few chunks defined, assignment is ambiguous, so use the actual shape of the areadef instead
            chunks_for_areadef = (self.defs[areadef_idx].shape[0], chunks_stacked_areadef[1])
        else:
            chunks_for_areadef = chunks_stacked_areadef

        return chunks_for_areadef

    def get_lonlats_dask(self, chunks=None, dtype=None):
        """Return lon and lat dask arrays of the area."""
        warnings.warn("'get_lonlats_dask' is deprecated, please use "
                      "'get_lonlats' with the 'chunks' keyword argument specified.",
                      DeprecationWarning, stacklevel=2)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_lonlats(chunks=chunks, dtype=dtype)

    def squeeze(self):
        """Generate a single AreaDefinition if possible."""
        if len(self.defs) == 1:
            return self.defs[0]
        else:
            return self

    @property
    def proj4_string(self):
        """Return projection definition as Proj.4 string."""
        warnings.warn("'proj4_string' is deprecated, please use 'proj_str' "
                      "instead.", DeprecationWarning, stacklevel=2)
        return self.defs[0].proj_str

    @property
    def proj_str(self):
        """Return projection definition as Proj.4 string."""
        return self.defs[0].proj_str

    def update_hash(self, the_hash=None):
        """Update the hash."""
        for areadef in self.defs:
            the_hash = areadef.update_hash(the_hash)
        return the_hash


def _get_slice(segments, shape):
    """Segment a 1D or 2D array."""
    if not (1 <= len(shape) <= 2):
        raise ValueError('Cannot segment array of shape: %s' % str(shape))
    else:
        size = shape[0]
        slice_length = int(np.ceil(float(size) / segments))
        start_idx = 0
        end_idx = slice_length
        while start_idx < size:
            if len(shape) == 1:
                yield slice(start_idx, end_idx)
            else:
                yield slice(start_idx, end_idx), slice(None)
            start_idx = end_idx
            end_idx = min(start_idx + slice_length, size)


def _flatten_cartesian_coords(cartesian_coords):
    """Flatten array to (n, 3) shape."""
    shape = cartesian_coords.shape
    if len(shape) > 2:
        cartesian_coords = cartesian_coords.reshape(shape[0] * shape[1], 3)
    return cartesian_coords


def _get_highest_level_class(obj1, obj2):
    if not issubclass(obj1.__class__, obj2.__class__) or not issubclass(obj2.__class__, obj1.__class__):
        raise TypeError('No common superclass for %s and %s' %
                        (obj1.__class__, obj2.__class__))

    if obj1.__class__ == obj2.__class__:
        klass = obj1.__class__
    elif issubclass(obj1.__class__, obj2.__class__):
        klass = obj2.__class__
    else:
        klass = obj1.__class__
    return klass


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    """Dump the data to YAML in ordered fashion."""
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items(), flow_style=False)

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def enclose_areas(*areas, area_id="joint-area"):
    r"""Return the smallest areadefinition enclosing one or more others.

    From one or more AreaDefinition objects (most usefully at least
    two), which shall differ only in extent, calculate the smallest
    AreaDefinition that encloses all.  Touches only the ``area_extent``;
    projection and units must be identical in all input areas and will be
    unchanged in the resulting area.  When the input areas :math:`i=1..n`
    have extent :math:`(a_i, b_i, c_i, d_i)`, the resulting area will have
    extent :math:`(\\min_i{a_i}, \\min_i{b_i}, \\max_i{c_i}, \\max_i{d_i})`.

    Args:
        *areas (AreaDefinition): AreaDefinition objects to enclose.
        area_id (Optional[str]): Name of joint area, defaults to "joint-area".
    """
    first = None
    if not areas:
        raise TypeError("Must pass at least one area, found zero.")
    for area in areas:
        if first is None:
            first = area
            largest_extent = list(area.area_extent)
        else:
            if not area.crs == first.crs:
                raise ValueError("Inconsistent projections between areas")
            if not np.isclose(area.resolution, first.resolution).all():
                raise ValueError("Inconsistent resolution between areas")
            largest_extent[0] = min(largest_extent[0], area.area_extent[0])
            largest_extent[1] = min(largest_extent[1], area.area_extent[1])
            largest_extent[2] = max(largest_extent[2], area.area_extent[2])
            largest_extent[3] = max(largest_extent[3], area.area_extent[3])

    return create_area_def(
        area_id=area_id,
        projection=first.crs,
        area_extent=largest_extent,
        resolution=first.resolution)


def _numpy_values_to_native(values):
    return [n.item() if isinstance(n, np.number) else n for n in values]
