#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2020 Pyresample developers
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

import hashlib
import math
import warnings
from collections import OrderedDict
from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
import pyproj
import yaml
from pyproj import Geod, transform

from pyresample import CHUNK_SIZE
from pyresample._spatial_mp import Cartesian, Cartesian_MP, Proj, Proj_MP
from pyresample.area_config import create_area_def
from pyresample.boundary import AreaDefBoundary, Boundary, SimpleBoundary
from pyresample.utils import (
    check_slice_orientation,
    get_geostationary_height,
    load_cf_area,
    proj4_dict_to_str,
    proj4_radius_parameters,
)
from pyresample.utils.array import _convert_2D_array

try:
    from xarray import DataArray
except ImportError:
    DataArray = np.ndarray

try:
    import dask.array as da
except ImportError:
    da = None

from pyproj import CRS

logger = getLogger(__name__)
HashType = hashlib._hashlib.HASH


class DimensionError(ValueError):
    """Wrap ValueError."""

    pass


class IncompatibleAreas(ValueError):
    """Error when the areas to combine are not compatible."""

    pass


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
        if type(lons) != type(lats):
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

    def update_hash(self, existing_hash: Optional[HashType] = None) -> HashType:
        """Update the hash."""
        if existing_hash is None:
            existing_hash = hashlib.sha1()
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
        if other.lons is None or other.lats is None:
            other_lons, other_lats = other.get_lonlats()
        else:
            other_lons = other.lons
            other_lats = other.lats

        if self.lons is None or self.lats is None:
            self_lons, self_lats = self.get_lonlats()
        else:
            self_lons = self.lons
            self_lats = self.lats

        if self_lons is other_lons and self_lats is other_lats:
            return True
        if isinstance(self_lons, DataArray) and np.ndarray is not DataArray:
            self_lons = self_lons.data
            self_lats = self_lats.data
        if isinstance(other_lons, DataArray) and np.ndarray is not DataArray:
            other_lons = other_lons.data
            other_lats = other_lats.data
        try:
            from dask.array import allclose
        except ImportError:
            from numpy import allclose
        try:
            lons_close = allclose(self_lons, other_lons, atol=1e-6, rtol=5e-9, equal_nan=True)
            if not lons_close:
                return False
            lats_close = allclose(self_lats, other_lats, atol=1e-6, rtol=5e-9, equal_nan=True)
            return lats_close
        except (AttributeError, ValueError):
            return False

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
                      "'get_lonlats' with the 'chunks' keyword argument specified.", DeprecationWarning)
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

    def get_bbox_lonlats(self, force_clockwise: bool = True) -> tuple:
        """Return the bounding box lons and lats.

        Args:
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
        s1_lon, s1_lat = self.get_lonlats(data_slice=(0, slice(None)))
        s2_lon, s2_lat = self.get_lonlats(data_slice=(slice(None), -1))
        s3_lon, s3_lat = self.get_lonlats(data_slice=(-1, slice(None, None, -1)))
        s4_lon, s4_lat = self.get_lonlats(data_slice=(slice(None, None, -1), 0))
        lons, lats = zip(*[(s1_lon.squeeze(), s1_lat.squeeze()),
                           (s2_lon.squeeze(), s2_lat.squeeze()),
                           (s3_lon.squeeze(), s3_lat.squeeze()),
                           (s4_lon.squeeze(), s4_lat.squeeze())])
        if hasattr(lons[0], 'compute') and da is not None:
            lons, lats = da.compute(lons, lats)
        if force_clockwise and not self._corner_is_clockwise(
                lons[0][-2], lats[0][-2], lons[0][-1], lats[0][-1], lons[1][1], lats[1][1]):
            # going counter-clockwise
            lons, lats = self._reverse_boundaries(lons, lats)
        return lons, lats

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
                          "future and data will not be cached.", PendingDeprecationWarning)

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
        """Return the corners of the current area."""
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


class CoordinateDefinition(BaseDefinition):
    """Base class for geometry definitions defined by lons and lats only."""

    def __init__(self, lons, lats, nprocs=1):
        """Initialize CoordinateDefinition."""
        if not isinstance(lons, (np.ndarray, DataArray)):
            lons = np.asanyarray(lons)
            lats = np.asanyarray(lats)
        super(CoordinateDefinition, self).__init__(lons, lats, nprocs)
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
        from pyproj import transform
        rows = self.shape[0]
        start_row = rows // 2  # middle row
        src = Proj('+proj=latlong +datum=WGS84')
        if radius:
            dst = Proj("+proj=cart +a={} +b={}".format(radius, radius))
        else:
            dst = Proj("+proj=cart +ellps={}".format(ellps))
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

        xyz = np.stack(transform(src, dst, lons, lats, alt), axis=1)
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
        super(GridDefinition, self).__init__(lons, lats, nprocs)
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
            return np.asarray(arr).view(np.uint8)  # np array


class SwathDefinition(CoordinateDefinition):
    """Swath defined by lons and lats.

    Parameters
    ----------
    lons : numpy array
    lats : numpy array
    nprocs : int, optional
        Number of processor cores to be used for calculations.

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

    def __init__(self, lons, lats, nprocs=1):
        """Initialize SwathDefinition."""
        if not isinstance(lons, (np.ndarray, DataArray)):
            lons = np.asanyarray(lons)
            lats = np.asanyarray(lats)
        super(SwathDefinition, self).__init__(lons, lats, nprocs)
        if lons.shape != lats.shape:
            raise ValueError('lon and lat arrays must have same shape')
        elif lons.ndim > 2:
            raise ValueError('Only 1 and 2 dimensional swaths are allowed')

    def copy(self):
        """Copy the current swath."""
        return SwathDefinition(self.lons, self.lats)

    @staticmethod
    def _do_transform(src, dst, lons, lats, alt):
        """Perform pyproj transformation and stack the results.

        If using pyproj >= 3.1, it employs thread-safe pyproj.transformer.Transformer.
        If using pyproj < 3.1, it employs pyproj.transform.

        Docs: https://pyproj4.github.io/pyproj/stable/advanced_examples.html#multithreading
        """
        if float(pyproj.__version__[0:3]) >= 3.1:
            from pyproj import Transformer
            transformer = Transformer.from_crs(src.crs, dst.crs)
            x, y, z = transformer.transform(lons, lats, alt, radians=False)
        else:
            x, y, z = transform(src, dst, lons, lats, alt)
        return np.dstack((x, y, z))

    def aggregate(self, **dims):
        """Return an aggregated version of the area."""
        warnings.warn("'aggregate' is deprecated, use 'downsample' instead.", PendingDeprecationWarning)
        return self.downsample(x=dims.get('x', 1), y=dims.get('y', 1))

    def downsample(self, x=1, y=1, **kwargs):
        """Downsample the SwathDefinition along x (columns) and y (lines) dimensions.

        Builds upon xarray.DataArray.coarsen averaging function.
        To downsample of a factor of 2, call swath_def.downsample(x=2, y=2)
        swath_def.downsample(x=1, y=1) simply returns the current swath_def.
        By default, it raise a ValueError if the dimension size is not a multiple of the window size.
        This can be changed by passing boundary="trim" or boundary="pad", but behaviour within pyresample is undefined.
        See https://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html for further details.
        """
        import dask.array as da
        import xarray as xr

        # Check input validity
        x = int(x)
        y = int(y)
        if x < 1 or y < 1:
            raise ValueError('SwathDefinition.downsample expects (integer) aggregation factors >=1 .')

        # Return SwathDefinition if nothing to downsample
        if x == 1 and y == 1:
            return self

        # Define geodetic and geocentric projection
        geocent = pyproj.Proj(proj='geocent')
        latlong = pyproj.Proj(proj='latlong')

        # Get xr.DataArray with dask array
        # - If input lats/lons are xr.DataArray, the specified dims ['x','y'] are ignored
        src_lons, src_lons_format = _convert_2D_array(self.lons, to='DataArray_Dask', dims=['y', 'x'])
        src_lats, src_lats_format = _convert_2D_array(self.lats, to='DataArray_Dask', dims=['y', 'x'])

        # Conversion to Geocentric Cartesian (x,y,z) CRS
        res = da.map_blocks(self._do_transform, latlong, geocent,
                            src_lons.data,
                            src_lats.data,
                            da.zeros_like(src_lons),  # altitude
                            new_axis=[2],
                            chunks=(src_lons.chunks[0], src_lons.chunks[1], 3))
        res = xr.DataArray(res, dims=['y', 'x', 'xyz'], coords=src_lons.coords)

        # Aggregating
        res = res.coarsen(x=x, y=y, **kwargs).mean()

        # Back-conversion to geographic CRS
        lonlatalt = da.map_blocks(self._do_transform, geocent, latlong,
                                  res[:, :, 0].data,  # x
                                  res[:, :, 1].data,  # y
                                  res[:, :, 2].data,  # z
                                  new_axis=[2],
                                  chunks=res.data.chunks)

        # Back-conversion array as input format
        lons, _ = _convert_2D_array(lonlatalt[:, :, 0], to=src_lons_format, dims=src_lons.dims)
        lats, _ = _convert_2D_array(lonlatalt[:, :, 1], to=src_lats_format, dims=src_lats.dims)

        # Add additional info if the source array is a DataArray
        if isinstance(self.lats, xr.DataArray) and isinstance(self.lons, xr.DataArray):
            lats = lats.assign_coords(res.coords)
            lons = lons.assign_coords(res.coords)
            lats.attrs = self.lats.attrs.copy()
            lons.attrs = self.lons.attrs.copy()
            try:
                resolution = lons.attrs['resolution'] * ((x + y) / 2)
                lons.attrs['resolution'] = resolution
                lats.attrs['resolution'] = resolution
            except KeyError:
                pass

        # Return the downsampled swath definition
        return SwathDefinition(lons, lats)

    def upsample(self, x=1, y=1):
        """Upsample the SwathDefinition along x (columns) and y (lines) dimensions.

        To upsample of a factor of 2 (each pixel splitted in 2x2 pixels),
        call swath_def.upsample(x=2, y=2).
        swath_def.upsample(x=1, y=1) simply returns the current swath_def.
        """
        # TODO: An alternative would be to use geotiepoints.geointerpolator.GeoInterpolator
        # But I have some problem using it, see code snippet in a comment of the PR.
        # TODO: Should we upsample also possible coords of lons/lats input xr.DataArray?
        import dask.array as da
        import numpy as np
        import xarray as xr
        from xarray.plot.utils import _infer_interval_breaks

        # https://github.com/pydata/xarray/blob/main/xarray/plot/utils.py#L784
        # Check input validity
        x = int(x)
        y = int(y)
        if x < 1 or y < 1:
            raise ValueError("SwathDefinition.upsample expects (integer) upscaling factors >=1 .")
        # Return SwathDefinition if nothing to upsample
        if x == 1 and y == 1:
            return self
        # --------------------------------------------------------------------.
        # TODO:
        # - Refactor for dask-compatibility
        # - Should we make _infer_interval_breaks dask-compatible?

        def _get_corners_from_centroids(centroids):
            breaks_xx = _infer_interval_breaks(centroids, axis=1)
            corners = _infer_interval_breaks(breaks_xx, axis=0)
            return corners

        # TODO: choose one of the two function below
        # - What is the best way to apply _upsample_centroid along each x-y plane with dask
        def _upsample_centroid(centroid, x=1, y=1):
            corners = _get_corners_from_centroids(centroid)
            # Retrieve corners of the the upsampled grid
            new_corners = _linspace2D_between_values(corners, num_x=x - 1, num_y=y - 1)
            # Get centroids from corners
            new_centroids = (new_corners[:-1, :-1] + new_corners[1:, 1:]) / 2
            return new_centroids

        def upsample_centroids(centroid_x, centroid_y, centroid_z, x=1, y=1):
            x_new_centroids = _upsample_centroid(centroid_x, x=x, y=y)
            y_new_centroids = _upsample_centroid(centroid_y, x=x, y=y)
            z_new_centroids = _upsample_centroid(centroid_z, x=x, y=y)
            return x_new_centroids, y_new_centroids, z_new_centroids

        # --------------------------------------------------------------------.
        # Define geodetic and geocentric projection
        geocent = pyproj.Proj(proj='geocent')
        latlong = pyproj.Proj(proj='latlong')

        # Get xr.DataArray with dask array
        # - If input lats/lons are xr.DataArray, the specified dims ['x','y'] are ignored
        src_lons, src_lons_format = _convert_2D_array(self.lons, to='DataArray_Dask', dims=['y', 'x'])
        src_lats, src_lats_format = _convert_2D_array(self.lats, to='DataArray_Dask', dims=['y', 'x'])

        # Conversion to Geocentric Cartesian (x,y,z) CRS
        res = da.map_blocks(self._do_transform, latlong, geocent,
                            src_lons.data,
                            src_lats.data,
                            da.zeros_like(src_lons),  # altitude
                            new_axis=[2],
                            chunks=(src_lons.chunks[0], src_lons.chunks[1], 3))
        res = xr.DataArray(res, dims=['y', 'x', 'xyz'])

        # Retrieve new centroids
        # TODO: make it dask compatible using _upsample_centroid_dask [HELP WANTED]
        # res1 = da.apply_along_axis(_upsample_centroid_dask,
        #                            2,
        #                            res.data,
        #                            x,
        #                            y)
        # res1 = xr.DataArray(res1, dims=['y', 'x', 'coord'], coords=src_lons.coords)
        res = np.stack(upsample_centroids(res[:, :, 0].data,
                                          res[:, :, 1].data,
                                          res[:, :, 2].data, x=x, y=y), axis=2)
        new_centroids = xr.DataArray(da.from_array(res), dims=['y', 'x', 'xyz'])

        # Back-conversion to geographic CRS
        lonlatalt = da.map_blocks(self._do_transform, geocent, latlong,
                                  new_centroids[:, :, 0].data,  # x
                                  new_centroids[:, :, 1].data,  # y
                                  new_centroids[:, :, 2].data,  # z
                                  new_axis=[2],
                                  chunks=new_centroids.data.chunks)

        # Back-conversion array as input format
        lons, _ = _convert_2D_array(lonlatalt[:, :, 0], to=src_lons_format, dims=src_lons.dims)
        lats, _ = _convert_2D_array(lonlatalt[:, :, 1], to=src_lats_format, dims=src_lats.dims)

        # Add additional info if the source array is a DataArray
        if isinstance(self.lats, xr.DataArray) and isinstance(self.lons, xr.DataArray):
            lats.attrs = self.lats.attrs.copy()
            lons.attrs = self.lons.attrs.copy()
            try:
                resolution = lons.attrs['resolution'] / ((x + y) / 2)
                lons.attrs['resolution'] = resolution
                lats.attrs['resolution'] = resolution
            except KeyError:
                pass

        # Return the downsampled swath definition
        return SwathDefinition(lons, lats)

    def extend(self, left=0, right=0, bottom=0, top=0):
        """Extend the SwathDefinition of n pixels on specific boundary sides.

        By default, it does not extend on any side.
        """
        import xarray as xr

        # Check input validity
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        top = int(top)
        if left < 0 or right < 0 or bottom < 0 or top < 0:
            raise ValueError('SwathDefinition.extend expects positive numbers of pixels.')

        # Return SwathDefinition if nothing to extend
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            return self

        # Get lats/lons numpy arrays
        src_lats, src_lats_format = _convert_2D_array(self.lats, to='numpy', dims=['y', 'x'])
        src_lons, src_lons_format = _convert_2D_array(self.lons, to='numpy', dims=['y', 'x'])

        dst_lats = src_lats
        dst_lons = src_lons

        # Extend swath sides
        if top > 0:
            list_side0 = (dst_lons[1, :], dst_lats[1, :], dst_lons[0, :], dst_lats[0, :])
            extended_side0_lonlats = _get_extended_lonlats(*list_side0, npts=top)
            dst_lats = np.concatenate((extended_side0_lonlats[1][::-1, :], dst_lats), axis=0)
            dst_lons = np.concatenate((extended_side0_lonlats[0][::-1, :], dst_lons), axis=0)

        if bottom > 0:
            list_side2 = (dst_lons[-2, :], dst_lats[-2, :], dst_lons[-1, :], dst_lats[-1, :])
            extended_side2_lonlats = _get_extended_lonlats(*list_side2, npts=bottom)
            dst_lats = np.concatenate((dst_lats, extended_side2_lonlats[1]), axis=0)
            dst_lons = np.concatenate((dst_lons, extended_side2_lonlats[0]), axis=0)

        if right > 0:
            list_side1 = (dst_lons[:, -2], dst_lats[:, -2], dst_lons[:, -1], dst_lats[:, -1])
            extended_side1_lonlats = _get_extended_lonlats(*list_side1, npts=right, transpose=False)
            dst_lats = np.concatenate((dst_lats, extended_side1_lonlats[1]), axis=1)
            dst_lons = np.concatenate((dst_lons, extended_side1_lonlats[0]), axis=1)

        if left > 0:
            list_side3 = (dst_lons[:, 1], dst_lats[:, 1], dst_lons[:, 0], dst_lats[:, 0])
            extended_side3_lonlats = _get_extended_lonlats(*list_side3, npts=left, transpose=False)
            dst_lats = np.concatenate((extended_side3_lonlats[1][:, ::-1], dst_lats), axis=1)
            dst_lons = np.concatenate((extended_side3_lonlats[0][:, ::-1], dst_lons), axis=1)

        # Back-conversion array as input format
        lons, _ = _convert_2D_array(dst_lons, to=src_lons_format, dims=['y', 'x'])
        lats, _ = _convert_2D_array(dst_lats, to=src_lats_format, dims=['y', 'x'])

        # Add additional info if the source array is a DataArray
        if isinstance(self.lats, xr.DataArray) and isinstance(self.lons, xr.DataArray):
            lats.attrs = self.lats.attrs.copy()
            lons.attrs = self.lons.attrs.copy()

        # Return the extended SwathDefinition
        return SwathDefinition(lons, lats)

    def shrink(self, left=0, right=0, bottom=0, top=0):
        """Shrink the SwathDefinition of n pixels on specific boundary sides.

        By default, it does not shrink on any side.
        """
        # Check input validity
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        top = int(top)
        if left < 0 or right < 0 or bottom < 0 or top < 0:
            raise ValueError('SwathDefinition.shrink expects positive numbers of pixels.')

        # Return SwathDefinition if nothing to shrink
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            return self

        # Ensure shrinked area is at least 2x2
        height = self.lats.shape[0]
        width = self.lats.shape[1]
        x_max_shrink = width - 2
        y_max_shrink = height - 2
        if (left + right) > x_max_shrink:
            raise ValueError("SwathDefinition.shrink can drop maximum {} pixels "
                             "along the x direction.".format(x_max_shrink))
        if (top + bottom) > y_max_shrink:
            raise ValueError("SwathDefinition.shrink can drop maximum {} pixels "
                             "along the y direction.".format(y_max_shrink))

        # Return the shrinked SwathDefinition
        return self[slice(top, height - bottom), slice(left, width - right)]

    def __hash__(self):
        """Compute the hash of this object."""
        if self.hash is None:
            self.hash = int(self.update_hash().hexdigest(), 16)
        return self.hash

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

    def get_edge_lonlats(self):
        """Get the concatenated boundary of the current swath."""
        lons, lats = self.get_bbox_lonlats(force_clockwise=False)
        blons = np.ma.concatenate(lons)
        blats = np.ma.concatenate(lats)
        return blons, blats

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

    def _compute_uniform_shape(self):
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
        vresolution = height * 1.0 / self.lons.shape[0]
        hresolution = width * 1.0 / self.lons.shape[1]
        resolution = min(vresolution, hresolution)
        width = int(width * 1.1 / resolution)
        height = int(height * 1.1 / resolution)
        return height, width

    def compute_optimal_bb_area(self, proj_dict=None):
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
        height, width = self._compute_uniform_shape()
        proj_dict = self.compute_bb_proj_params(proj_dict)

        area = DynamicAreaDefinition(area_id, description, proj_dict)
        lons, lats = self.get_edge_lonlats()
        return area.freeze((lons, lats), shape=(height, width))


def _linspace2D_between_values(arr, num_x=0, num_y=0):
    """Dask-friendly function linearly interpolating values between each 2D array values.

    This function does not perform extrapolation.
    It expects a 2D array as input!

    Parameters
    ----------
    arr : (np.ndarray, dask.array.Array)
        Numpy or Dask Array to be linearly interpolated between values.
    num_x : int, optional
        The number of linearly spaced values to infer between array values (along x).
        . The default is 0.
    num_y : int, optional
        The number of linearly spaced values to infer between array values (along y).
        The default is 0.

    Returns
    -------
    arr : (np.ndarray, dask.array.Array)
        Numpy or Dask Array with in-between linearly interpolated values.

    Example
    -------

    Function call: _linspace2D_between_values(arr, num_x=1, num_y=1)
    Input:
       np.array([[5.0, 7.0],
                 [7.0, 9.0]])
    Output:
        np.array([[5.0, 6.0, 7.0],
                  [6.0, 7.0, 8.0],
                  [7.0, 8.0, 9.0]])
    """
    import xarray as xr

    # Check input validity
    if arr.ndim != 2:
        raise ValueError("'_linspace2D_between_values' expects a 2D array.")
    num_x = int(num_x)
    num_y = int(num_y)
    if num_x < 0 or num_y < 0:
        raise ValueError("'x' and 'y' must be an integer equal or larger than 0.")
    if num_x == 0 and num_y == 0:
        return arr
    # Define src and dst ties
    shape = arr.shape
    Nx_dst = (shape[1] - 1) * (num_x + 1) + 1
    Ny_dst = (shape[0] - 1) * (num_y + 1) + 1

    src_ties_x = np.arange(Nx_dst, step=num_x + 1)
    src_ties_y = np.arange(Ny_dst, step=num_y + 1)
    dst_ties_x = np.arange(Nx_dst)
    dst_ties_y = np.arange(Ny_dst)
    # Interpolate
    da = xr.DataArray(
        data=arr,
        dims=("y", "x"),
        coords={"y": src_ties_y, "x": src_ties_x}
    )
    da_interp = da.interp({"y": dst_ties_y, "x": dst_ties_x}, method="linear")
    return da_interp.data


def _get_extended_lonlats(lon_start, lat_start, lon_end, lat_end, npts,
                          ellps="sphere",
                          transpose=True):
    """Utils employed by SwathDefinition.extend.

    It extrapolate npts following the forward azimuth with an interdistance
    equal to the distance between the starting point and the end point.
    """
    geod = pyproj.Geod(ellps=ellps)
    #  geod = pyproj.Geod(ellps='WGS84') # sphere
    az12_arr, _, dist_arr = geod.inv(lon_start, lat_start, lon_end, lat_end)
    list_lat = []
    list_lon = []
    for lon, lat, az12, dist in zip(lon_end, lat_end, az12_arr, dist_arr):
        points = geod.fwd_intermediate(lon, lat, az12, del_s=dist, npts=npts,
                                       out_lons=None, out_lats=None, radians=False)
        list_lat.append(points.lats)
        list_lon.append(points.lons)

    new_lats = np.stack(list_lat)
    new_lons = np.stack(list_lon)
    if transpose:
        new_lats = new_lats.T
        new_lons = new_lons.T
    return new_lons, new_lats


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
        rotation:
            Rotation in degrees (negative is cw)

    """

    def __init__(self, area_id=None, description=None, projection=None,
                 width=None, height=None, area_extent=None,
                 resolution=None, optimize_projection=False, rotation=None):
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
        self.rotation = rotation
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
        """Return pixel size in X direction."""
        if self.resolution is None:
            return None
        return self.resolution[0]

    @property
    def pixel_size_y(self):
        """Return pixel size in Y direction."""
        if self.resolution is None:
            return None
        return self.resolution[1]

    def compute_domain(self, corners, resolution=None, shape=None):
        """Compute shape and area_extent from corners and [shape or resolution] info.

        Corners represents the center of pixels, while area_extent represents the edge of pixels.

        Note that ``shape`` is (rows, columns) and ``resolution`` is
        (x_size, y_size); the dimensions are flipped.
        """
        if resolution is not None and shape is not None:
            raise ValueError("Both resolution and shape can't be provided.")
        elif resolution is None and shape is None:
            raise ValueError("Either resolution or shape must be provided.")

        if shape:
            height, width = shape
            x_resolution = (corners[2] - corners[0]) * 1.0 / (width - 1)
            y_resolution = (corners[3] - corners[1]) * 1.0 / (height - 1)
        else:
            if isinstance(resolution, (int, float)):
                resolution = (resolution, resolution)
            x_resolution, y_resolution = resolution
            width = int(np.rint((corners[2] - corners[0]) * 1.0 / x_resolution + 1))
            height = int(np.rint((corners[3] - corners[1]) * 1.0 / y_resolution + 1))

        area_extent = (corners[0] - x_resolution / 2,
                       corners[1] - y_resolution / 2,
                       corners[2] + x_resolution / 2,
                       corners[3] + y_resolution / 2)
        return area_extent, width, height

    def freeze(self, lonslats=None, resolution=None, shape=None, proj_info=None):
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

        Resolution and shape parameters are ignored if the instance is created
        with the `optimize_projection` flag set to True.
        """
        proj_dict = self._get_proj_dict()
        projection = self._projection
        if proj_info is not None:
            # this is now our complete projection information
            proj_dict.update(proj_info)
            projection = proj_dict

        if self.optimize_projection:
            return lonslats.compute_optimal_bb_area(proj_dict)
        if resolution is None:
            resolution = self.resolution
        if shape is None:
            shape = self.shape
        height, width = shape
        shape = None if None in shape else shape
        area_extent = self.area_extent
        if not area_extent or not width or not height:
            corners = self._compute_bound_centers(proj_dict, lonslats)
            area_extent, width, height = self.compute_domain(corners, resolution, shape)
        return AreaDefinition(self.area_id, self.description, '',
                              projection, width, height,
                              area_extent, self.rotation)

    def _compute_bound_centers(self, proj_dict, lonslats):
        lons, lats = self._extract_lons_lats(lonslats)
        if hasattr(lons, 'compute'):
            return self._compute_bound_centers_dask(proj_dict, lons, lats)
        return self._compute_bound_centers_numpy(proj_dict, lons, lats)

    def _compute_bound_centers_numpy(self, proj_dict, lons, lats):
        # TODO: Do more dask-friendly things here
        proj4 = Proj(proj_dict)
        xarr, yarr = proj4(np.asarray(lons), np.asarray(lats))
        xarr[xarr > 9e29] = np.nan
        yarr[yarr > 9e29] = np.nan
        xmin = np.nanmin(xarr)
        xmax = np.nanmax(xarr)
        ymin = np.nanmin(yarr)
        ymax = np.nanmax(yarr)
        x_passes_antimeridian = (xmax - xmin) > 355
        epsilon = 0.1
        y_is_pole = (ymax >= 90 - epsilon) or (ymin <= -90 + epsilon)
        if proj4.crs.is_geographic and x_passes_antimeridian and not y_is_pole:
            # cross anti-meridian of projection
            xmin = np.nanmin(xarr[xarr >= 0])
            xmax = np.nanmax(xarr[xarr < 0]) + 360
        return xmin, ymin, xmax, ymax

    def _compute_bound_centers_dask(self, proj_dict, lons, lats):
        import dask.array as da

        from pyresample.utils.proj4 import DaskFriendlyTransformer
        crs = CRS(proj_dict)
        transformer = DaskFriendlyTransformer.from_crs(CRS(4326), crs,
                                                       always_xy=True)
        xarr, yarr = transformer.transform(lons, lats)
        xarr = da.where(xarr > 9e29, np.nan, xarr)
        yarr = da.where(yarr > 9e29, np.nan, yarr)
        _xmin = np.nanmin(xarr)
        _xmax = np.nanmax(xarr)
        _ymin = np.nanmin(yarr)
        _ymax = np.nanmax(yarr)
        xmin, xmax, ymin, ymax = da.compute(
            _xmin,
            _xmax,
            _ymin,
            _ymax)

        x_passes_antimeridian = (xmax - xmin) > 355
        epsilon = 0.1
        y_is_pole = (ymax >= 90 - epsilon) or (ymin <= -90 + epsilon)
        if crs.is_geographic and x_passes_antimeridian and not y_is_pole:
            # cross anti-meridian of projection
            xarr_pos = da.where(xarr >= 0, xarr, np.nan)
            xarr_neg = da.where(xarr < 0, xarr, np.nan)
            xmin = np.nanmin(xarr_pos)
            xmax = np.nanmax(xarr_neg) + 360
            xmin, xmax = da.compute(xmin, xmax)
        return xmin, ymin, xmax, ymax

    def _extract_lons_lats(self, lonslats):
        try:
            lons, lats = lonslats
        except (TypeError, ValueError):
            lons, lats = lonslats.get_lonlats()
        return lons, lats


def _invproj(data_x, data_y, proj_dict):
    """Perform inverse projection."""
    # XXX: does pyproj copy arrays? What can we do so it doesn't?
    target_proj = Proj(proj_dict)
    lon, lat = target_proj(data_x, data_y, inverse=True)
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
        warnings.warn("'x_size' is deprecated, use 'width' instead.", PendingDeprecationWarning)
        return self.width

    @property
    def y_size(self):
        """Return area height."""
        warnings.warn("'y_size' is deprecated, use 'height' instead.", PendingDeprecationWarning)
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
        x__ = np.round(x__).astype(int)
        y__ = np.round(y__).astype(int)

        x_mask = ((x__ < 0) | (x__ >= self.width))
        y_mask = ((y__ < 0) | (y__ >= self.height))
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
    rotation: float, optional
        rotation in degrees (negative is clockwise)
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
    rotation: float
        rotation in degrees (negative is cw)
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
                 area_extent, rotation=None, nprocs=1, lons=None, lats=None,
                 dtype=np.float64):
        """Initialize AreaDefinition."""
        super(AreaDefinition, self).__init__(lons, lats, nprocs)
        self.area_id = area_id
        self.description = description
        self.proj_id = proj_id
        self.width = int(width)
        self.height = int(height)
        self.crop_offset = (0, 0)
        try:
            self.rotation = float(rotation)
        except TypeError:
            self.rotation = 0
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
                  'rotation': self.rotation}
        kwargs.update(override_kwargs)
        return AreaDefinition(**kwargs)

    def aggregate(self, **dims):
        """Return an aggregated version of the area.

        Aggregate allows to mix between downsample and upsample in different directions.
        Example: area_def.aggregate(x=2, y=0.5) <-> area_def.downsample(x=2).upsample(y=2).
        """
        x = dims.get('x', 1)
        y = dims.get('y', 1)
        if x <= 0 or y <= 0:
            raise ValueError('AreaDefinition.aggregate x and y arguments must be > 0.')
        if x == 1 and y == 1:
            return self
        width = int(self.width / x)
        height = int(self.height / y)
        return self.copy(height=height, width=width)

    def downsample(self, x=1, y=1):
        """Return a downsampled version of the area."""
        # Check input validity
        if x == 1 and y == 1:
            return self
        if x < 1 or y < 1:
            raise ValueError('AreaDefinition.downsample x and y arguments must be >= 1.')
        # Downsample
        return self.aggregate(x=x, y=y)

    def upsample(self, x=1, y=1):
        """Return an upsampled version of the area."""
        # Check input validity
        if x == 1 and y == 1:
            return self
        if x < 1 or y < 1:
            raise ValueError('AreaDefinition.upsample x and y arguments must be >= 1.')
        # Upsample
        return self.aggregate(x=1 / x, y=1 / y)

    def extend(self, left=0, right=0, bottom=0, top=0):
        """Extend AreaDefinition by n pixels on specific boundary sides.

        By default, it does not extend on any side.
        """
        if self.is_geostationary:
            raise NotImplementedError("AreaDefinition.extend method is not implemented for GEO AreaDefinition.")
        # Check input validity
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        top = int(top)
        if left < 0 or right < 0 or bottom < 0 or top < 0:
            raise ValueError('AreaDefinition.extend expects positive numbers of pixels.')

        # Return AreaDefinition if nothing to extend
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            return self

        # Retrieve pixel and area info
        new_width = self.width + left + right
        new_height = self.height + bottom + top
        pixel_size_x = self.pixel_size_x
        pixel_size_y = self.pixel_size_y

        # Extend area_extent (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        area_extent = self._area_extent
        new_area_extent = list(area_extent)
        new_area_extent[0] = new_area_extent[0] - pixel_size_x * left
        new_area_extent[1] = new_area_extent[1] - pixel_size_y * bottom
        new_area_extent[2] = new_area_extent[2] + pixel_size_x * right
        new_area_extent[3] = new_area_extent[3] + pixel_size_y * top

        # Define new AreaDefinition
        projection = self.crs_wkt
        area_def = AreaDefinition(self.area_id, self.description, self.proj_id,
                                  projection=projection,
                                  width=new_width,
                                  height=new_height,
                                  area_extent=new_area_extent,
                                  rotation=self.rotation,
                                  nprocs=self.nprocs,
                                  dtype=self.dtype)

        return area_def

    def shrink(self, left=0, right=0, bottom=0, top=0):
        """Shrink AreaDefinition by n pixels on specific boundary sides.

        By default, it does not shrink on any side.
        """
        if self.is_geostationary:
            raise NotImplementedError("AreaDefinition.shrink method is not implemented for GEO AreaDefinition.")
        # Check input validity
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        top = int(top)
        if left < 0 or right < 0 or bottom < 0 or top < 0:
            raise ValueError('AreaDefinition.shrink expects positive numbers of pixels.')

        # Return AreaDefinition if nothing to extend
        if left == 0 and right == 0 and bottom == 0 and top == 0:
            return self

        # Ensure shrinked area is at least 2x2
        width = self.width
        height = self.height
        x_max_shrink = width - 2
        y_max_shrink = height - 2
        if (left + right) > x_max_shrink:
            raise ValueError("AreaDefinition.shrink can drop maximum {} pixels "
                             "along the x direction.".format(x_max_shrink))
        if (top + bottom) > y_max_shrink:
            raise ValueError("AreaDefinition.shrink can drop maximum {} pixels "
                             "along the y direction.".format(y_max_shrink))

        return self[slice(top, height - bottom), slice(left, width - right)]

    @property
    def resolution(self):
        """Return area resolution in X and Y direction."""
        return self.pixel_size_x, self.pixel_size_y

    @property
    def name(self):
        """Return area name."""
        warnings.warn("'name' is deprecated, use 'description' instead.", PendingDeprecationWarning)
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
        return create_area_def(crs.name, crs.to_dict(), area_extent=area_extent, resolution=resolution)

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
        rotation: float, optional
            rotation in degrees (negative is cw)
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
        rotation: float, optional
            rotation in degrees (negative is cw)
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
        rotation: float, optional
            rotation in degrees (negative is cw)
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
        rotation: float, optional
            rotation in degrees (negative is cw)
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

    def create_areas_def(self):
        """Generate YAML formatted representation of this area."""
        warnings.warn("'create_areas_def' is deprecated. Please use `dump` instead, which also "
                      "supports writing directly to a file.", DeprecationWarning)

        return self.dump()

    def dump(self, filename=None):
        """Generate YAML formatted representation of this area.

        Args:
            filename (str or pathlib.Path or file-like object): Yaml file location to dump the area to.

        Returns:
            If file is None returns yaml str
        """
        if self.crs.to_epsg() is not None:
            proj_dict = {'EPSG': self.crs.to_epsg()}
        else:
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
                      "Use the 'YAML' format instead.")
        proj_dict = self.proj_dict
        proj_str = ','.join(["%s=%s" % (str(k), str(proj_dict[k]))
                             for k in sorted(proj_dict.keys())])

        fmt = "REGION: {name} {{\n"
        fmt += "\tNAME:\t{name}\n"
        fmt += "\tPCS_ID:\t{area_id}\n"
        fmt += "\tPCS_DEF:\t{proj_str}\n"
        fmt += "\tXSIZE:\t{x_size}\n"
        fmt += "\tYSIZE:\t{y_size}\n"
        # fmt += "\tROTATION:\t{rotation}\n"
        fmt += "\tAREA_EXTENT: {area_extent}\n}};\n"
        area_def_str = fmt.format(name=self.description, area_id=self.area_id,
                                  proj_str=proj_str, x_size=self.width,
                                  y_size=self.height,
                                  area_extent=self.area_extent)
        return area_def_str

    def __eq__(self, other):
        """Test for equality."""
        try:
            return ((self.crs == other.crs) and
                    (self.shape == other.shape) and
                    (np.allclose(self.area_extent, other.area_extent)))
        except AttributeError:
            return super(AreaDefinition, self).__eq__(other)

    def __ne__(self, other):
        """Test for equality."""
        return not self.__eq__(other)

    def update_hash(self, existing_hash: Optional[HashType] = None) -> HashType:
        """Update a hash, or return a new one if needed."""
        if existing_hash is None:
            existing_hash = hashlib.sha1()
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
        p = Proj(self.proj_str)
        x = self.projection_x_coords
        y = self.projection_y_coords
        return p(x[cols], y[rows], inverse=True)

    def lonlat2colrow(self, lons, lats):
        """Return image columns and rows for the given lons and lats.

        Both scalars and arrays are supported.  Same as
        get_xy_from_lonlat, renamed for convenience.
        """
        warnings.warn("'lonlat2colrow' is deprecated, please use "
                      "'get_array_indices_from_lonlat' instead.", DeprecationWarning)

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
                      "'get_array_indices_from_lonlat' instead.", DeprecationWarning)

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
                      "'get_array_indices_from_projection_coordinates' instead.", DeprecationWarning)

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

    @staticmethod
    def _do_rotation(xspan, yspan, rot_deg=0):
        """Apply a rotation factor to a matrix of points."""
        rot_rad = np.radians(rot_deg)
        rot_mat = np.array([[np.cos(rot_rad), np.sin(rot_rad)], [-np.sin(rot_rad), np.cos(rot_rad)]])
        x, y = np.meshgrid(xspan, yspan)
        return np.einsum('ji, mni -> jmn', rot_mat, np.dstack([x, y]))

    def get_proj_vectors_dask(self, chunks=None, dtype=None):
        """Get projection vectors."""
        warnings.warn("'get_proj_vectors_dask' is deprecated, please use "
                      "'get_proj_vectors' with the 'chunks' keyword argument specified.", DeprecationWarning)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_proj_vectors(dtype=dtype, chunks=chunks)

    def _get_proj_vectors(self, dtype=None, check_rotation=True, chunks=None):
        """Get 1D projection coordinates."""
        if check_rotation and self.rotation != 0:
            warnings.warn("Projection vectors will not be accurate because rotation is not 0", RuntimeWarning)
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
                      "'get_proj_coords' with the 'chunks' keyword argument specified.", DeprecationWarning)
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
        if self.rotation != 0 and chunks is not None:
            raise ValueError("'rotation' is not supported with dask operations.")
        if dtype is None:
            dtype = self.dtype
        y_slice, x_slice = self._get_yx_data_slice(data_slice)
        if chunks is not None:
            target_x, target_y = self._proj_coords_dask(chunks, dtype)
            if y_slice is not None:
                target_x = target_x[y_slice, x_slice]
                target_y = target_y[y_slice, x_slice]
            return target_x, target_y

        target_x, target_y = self._get_proj_vectors(dtype=dtype, check_rotation=False, chunks=chunks)
        if y_slice is not None:
            target_y = target_y[y_slice]
        if x_slice is not None:
            target_x = target_x[x_slice]

        if self.rotation != 0:
            res = self._do_rotation(target_x, target_y, self.rotation)
            target_x, target_y = res[0, :, :], res[1, :, :]
        else:
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
        if self.rotation != 0:
            # rotation is only supported in 'get_proj_coords' right now
            return self.get_proj_coords(data_slice=(0, slice(None)))[0].squeeze()
        return self.get_proj_vectors()[0]

    @property
    def projection_y_coords(self):
        """Return projection Y coordinates."""
        if self.rotation != 0:
            # rotation is only supported in 'get_proj_coords' right now
            return self.get_proj_coords(data_slice=(slice(None), 0))[1].squeeze()
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
                      "'get_lonlats' with the 'chunks' keyword argument specified.", DeprecationWarning)
        if chunks is None:
            chunks = CHUNK_SIZE  # FUTURE: Use a global config object instead
        return self.get_lonlats(chunks=chunks, dtype=dtype)

    def get_lonlats(self, nprocs=None, data_slice=None, cache=False, dtype=None, chunks=None):
        """Return lon and lat arrays of area.

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
                          "future and data will not be cached.", PendingDeprecationWarning)
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
                             new_axis=[0], proj_dict=self.crs_wkt)
            lons, lats = res[0], res[1]
            return lons, lats

        if nprocs > 1:
            target_proj = Proj_MP(self.crs)
        else:
            target_proj = Proj(self.crs)

        # Get corresponding longitude and latitude values
        lons, lats = target_proj(target_x, target_y, inverse=True, nprocs=nprocs)
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
                      "instead.", DeprecationWarning)
        return proj4_dict_to_str(self.proj_dict)

    def _get_slice_starts_stops(self, area_to_cover):
        """Get x and y start and stop points for slicing."""
        llx, lly, urx, ury = area_to_cover.area_extent
        x, y = self.get_array_coordinates_from_projection_coordinates([llx, urx], [lly, ury])

        # we use `round` because we want the *exterior* of the pixels to contain the area_to_cover's area extent.
        if (self.area_extent[0] > self.area_extent[2]) ^ (llx > urx):
            xstart = max(0, round(x[1]))
            xstop = min(self.width, round(x[0]) + 1)
        else:
            xstart = max(0, round(x[0]))
            xstop = min(self.width, round(x[1]) + 1)
        if (self.area_extent[1] > self.area_extent[3]) ^ (lly > ury):
            ystart = max(0, round(y[0]))
            ystop = min(self.height, round(y[1]) + 1)
        else:
            ystart = max(0, round(y[1]))
            ystop = min(self.height, round(y[0]) + 1)

        return xstart, xstop, ystart, ystop

    def get_area_slices(self, area_to_cover, shape_divisible_by=None):
        """Compute the slice to read based on an `area_to_cover`."""
        if not isinstance(area_to_cover, AreaDefinition):
            raise NotImplementedError('Only AreaDefinitions can be used')

        # Intersection only required for two different projections
        proj_def_to_cover = area_to_cover.crs
        proj_def = self.crs
        if proj_def_to_cover == proj_def:
            logger.debug('Projections for data and slice areas are'
                         ' identical: %s',
                         proj_def_to_cover)
            # Get slice parameters
            xstart, xstop, ystart, ystop = self._get_slice_starts_stops(area_to_cover)

            x_slice = check_slice_orientation(slice(xstart, xstop))
            y_slice = check_slice_orientation(slice(ystart, ystop))
            x_slice = _ensure_integer_slice(x_slice)
            y_slice = _ensure_integer_slice(y_slice)
            return x_slice, y_slice

        if not self.is_geostationary:
            raise NotImplementedError("Source projection must be 'geos' if "
                                      "source/target projections are not "
                                      "equal.")

        data_boundary = Boundary(*get_geostationary_bounding_box(self))
        if area_to_cover.is_geostationary:
            area_boundary = Boundary(
                *get_geostationary_bounding_box(area_to_cover))
        else:
            area_boundary = AreaDefBoundary(area_to_cover, 100)

        intersection = data_boundary.contour_poly.intersection(
            area_boundary.contour_poly)
        if intersection is None:
            logger.debug('Cannot determine appropriate slicing. '
                         "Data and projection area do not overlap.")
            raise NotImplementedError
        x, y = self.get_array_indices_from_lonlat(
            np.rad2deg(intersection.lon), np.rad2deg(intersection.lat))
        x_slice = slice(np.ma.min(x), np.ma.max(x) + 1)
        y_slice = slice(np.ma.min(y), np.ma.max(y) + 1)
        x_slice = _ensure_integer_slice(x_slice)
        y_slice = _ensure_integer_slice(y_slice)
        if shape_divisible_by is not None:
            x_slice = _make_slice_divisible(x_slice, self.width,
                                            factor=shape_divisible_by)
            y_slice = _make_slice_divisible(y_slice, self.height,
                                            factor=shape_divisible_by)

        return (check_slice_orientation(x_slice),
                check_slice_orientation(y_slice))

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
            hor_res = np.mean(np.histogram_bin_edges(hor_dist)[:2])
        if vert_dist.size:
            vert_res = np.mean(np.histogram_bin_edges(vert_dist)[:2])
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


def _make_slice_divisible(sli, max_size, factor=2):
    """Make the given slice even in size."""
    rem = (sli.stop - sli.start) % factor
    if rem != 0:
        adj = factor - rem
        if sli.stop + 1 + rem < max_size:
            sli = slice(sli.start, sli.stop + adj)
        elif sli.start > 0:
            sli = slice(sli.start - adj, sli.stop)
        else:
            sli = slice(sli.start, sli.stop - rem)

    return sli


def _ensure_integer_slice(sli):
    start = sli.start
    stop = sli.stop
    step = sli.step
    return slice(
        math.floor(start) if start is not None else None,
        math.ceil(stop) if stop is not None else None,
        math.floor(step) if step is not None else None
    )


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

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    xmax = np.arccos(np.sqrt(aeq))
    ymax = np.arccos(np.sqrt(ap_))
    return xmax, ymax


def get_geostationary_bounding_box(geos_area, nb_points=50):
    """Get the bbox in lon/lats of the valid pixels inside `geos_area`.

    Args:
      nb_points: Number of points on the polygon
    """
    xmax, ymax = get_geostationary_angle_extent(geos_area)
    h = get_geostationary_height(geos_area.crs)

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    x = np.cos(np.linspace(-np.pi, 0, int(nb_points / 2.0))) * (xmax - 0.0001)
    y = -np.sin(np.linspace(-np.pi, 0, int(nb_points / 2.0))) * (ymax - 0.0001)

    ll_x, ll_y, ur_x, ur_y = geos_area.area_extent

    x *= h
    y *= h

    x = np.clip(np.concatenate([x, x[::-1]]), min(ll_x, ur_x), max(ll_x, ur_x))
    y = np.clip(np.concatenate([y, -y]), min(ll_y, ur_y), max(ll_y, ur_y))

    return Proj(geos_area.crs)(x, y, inverse=True)


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
        super(StackedAreaDefinition, self).__init__(nprocs=nprocs)
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
                      DeprecationWarning)
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
                      "instead.", DeprecationWarning)
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
