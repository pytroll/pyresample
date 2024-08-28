#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021-2022 Pyresample developers
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
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Area and Swath Slicers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from pyproj import Transformer
from pyproj.enums import TransformDirection

from pyresample import AreaDefinition, SwathDefinition
from pyresample.geometry import IncompatibleAreas, InvalidArea, get_geostationary_bounding_box_in_proj_coords

try:
    import dask.array as da
except ImportError:
    da = None


def create_slicer(area_to_crop, area_to_contain):
    """Create a slicer for cropping *area_to_crop* based on *area_to_contain*.

    Return an AreaSlicer or a SwathSlicer based on the first area type.
    """
    if isinstance(area_to_crop, SwathDefinition):
        return SwathSlicer(area_to_crop, area_to_contain)
    elif isinstance(area_to_crop, AreaDefinition):
        return AreaSlicer(area_to_crop, area_to_contain)
    else:
        raise NotImplementedError("Don't know how to slice a " + str(type(area_to_crop)))


class Slicer(ABC):
    """Abstract Slicer.

    Provided an Area-to-crop and an Area-to-contain, a Slicer provides methods
    to find slices that enclose `area-to-contain` inside `area-to-crop`.

    Example:
        For slicing a full-disk MSG area using a polar-stereographic area over Germany:

        >>> from pyresample import slicer
        >>> from satpy.resample import get_area_def
        >>> msg_area = get_area_def("msg_seviri_fes_3km")
        >>> germ_area = get_area_def("germ")
        >>> slc = slicer.create_slicer(msg_area, germ_area)
        >>> slc.get_slices()
        (slice(1900, 2242, None), slice(233, 423, None))

    """

    def __init__(self, area_to_crop, area_to_contain):
        """Set up the Slicer."""
        self.area_to_crop = area_to_crop
        self.area_to_contain = area_to_contain
        self._transformer = Transformer.from_crs(self.area_to_contain.crs, self.area_to_crop.crs, always_xy=True)

    def get_slices(self):
        """Get the slices to crop *area_to_crop* enclosing *area_to_contain*."""
        poly = self.get_polygon_to_contain()
        return self.get_slices_from_polygon(poly)

    @abstractmethod
    def get_polygon_to_contain(self):
        """Get the shapely Polygon corresponding to *area_to_contain*."""
        raise NotImplementedError

    @abstractmethod
    def get_slices_from_polygon(self, poly):
        """Get the slices based on the polygon."""
        raise NotImplementedError


class SwathSlicer(Slicer):
    """A Slicer for cropping SwathDefinitions."""

    def get_polygon_to_contain(self):
        """Get the shapely Polygon corresponding to *area_to_contain* in lon/lat coordinates."""
        from shapely.geometry import Polygon
        x, y = self.area_to_contain.get_edge_bbox_in_projection_coordinates(10)
        poly = Polygon(zip(*self._transformer.transform(x, y)))
        return poly

    def get_slices_from_polygon(self, poly):
        """Get the slices based on the polygon."""
        intersecting_chunk_slices = []
        for smaller_poly, slices in _get_chunk_polygons_for_swath_to_crop(self.area_to_crop):
            if smaller_poly.intersects(poly):
                intersecting_chunk_slices.append(slices)
        if not intersecting_chunk_slices:
            raise IncompatibleAreas
        return self._assemble_slices(intersecting_chunk_slices)

    @staticmethod
    def _assemble_slices(chunk_slices):
        """Assemble slices to one slice per dimension."""
        lines, cols = zip(*chunk_slices)
        line_slice = slice(min(slc.start for slc in lines), max(slc.stop for slc in lines))
        col_slice = slice(min(slc.start for slc in cols), max(slc.stop for slc in cols))
        slices = col_slice, line_slice
        return slices


@lru_cache(maxsize=10)
def _get_chunk_polygons_for_swath_to_crop(swath_to_crop):
    """Get the polygons for each chunk of the area_to_crop."""
    res = []
    from shapely.geometry import Polygon
    src_chunks = swath_to_crop.lons.chunks
    for _position, (line_slice, col_slice) in _enumerate_chunk_slices(src_chunks):
        line_slice = expand_slice(line_slice)
        col_slice = expand_slice(col_slice)
        smaller_swath = swath_to_crop[line_slice, col_slice]
        lons, lats = smaller_swath.get_edge_lonlats(10)
        lons = np.hstack(lons)
        lats = np.hstack(lats)
        smaller_poly = Polygon(zip(lons, lats))
        res.append((smaller_poly, (line_slice, col_slice)))
    return res


def expand_slice(small_slice):
    """Expand slice by one."""
    return slice(max(small_slice.start - 1, 0), small_slice.stop + 1, small_slice.step)


class AreaSlicer(Slicer):
    """A Slicer for cropping AreaDefinitions."""

    def get_polygon_to_contain(self):
        """Get the shapely Polygon corresponding to *area_to_contain* in projection coordinates of *area_to_crop*."""
        from shapely.geometry import Polygon
        x, y = self.area_to_contain.get_edge_bbox_in_projection_coordinates(frequency=10)
        if self.area_to_crop.is_geostationary:
            x_geos, y_geos = get_geostationary_bounding_box_in_proj_coords(self.area_to_crop, 360)
            x_geos, y_geos = self._transformer.transform(x_geos, y_geos, direction=TransformDirection.INVERSE)
            geos_poly = Polygon(zip(x_geos, y_geos))
            poly = Polygon(zip(x, y))
            poly = poly.intersection(geos_poly)
            if poly.is_empty:
                raise IncompatibleAreas('No slice on area.')
            x, y = zip(*poly.exterior.coords)

        return Polygon(zip(*self._transformer.transform(x, y)))

    def get_slices_from_polygon(self, poly_to_contain):
        """Get the slices based on the polygon."""
        if not poly_to_contain.is_valid:
            raise IncompatibleAreas("Area outside of domain.")
        try:
            # We take a little margin around the polygon to ensure all needed pixels will be included.
            if self.area_to_crop.crs.axis_info[0].unit_name == self.area_to_contain.crs.axis_info[0].unit_name:
                buffer_size = np.max(self.area_to_contain.resolution)
            else:
                buffer_size = 0
            buffered_poly = poly_to_contain.buffer(buffer_size)
            bounds = buffered_poly.bounds
        except ValueError as err:
            raise InvalidArea("Invalid area") from err
        from shapely.geometry import Polygon
        poly_to_crop = Polygon(zip(*self.area_to_crop.get_edge_bbox_in_projection_coordinates(frequency=10)))
        if not poly_to_crop.intersects(buffered_poly):
            raise IncompatibleAreas("Areas not overlapping.")
        bounds = self._sanitize_polygon_bounds(bounds)
        slice_x, slice_y = self._create_slices_from_bounds(bounds)
        return slice_x, slice_y

    def _sanitize_polygon_bounds(self, bounds):
        """Reset the bounds within the shape of the area."""
        try:
            (minx, miny, maxx, maxy) = bounds
        except ValueError as err:
            raise IncompatibleAreas('No slice on area.') from err
        x_bounds, y_bounds = self.area_to_crop.get_array_coordinates_from_projection_coordinates(np.array([minx, maxx]),
                                                                                                 np.array([miny, maxy]))
        y_size, x_size = self.area_to_crop.shape
        if np.all(x_bounds < 0) or np.all(y_bounds < 0) or np.all(x_bounds >= x_size) or np.all(y_bounds >= y_size):
            raise IncompatibleAreas('No slice on area.')
        return x_bounds, y_bounds

    @staticmethod
    def _create_slices_from_bounds(bounds):
        """Create slices from bounds."""
        x_bounds, y_bounds = bounds
        try:
            slice_x = slice(int(np.floor(max(np.min(x_bounds), 0))),
                            int(np.ceil(np.max(x_bounds))))
            slice_y = slice(int(np.floor(max(np.min(y_bounds), 0))),
                            int(np.ceil(np.max(y_bounds))))
        except OverflowError as err:
            raise IncompatibleAreas("Area not within finite bounds.") from err
        return expand_slice(slice_x), expand_slice(slice_y)


def _enumerate_chunk_slices(chunks):
    """Enumerate chunks with slices."""
    for position in np.ndindex(tuple(map(len, (chunks)))):
        slices = []
        for pos, chunk in zip(position, chunks):
            chunk_size = chunk[pos]
            offset = sum(chunk[:pos])
            slices.append(slice(offset, offset + chunk_size))

        yield (position, slices)
