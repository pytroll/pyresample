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
"""Definitions for uniform gridded areas."""

from __future__ import annotations

from typing import Optional, Union

from pyproj import CRS

from pyresample.geometry import AreaDefinition as LegacyAreaDefinition  # noqa
from pyresample.geometry import (  # noqa
    DynamicAreaDefinition,
    get_full_geostationary_bounding_box_in_proj_coords,
    get_geostationary_angle_extent,
    get_geostationary_bounding_box_in_lonlats,
    get_geostationary_bounding_box_in_proj_coords,
    ignore_pyproj_proj_warnings,
)


class AreaDefinition(LegacyAreaDefinition):
    """Uniformly-spaced grid of pixels on a coordinate referenced system.

    Args:
        crs:
            Dictionary of PROJ parameters or string of PROJ or WKT parameters.
            Can also be a :class:`pyproj.crs.CRS` object.
        shape:
            Shape of the geographic region. Currently only a 2-element tuple
            is supported. The first element should be the number of elements
            in the Y direction (rows) and the second in the X direction
            (columns). So the final tuple is (rows, columns).
        area_extent:
            Area extent as a list (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        attrs:
            Arbitrary metadata related to the area. Some keys in this
            dictionary have special meaning and may be used for various
            logging or serialization processes. The primary special keys are:

            * name: The identifying name for this area. This is equivalent
                to 'area_id' in the legacy AreaDefinition class. It is used
                in YAML serialization if provided. Defaults to an empty string
                if not provided. Not providing this should not affect normal
                coordinate operations.
            * description: A human-readable description of the area. This is
                equivalent to 'description' in the legacy AreaDefinition class.
                It is used in YAML serialization if provided.

    Attributes:
        area_id (str):
            Identifier for the area. This is a convenience for backwards
            compatibility and accesses the ``.attrs['name']`` metadata.
            This will be set to an empty string if not provided.
        shape:
            Shape of the grid as (rows, columns).
        width (int):
            x dimension in number of pixels, aka number of grid columns
        height (int):
            y dimension in number of pixels, aka number of grid rows
        size (int):
            Number of points in grid
        area_extent_ll (tuple):
            Area extent in lons lats as a tuple (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
        pixel_size_x (float):
            Pixel width in projection units
        pixel_size_y (float):
            Pixel height in projection units
        upper_left_extent (tuple):
            Coordinates (x, y) of upper left corner of upper left pixel in projection units
        pixel_upper_left (tuple):
            Coordinates (x, y) of center of upper left pixel in projection units
        pixel_offset_x (float):
            x offset between projection center and upper left corner of upper
            left pixel in units of pixels.
        pixel_offset_y (float):
            y offset between projection center and upper left corner of upper
            left pixel in units of pixels.
        crs (CRS):
            Coordinate reference system object similar to the PROJ parameters in
            `proj_dict` and `proj_str`. This is the preferred attribute to use
            when working with the `pyproj` library. Note, however, that this
            object is not thread-safe and should not be passed between threads.
        attrs (dict):
            Arbitrary metadata related to the area.

    """

    def __init__(
            self,
            crs: Union[str, int, dict, CRS],
            shape: tuple[int, ...],
            area_extent: tuple[float, float, float, float],
            attrs: Optional[dict] = None
    ):
        # FUTURE: Add __slots__
        # FUTURE: Convert this to new class that uses a legacy area internally
        #         Use this to more easily deprecate usage of old properties
        if len(shape) != 2:
            raise NotImplementedError("Only 2-dimensional areas are supported at this time.")

        attrs = attrs or {}
        area_id = attrs.get("name", "")
        super().__init__(
            area_id,
            "",
            area_id,
            crs,
            shape[1],
            shape[0],
            area_extent,
        )
        self.attrs = attrs

    def to_legacy(self) -> LegacyAreaDefinition:
        """Create a pyresample 1.x AreaDefinition object from this instance."""
        return LegacyAreaDefinition(
            self.attrs.get("name", ""),
            self.attrs.get("description", ""),
            self.attrs.get("proj_id", ""),
            self.crs,
            self.width,
            self.height,
            self.area_extent,
        )
