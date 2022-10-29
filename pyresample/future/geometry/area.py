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

from typing import Union

from pyproj import CRS

from pyresample.geometry import AreaDefinition as LegacyAreaDefinition  # noqa
from pyresample.geometry import (  # noqa
    DynamicAreaDefinition,
    get_geostationary_angle_extent,
    get_geostationary_bounding_box,
)


class AreaDefinition(LegacyAreaDefinition):
    """Uniformly-spaced grid of pixels on a coordinate referenced system.

    Args:
        area_id
            Identifier for the area
        projection:
            Dictionary of PROJ parameters or string of PROJ or WKT parameters.
            Can also be a :class:`pyproj.crs.CRS` object.
        width:
            x dimension in number of pixels, aka number of grid columns
        height:
            y dimension in number of pixels, aka number of grid rows
        area_extent:
            Area extent as a list (lower_left_x, lower_left_y, upper_right_x, upper_right_y)

    Attributes:
        area_id (str):
            Identifier for the area
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

    """

    def __init__(
            self,
            area_id: str,
            crs: Union[str, int, dict, CRS],
            width: int,
            height: int,
            area_extent: tuple[float, float, float, float],
    ):
        super().__init__(
            area_id,
            "",
            area_id,
            crs,
            width,
            height,
            area_extent,
        )
