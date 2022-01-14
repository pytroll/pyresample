#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Pyresample developers
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
"""Classes for geometry operations.

See the related Cartopy pull request that implements this functionality
directly in Cartopy: https://github.com/SciTools/cartopy/pull/1888

"""
from __future__ import annotations

import numpy as np
import pyproj

try:
    from xarray import DataArray
except ImportError:
    DataArray = np.ndarray

import cartopy.crs as ccrs


class Projection(ccrs.Projection):
    """Flexible generic Projection with optionally specified bounds."""

    def __init__(self,
                 crs: pyproj.CRS,
                 bounds: list[float, float, float, float] = None,
                 transform_bounds: bool = False):
        """Initialize CRS instance and compute bounds if possible."""
        # NOTE: Skip the base cartopy Projection class __init__
        super(ccrs.Projection, self).__init__(crs)
        if bounds is None and self.area_of_use is not None:
            bounds = (
                self.area_of_use.west,
                self.area_of_use.east,
                self.area_of_use.south,
                self.area_of_use.north,
            )
            transform_bounds = True

        self.bounds = bounds
        if bounds is not None and transform_bounds:
            # Convert lat/lon bounds to projected bounds.
            # Geographic area of the entire dataset referenced to WGS 84
            # NB. We can't use a polygon transform at this stage because
            # that relies on the existence of the map boundary... the very
            # thing we're trying to work out! ;-)
            x0, x1, y0, y1 = bounds
            lons = np.array([x0, x0, x1, x1])
            lats = np.array([y0, y1, y1, y0])
            points = self.transform_points(self.as_geodetic(), lons, lats)
            x = points[:, 0]
            y = points[:, 1]
            self.bounds = (x.min(), x.max(), y.min(), y.max())
        if self.bounds is not None:
            x0, x1, y0, y1 = self.bounds
            self.threshold = min(abs(x1 - x0), abs(y1 - y0)) / 100.
