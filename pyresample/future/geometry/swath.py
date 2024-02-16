#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
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
"""Definitions for swath-based or non-uniform geometries."""

from __future__ import annotations

import numpy as np

from pyresample.geometry import _get_highest_level_class  # noqa
from pyresample.geometry import CoordinateDefinition, DimensionError  # noqa
from pyresample.geometry import SwathDefinition as LegacySwathDefinition


class SwathDefinition(LegacySwathDefinition):
    """Swath defined by lons and lats.

    Parameters
    ----------
    lons : numpy array
    lats : numpy array
    crs: pyproj.CRS,
       The CRS to use. longlat on WGS84 by default.
    attrs: dict,
       A dictionary made to store metadata.

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

    def __init__(self, lons, lats, crs=None, attrs=None):
        super().__init__(lons, lats, crs=crs)
        self.attrs = attrs or {}

    def __getitem__(self, key):
        """Slice a 2D geographic definition."""
        y_slice, x_slice = key
        return self.__class__(
            lons=self.lons[y_slice, x_slice],
            lats=self.lats[y_slice, x_slice],
            attrs=self.attrs
        )

    def concatenate(self, other):
        """Concatenate coordinate definitions."""
        if self.ndim != other.ndim:
            raise DimensionError(('Unable to concatenate %sD and %sD '
                                  'geometries') % (self.ndim, other.ndim))
        if self.crs != other.crs:
            raise ValueError("Incompatible CRSs.")
        klass = _get_highest_level_class(self, other)
        lons = np.concatenate((self.lons, other.lons))
        lats = np.concatenate((self.lats, other.lats))
        attrs = self.attrs.copy()
        attrs.update(other.attrs)
        return klass(lons, lats, attrs=attrs)
