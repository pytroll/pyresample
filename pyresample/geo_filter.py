#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011-2021 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Filters based on geolocation validity."""

import numpy as np
from pyproj import Proj

from . import _spatial_mp, geometry


class GridFilter(object):
    """Geographic filter from a grid.

    Args:
        grid_ll_x (float):
            Projection x coordinate of lower left corner of lower left pixel
        grid_ll_y (float):
            Projection y coordinate of lower left corner of lower left pixel
        grid_ur_x (float):
            Projection x coordinate of upper right corner of upper right pixel
        grid_ur_y (float):
            Projection y coordinate of upper right corner of upper right pixel
        proj4_string (str):
            Projection definition as a PROJ.4 string.
        mask (numpy array):
            Mask as boolean numpy array

    """

    def __init__(self, area_def, filter, nprocs=1):
        self.area_def = area_def
        self._filter = filter.astype(bool)
        self.nprocs = nprocs

    def get_valid_index(self, geometry_def):
        """Calculate valid_index array based on lons and lats.

        Args:
            geometry_def: Geometry definition (ex. SwathDefinition)

        Returns:
            Boolean numpy array of same shape as lons and lats
        """
        lons = geometry_def.lons[:]
        lats = geometry_def.lats[:]

        # Get projection coords
        proj_kwargs = {}
        if self.nprocs > 1:
            proj = _spatial_mp.Proj_MP(self.area_def.crs)
            proj_kwargs["nprocs"] = self.nprocs
        else:
            proj = Proj(self.area_def.crs)

        x_coord, y_coord = proj(lons, lats, **proj_kwargs)

        # Find array indices of coordinates
        target_x = ((x_coord / self.area_def.pixel_size_x) +
                    self.area_def.pixel_offset_x).astype(np.int32)
        target_y = (self.area_def.pixel_offset_y -
                    (y_coord / self.area_def.pixel_size_y)).astype(np.int32)

        # Create mask for pixels outside array (invalid pixels)
        target_x_valid = (target_x >= 0) & (target_x < self.area_def.width)
        target_y_valid = (target_y >= 0) & (target_y < self.area_def.height)

        # Set index of invalid pixels to 0
        target_x[np.invert(target_x_valid)] = 0
        target_y[np.invert(target_y_valid)] = 0

        # Find mask
        filter = self._filter[target_y, target_x]

        # Remove invalid pixels
        filter = (filter & target_x_valid & target_y_valid).astype(bool)

        return filter

    def filter(self, geometry_def, data):
        """Get coordinate definition and data where invalid lon/lats are removed."""
        lons = geometry_def.lons[:]
        lats = geometry_def.lats[:]
        valid_index = self.get_valid_index(geometry_def)
        lons_f = lons[valid_index]
        lats_f = lats[valid_index]
        data_f = data[valid_index]
        geometry_def_f = \
            geometry.CoordinateDefinition(lons_f, lats_f,
                                          nprocs=geometry_def.nprocs)
        return geometry_def_f, data_f
