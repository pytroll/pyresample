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
"""Helper functions related to transforming coordinates."""
import numpy as np


def lonlat2xyz(lons, lats):
    """Convert lon/lat degrees to geocentric x/y/z coordinates."""
    R = 6370997.0
    lats = np.deg2rad(lats)
    r_cos_lats = R * np.cos(lats)
    lons = np.deg2rad(lons)
    x_coords = r_cos_lats * np.cos(lons)
    y_coords = r_cos_lats * np.sin(lons)
    z_coords = R * np.sin(lats)

    return np.stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), axis=-1)
