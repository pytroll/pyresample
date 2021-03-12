#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Pyresample developers
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

import math
from collections import OrderedDict

from pyproj import CRS


def convert_proj_floats(proj_pairs):
    """Convert PROJ.4 parameters to floats if possible."""
    proj_dict = OrderedDict()
    for x in proj_pairs:
        if len(x) == 1 or x[1] is True:
            proj_dict[x[0]] = True
            continue

        try:
            proj_dict[x[0]] = float(x[1])
        except ValueError:
            proj_dict[x[0]] = x[1]

    return proj_dict


def proj4_str_to_dict(proj4_str):
    """Convert PROJ.4 compatible string definition to dict.

    EPSG codes should be provided as "EPSG:XXXX" where "XXXX"
    is the EPSG number code. It can also be provided as
    ``"+init=EPSG:XXXX"`` as long as the underlying PROJ library
    supports it (deprecated in PROJ 6.0+).

    Note: Key only parameters will be assigned a value of `True`.
    """
    # convert EPSG codes to equivalent PROJ4 string definition
    crs = CRS(proj4_str)
    return crs.to_dict()


def proj4_dict_to_str(proj4_dict, sort=False):
    """Convert a dictionary of PROJ.4 parameters to a valid PROJ.4 string."""
    items = proj4_dict.items()
    if sort:
        items = sorted(items)
    params = []
    for key, val in items:
        key = str(key) if key.startswith('+') else '+' + str(key)
        if key in ['+no_defs', '+no_off', '+no_rot']:
            param = key
        else:
            param = '{}={}'.format(key, val)
        params.append(param)
    return ' '.join(params)


def proj4_radius_parameters(proj4_dict):
    """Calculate 'a' and 'b' radius parameters.

    Arguments:
        proj4_dict (str or dict): PROJ.4 parameters

    Returns:
        a (float), b (float): equatorial and polar radius
    """
    crs = proj4_dict
    if not isinstance(crs, CRS):
        crs = CRS(crs)
    a = crs.ellipsoid.semi_major_metre
    b = crs.ellipsoid.semi_minor_metre
    if not math.isnan(b):
        return a, b


def get_geostationary_height(geos_area_crs):
    params = geos_area_crs.coordinate_operation.params
    h_param = [p for p in params if 'satellite height' in p.name.lower()][0]
    return h_param.value
