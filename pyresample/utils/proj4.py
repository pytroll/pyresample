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

from collections import OrderedDict

try:
    from pyproj.crs import CRS
except ImportError:
    CRS = None


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
    """Convert PROJ.4 compatible string definition to dict

    EPSG codes should be provided as "EPSG:XXXX" where "XXXX"
    is the EPSG number code. It can also be provided as
    ``"+init=EPSG:XXXX"`` as long as the underlying PROJ library
    supports it (deprecated in PROJ 6.0+).

    Note: Key only parameters will be assigned a value of `True`.
    """
    # convert EPSG codes to equivalent PROJ4 string definition
    if proj4_str.startswith('EPSG:') and CRS is not None:
        crs = CRS(proj4_str)
        if hasattr(crs, 'to_dict'):
            # pyproj 2.2+
            return crs.to_dict()
        proj4_str = crs.to_proj4()
    elif proj4_str.startswith('EPSG:'):
        # legacy +init= PROJ4 string and no pyproj 2.0+ to help convert
        proj4_str = "+init={}".format(proj4_str)

    pairs = (x.split('=', 1) for x in proj4_str.replace('+', '').split(" "))
    return convert_proj_floats(pairs)


def proj4_dict_to_str(proj4_dict, sort=False):
    """Convert a dictionary of PROJ.4 parameters to a valid PROJ.4 string"""
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
    if CRS is not None:
        import math
        crs = CRS(proj4_dict)
        a = crs.ellipsoid.semi_major_metre
        b = crs.ellipsoid.semi_minor_metre
        if not math.isnan(b):
            return a, b
        # older versions of pyproj didn't always have a valid minor radius
        proj4_dict = crs.to_dict()

    if isinstance(proj4_dict, str):
        new_info = proj4_str_to_dict(proj4_dict)
    else:
        new_info = proj4_dict.copy()

    # load information from PROJ.4 about the ellipsis if possible

    from pyproj import Geod

    if 'ellps' in new_info:
        geod = Geod(**new_info)
        new_info['a'] = geod.a
        new_info['b'] = geod.b
    elif 'a' not in new_info or 'b' not in new_info:

        if 'rf' in new_info and 'f' not in new_info:
            new_info['f'] = 1. / float(new_info['rf'])

        if 'a' in new_info and 'f' in new_info:
            new_info['b'] = float(new_info['a']) * (1 - float(new_info['f']))
        elif 'b' in new_info and 'f' in new_info:
            new_info['a'] = float(new_info['b']) / (1 - float(new_info['f']))
        elif 'R' in new_info:
            new_info['a'] = new_info['R']
            new_info['b'] = new_info['R']
        else:
            geod = Geod(**{'ellps': 'WGS84'})
            new_info['a'] = geod.a
            new_info['b'] = geod.b

    return float(new_info['a']), float(new_info['b'])
