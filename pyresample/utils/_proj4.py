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
import six


def convert_proj_floats(proj_pairs):
    """Convert PROJ.4 parameters to floats if possible."""
    proj_dict = OrderedDict()
    for x in proj_pairs:
        if len(x) == 1 or x[1] is True:
            proj_dict[x[0]] = True
            continue
        if x[0] == 'EPSG':
            proj_dict[x[0]] = x[1]
            continue

        try:
            proj_dict[x[0]] = float(x[1])
        except ValueError:
            proj_dict[x[0]] = x[1]

    return proj_dict


def proj4_str_to_dict(proj4_str):
    """Convert PROJ.4 compatible string definition to dict

    Note: Key only parameters will be assigned a value of `True`.
    """
    if proj4_str.startswith('EPSG:'):
        try:
            code = int(proj4_str.split(':', 1)[1])
        except ValueError as err:
            six.raise_from(ValueError("Invalid EPSG code '{}': {}".format(proj4_str, err)),
                           None)  # Suppresses original exception context in python 3
        return OrderedDict(EPSG=code)
    pairs = (x.split('=', 1) for x in proj4_str.replace('+', '').split(" "))
    return convert_proj_floats(pairs)


def proj4_dict_to_str(proj4_dict, sort=False):
    """Convert a dictionary of PROJ.4 parameters to a valid PROJ.4 string"""
    items = proj4_dict.items()
    if sort:
        items = sorted(items)
    params = []
    for key, val in items:
        if key == 'EPSG':
            # If EPSG code is present, ignore other parameters
            params = ['EPSG:{}'.format(val)]
            break

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
        else:
            geod = Geod(**{'ellps': 'WGS84'})
            new_info['a'] = geod.a
            new_info['b'] = geod.b

    return float(new_info['a']), float(new_info['b'])
