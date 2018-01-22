# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2015
#
# Authors:
#    Esben S. Nielsen
#    Thomas Lavergne
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

"""Utility functions for pyresample"""

from __future__ import absolute_import

import os
import numpy as np
import six
import yaml
from configobj import ConfigObj
from collections import Mapping


class AreaNotFound(KeyError):

    """Exception raised when specified are is no found in file"""
    pass


def load_area(area_file_name, *regions):
    """Load area(s) from area file

    Parameters
    -----------
    area_file_name : str
        Path to area definition file
    regions : str argument list
        Regions to parse. If no regions are specified all
        regions in the file are returned

    Returns
    -------
    area_defs : object or list
        If one area name is specified a single AreaDefinition object is returned
        If several area names are specified a list of AreaDefinition objects is returned

    Raises
    ------
    AreaNotFound:
        If a specified area name is not found
    """

    area_list = parse_area_file(area_file_name, *regions)
    if len(area_list) == 1:
        return area_list[0]
    else:
        return area_list


def parse_area_file(area_file_name, *regions):
    """Parse area information from area file

    Parameters
    -----------
    area_file_name : str
        Path to area definition file
    regions : str argument list
        Regions to parse. If no regions are specified all
        regions in the file are returned

    Returns
    -------
    area_defs : list
        List of AreaDefinition objects

    Raises
    ------
    AreaNotFound:
        If a specified area is not found
    """

    try:
        return _parse_yaml_area_file(area_file_name, *regions)
    except yaml.scanner.ScannerError:
        return _parse_legacy_area_file(area_file_name, *regions)


def _read_yaml_area_file_content(area_file_name):
    """Read one or more area files in to a single dict object."""
    if isinstance(area_file_name, (str, six.text_type)):
        area_file_name = [area_file_name]

    area_dict = {}
    for area_file_obj in area_file_name:
        if (isinstance(area_file_obj, (str, six.text_type)) and
                os.path.isfile(area_file_obj)):
            with open(area_file_obj) as area_file_obj:
                tmp_dict = yaml.load(area_file_obj)
        else:
            tmp_dict = yaml.load(area_file_obj)
        area_dict = recursive_dict_update(area_dict, tmp_dict)

    return area_dict


def _parse_yaml_area_file(area_file_name, *regions):
    """Parse area information from a yaml area file.

    Args:
        area_file_name: filename, file-like object, yaml string, or list of
                        these.

    The result of loading multiple area files is the combination of all
    the files, using the first file as the "base", replacing things after
    that.
    """
    from pyresample.geometry import DynamicAreaDefinition
    area_dict = _read_yaml_area_file_content(area_file_name)
    area_list = regions or area_dict.keys()

    res = []

    for area_name in area_list:
        try:
            params = area_dict[area_name]
        except KeyError:
            raise AreaNotFound('Area "{0}" not found in file "{1}"'.format(
                area_name, area_file_name))
        description = params['description']
        projection = params['projection']
        optimize_projection = params.get('optimize_projection', False)
        try:
            xsize = params['shape']['width']
            ysize = params['shape']['height']
        except KeyError:
            xsize, ysize = None, None
        try:
            area_extent = (params['area_extent']['lower_left_xy'] +
                           params['area_extent']['upper_right_xy'])
        except KeyError:
            area_extent = None
        try:
            rotation = params['rotation']
        except KeyError:
            rotation = 0
        area = DynamicAreaDefinition(area_name, description,
                                     projection, xsize, ysize,
                                     area_extent,
                                     optimize_projection,
                                     rotation)
        try:
            area = area.freeze()
        except (TypeError, AttributeError):
            pass

        res.append(area)
    return res


def _read_legacy_area_file_lines(area_file_name):
    if isinstance(area_file_name, (str, six.text_type)):
        area_file_name = [area_file_name]

    for area_file_obj in area_file_name:
        if (isinstance(area_file_obj, (str, six.text_type)) and
           not os.path.isfile(area_file_obj)):
            # file content string
            for line in area_file_obj.splitlines():
                yield line
            continue
        elif isinstance(area_file_obj, (str, six.text_type)):
            # filename
            with open(area_file_obj, 'r') as area_file_obj:

                for line in area_file_obj.readlines():
                    yield line


def _parse_legacy_area_file(area_file_name, *regions):
    """Parse area information from a legacy area file."""

    area_file = _read_legacy_area_file_lines(area_file_name)
    area_list = list(regions)
    if len(area_list) == 0:
        select_all_areas = True
        area_defs = []
    else:
        select_all_areas = False
        area_defs = [None for i in area_list]

    # Extract area from file
    in_area = False
    for line in area_file:
        if not in_area:
            if 'REGION' in line:
                area_id = line.replace('REGION:', ''). \
                    replace('{', '').strip()
                if area_id in area_list or select_all_areas:
                    in_area = True
                    area_content = ''
        elif '};' in line:
            in_area = False
            if select_all_areas:
                area_defs.append(_create_area(area_id, area_content))
            else:
                area_defs[area_list.index(area_id)] = _create_area(area_id,
                                                                   area_content)
        else:
            area_content += line

    # Check if all specified areas were found
    if not select_all_areas:
        for i, area in enumerate(area_defs):
            if area is None:
                raise AreaNotFound('Area "%s" not found in file "%s"' %
                                   (area_list[i], area_file_name))
    return area_defs


def _create_area(area_id, area_content):
    """Parse area configuration"""
    from pyresample.geometry import AreaDefinition

    config_obj = area_content.replace('{', '').replace('};', '')
    config_obj = ConfigObj([line.replace(':', '=', 1)
                            for line in config_obj.splitlines()])
    config = config_obj.dict()
    config['REGION'] = area_id

    try:
        string_types = basestring
    except NameError:
        string_types = str
    if not isinstance(config['NAME'], string_types):
        config['NAME'] = ', '.join(config['NAME'])

    config['XSIZE'] = int(config['XSIZE'])
    config['YSIZE'] = int(config['YSIZE'])
    if 'ROTATION' in config.keys():
        config['ROTATION'] = float(config['ROTATION'])
    else:
        config['ROTATION'] = 0
    config['AREA_EXTENT'][0] = config['AREA_EXTENT'][0].replace('(', '')
    config['AREA_EXTENT'][3] = config['AREA_EXTENT'][3].replace(')', '')

    for i, val in enumerate(config['AREA_EXTENT']):
        config['AREA_EXTENT'][i] = float(val)

    config['PCS_DEF'] = _get_proj4_args(config['PCS_DEF'])
    return AreaDefinition(config['REGION'], config['NAME'],
                          config['PCS_ID'], config['PCS_DEF'],
                          config['XSIZE'], config['YSIZE'],
                          config['AREA_EXTENT'], config['ROTATION'])


def get_area_def(area_id, area_name, proj_id, proj4_args, x_size, y_size,
                 area_extent, rotation=0):
    """Construct AreaDefinition object from arguments

    Parameters
    -----------
    area_id : str
        ID of area
    proj_id : str
        ID of projection
    area_name :str
        Description of area
    proj4_args : list or str
        Proj4 arguments as list of arguments or string
    x_size : int
        Number of pixel in x dimension
    y_size : int
        Number of pixel in y dimension
    rotation: float
        Rotation in degrees (negative is cw)
    area_extent : list
        Area extent as a list of ints (LL_x, LL_y, UR_x, UR_y)

    Returns
    -------
    area_def : object
        AreaDefinition object
    """

    from pyresample.geometry import AreaDefinition
    proj_dict = _get_proj4_args(proj4_args)
    return AreaDefinition(area_id, area_name, proj_id, proj_dict,
                          x_size, y_size, area_extent)


def generate_quick_linesample_arrays(source_area_def, target_area_def,
                                     nprocs=1):
    """Generate linesample arrays for quick grid resampling

    Parameters
    -----------
    source_area_def : object
        Source area definition as geometry definition object
    target_area_def : object
        Target area definition as geometry definition object
    nprocs : int, optional
        Number of processor cores to be used

    Returns
    -------
    (row_indices, col_indices) : tuple of numpy arrays
    """
    from pyresample.grid import get_linesample
    lons, lats = target_area_def.get_lonlats(nprocs)

    source_pixel_y, source_pixel_x = get_linesample(lons, lats,
                                                    source_area_def,
                                                    nprocs=nprocs)

    source_pixel_x = _downcast_index_array(source_pixel_x,
                                           source_area_def.shape[1])
    source_pixel_y = _downcast_index_array(source_pixel_y,
                                           source_area_def.shape[0])

    return source_pixel_y, source_pixel_x


def generate_nearest_neighbour_linesample_arrays(source_area_def,
                                                 target_area_def,
                                                 radius_of_influence,
                                                 nprocs=1):
    """Generate linesample arrays for nearest neighbour grid resampling

    Parameters
    -----------
    source_area_def : object
        Source area definition as geometry definition object
    target_area_def : object
        Target area definition as geometry definition object
    radius_of_influence : float
        Cut off distance in meters
    nprocs : int, optional
        Number of processor cores to be used

    Returns
    -------
    (row_indices, col_indices) : tuple of numpy arrays
    """

    from pyresample.kd_tree import get_neighbour_info
    valid_input_index, valid_output_index, index_array, distance_array = \
        get_neighbour_info(source_area_def,
                           target_area_def,
                           radius_of_influence,
                           neighbours=1,
                           nprocs=nprocs)
    # Enumerate rows and cols
    rows = np.fromfunction(lambda i, j: i, source_area_def.shape,
                           dtype=np.int32).ravel()
    cols = np.fromfunction(lambda i, j: j, source_area_def.shape,
                           dtype=np.int32).ravel()

    # Reduce to match resampling data set
    rows_valid = rows[valid_input_index]
    cols_valid = cols[valid_input_index]

    # Get result using array indexing
    number_of_valid_points = valid_input_index.sum()
    index_mask = (index_array == number_of_valid_points)
    index_array[index_mask] = 0
    row_sample = rows_valid[index_array]
    col_sample = cols_valid[index_array]
    row_sample[index_mask] = -1
    col_sample[index_mask] = -1

    # Reshape to correct shape
    row_indices = row_sample.reshape(target_area_def.shape)
    col_indices = col_sample.reshape(target_area_def.shape)

    row_indices = _downcast_index_array(row_indices,
                                        source_area_def.shape[0])
    col_indices = _downcast_index_array(col_indices,
                                        source_area_def.shape[1])

    return row_indices, col_indices


def fwhm2sigma(fwhm):
    """Calculate sigma for gauss function from FWHM (3 dB level)

    Parameters
    ----------
    fwhm : float
        FWHM of gauss function (3 dB level of beam footprint)

    Returns
    -------
    sigma : float
        sigma for use in resampling gauss function

    """

    return fwhm / (2 * np.sqrt(np.log(2)))


def _get_proj4_args(proj4_args):
    """Create dict from proj4 args
    """

    if isinstance(proj4_args, (str, six.text_type)):
        proj_config = ConfigObj(str(proj4_args).replace('+', '').split())
    else:
        proj_config = ConfigObj(proj4_args)
    return proj_config.dict()


def proj4_str_to_dict(proj4_str):
    """Convert PROJ.4 compatible string definition to dict

    Note: Key only parameters will be assigned a value of `True`.
    """
    pairs = (x.split('=', 1) for x in proj4_str.replace('+', '').split(" "))
    return dict((x[0], (x[1] if len(x) == 2 else True)) for x in pairs)


def proj4_dict_to_str(proj4_dict, sort=False):
    """Convert a dictionary of PROJ.4 parameters to a valid PROJ.4 string"""
    keys = proj4_dict.keys()
    if sort:
        keys = sorted(keys)
    params = []
    for key in keys:
        val = proj4_dict[key]
        key = str(key) if key.startswith('+') else '+' + str(key)
        if str(val) in ['True', 'False']:
            # could be string or boolean object
            val = ''
        if val:
            param = '{}={}'.format(key, val)
        else:
            # example "+no_defs"
            param = key
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


def _downcast_index_array(index_array, size):
    """Try to downcast array to uint16
    """

    if size <= np.iinfo(np.uint16).max:
        mask = (index_array < 0) | (index_array >= size)
        index_array[mask] = size
        index_array = index_array.astype(np.uint16)
    return index_array


def wrap_longitudes(lons):
    """Wrap longitudes to the [-180:+180[ validity range (preserves dtype)

    Parameters
    ----------
    lons : numpy array
        Longitudes in degrees

    Returns
    -------
    lons : numpy array
        Longitudes wrapped into [-180:+180[ validity range

    """
    return (lons + 180) % 360 - 180


def check_and_wrap(lons, lats):
    """Wrap longitude to [-180:+180[ and check latitude for validity.

    Args:
        lons (ndarray): Longitude degrees
        lats (ndarray): Latitude degrees

    Returns:
        lons, lats: Longitude degrees in the range [-180:180[ and the original
                    latitude array

    Raises:
        ValueError: If latitude array is not between -90 and 90

    """
    # check the latitutes
    if lats.min() < -90. or lats.max() > 90.:
        raise ValueError(
            'Some latitudes are outside the [-90.:+90] validity range')

    # check the longitudes
    if lons.min() < -180. or lons.max() >= 180.:
        # wrap longitudes to [-180;+180[
        lons = wrap_longitudes(lons)

    return lons, lats


def recursive_dict_update(d, u):
    """Recursive dictionary update using

    Copied from:

        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
