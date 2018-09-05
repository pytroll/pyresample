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
from xarray import DataArray


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
    area_dict = _read_yaml_area_file_content(area_file_name)
    area_list = regions or area_dict.keys()
    res = []
    for area_name in area_list:
        try:
            params = area_dict[area_name]
        except KeyError:
            raise AreaNotFound('Area "{0}" not found in file "{1}"'.format(
                area_name, area_file_name))
        description = params.pop('description', None)
        projection = params.pop('projection', None)
        params['area_id'] = params.get('area_id', area_name)
        params['shape'] = _get_list(params, 'shape', ['height', 'width', 'size'])
        params['top_left_extent'] = _get_list(params, 'top_left_extent', ['x', 'y', 'size'])
        params['center'] = _get_list(params, 'center', ['x', 'y', 'size'])
        params['area_extent'] = _get_list(params, 'area_extent', ['lower_left_xy', 'upper_right_xy', 'extents'])
        params['pixel_size'] = _get_list(params, 'pixel_size', ['x', 'y', 'size'])
        params['radius'] =  _get_list(params, 'radius', ['x', 'y', 'size'])
        res.append(from_params(description, projection, **params))
    return res


def _get_list(params, var, arg_list, default=None):
    """Reads a list-like param variable."""
    # Check if variable is in yaml.
    list_of_values = []
    try:
        variable = params[var]
    except KeyError:
        return default
    # Single number format.
    if isinstance(variable, int) or isinstance(variable, float):
        if var == 'area_extent':
            return variable
        return variable, variable
    # Non-dicts (lists will work).
    if not isinstance(variable, dict):
        return variable
    # Iterate through dict.
    for arg in arg_list:
        try:
            values = variable[arg]
            # Append values of list if the arg can be list-like.
            if arg in ('lower_left_xy', 'upper_right_xy', 'size', 'extents') and isinstance(values, list):
                list_of_values.extend(values)
            else:
                list_of_values.append(values)
                # Single number format.
                if arg in ('lower_left_xy', 'upper_right_xy', 'size'):
                    list_of_values.append(values)
        except (TypeError, AttributeError, ValueError, KeyError):
            pass
    # If units are present, convert to xarray.
    units = variable.get('units')
    if units is not None:
        return DataArray(list_of_values, attrs={'units': units})
    return tuple(list_of_values)


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
    return from_params(config['NAME'], config['PCS_DEF'], area_id=config['REGION'],
                                      proj_id=config['PCS_ID'], shape=(config['YSIZE'], config['XSIZE']),
                                      area_extent=config['AREA_EXTENT'], rotation=config['ROTATION'])


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
    # return AreaDefinition(area_id, area_name, proj_id, proj_dict,
    #                       x_size, y_size, area_extent)
    return from_params(area_name, proj_dict, area_id=area_id, proj_id=proj_id,
                                      shape=(y_size, x_size), area_extent=area_extent)


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


def from_params(name, proj4, shape=None, top_left_extent=None, center=None, area_extent=None, pixel_size=None,
                radius=None, units=None, **kwargs):
    """Takes data the user knows and tries to make an area definition from what can be found."""
    area_id, proj_id = kwargs.pop('area_id', name), kwargs.pop('proj_id', None)

    for key, value in {'name': name, 'area_id': area_id, 'proj_id': proj_id}.items():
        if not isinstance(value, str) and (value is not None or key != 'proj_id'):
            raise ValueError('{0} must be a string. Type entered: {1}'.format(key, type(value)))

    # Get a proj4_dict from either a proj4_dict or a proj4_string.
    proj_dict, p = _get_proj_data(proj4)

    # If no units are provided, try to get units used in proj_dict. If still none are provided, use meters.
    if units is None:
        units = proj_dict.get('units', 'meters')

    # Makes sure list-like objects are list-like, have the right shape, and contain only numbers.
    center, radius, top_left_extent, pixel_size, area_extent, shape =\
        [_verify_list(name, var, length) for name, var, length in
         zip(*[['center', 'radius', 'top_left_extent', 'pixel_size', 'area_extent', 'shape'],
               [center, radius, top_left_extent, pixel_size, area_extent, shape], [2, 2, 2, 2, 4, 2]])]

    # Converts from lat/lon to projection coordinates (x,y) if not in projection coordinates. Returns tuples.
    center, radius, top_left_extent, pixel_size, area_extent = _get_units(center, radius, top_left_extent, pixel_size,
                                                                          area_extent, units, p)

    # Fills in missing information to attempt to create an area definition.
    if None in (area_extent, shape):
        area_extent, shape = _extrapolate_information(area_extent, shape, center, radius, pixel_size, top_left_extent)

    return _make_area(area_id, name, proj_id, proj_dict, shape, area_extent, **kwargs)


def _make_area(area_id, name, proj_id, proj_dict, shape, area_extent, **kwargs):
    """Handles the actual creation of an area definition for from_params."""
    from pyresample.geometry import AreaDefinition
    from pyresample.geometry import DynamicAreaDefinition

    # Used for area definition to prevent indexing None.
    x_size, y_size = None, None
    if shape is not None:
        x_size, y_size = int(round(shape[1])), int(round(shape[0]))
        # Make sure shape is an integer.
        if not np.allclose([y_size, x_size], shape):
            raise ValueError('Shape found or provided must be an integer: {0}'.format(shape))
    # If enough data is provided, create an area_definition. If only shape or area_extent are found, make a
    # DynamicAreaDefinition. If not enough information was provided, raise a ValueError.
    if None not in (area_extent, shape):
        return AreaDefinition(area_id, name, proj_id, proj_dict, x_size, y_size, area_extent, **kwargs)
    elif area_extent is not None or shape is not None:
        return DynamicAreaDefinition(area_id=area_id, description=name, proj_dict=proj_dict, x_size=x_size,
                                     y_size=y_size, area_extent=area_extent, rotation=kwargs.get('rotation'),
                                     optimize_projection=kwargs.get('optimize_projection', False))
    raise ValueError('Not enough information provided to create an area definition')


def _get_proj_data(proj4):
    """Takes a proj4_dict or proj4_string and returns a proj4_dict."""
    from pyproj import Proj

    if isinstance(proj4, str):
        proj_dict = proj4_str_to_dict(proj4)
    elif isinstance(proj4, dict):
        proj_dict = proj4
    else:
        raise ValueError('"proj4" must be a proj4 dict or a proj4 string. Type entered: {0}'.format(proj4.__class__))
    return proj_dict, Proj(proj_dict)


def _get_units(center, radius, top_left_extent, pixel_size, area_extent, units, p):
    """handles area_extent being a set of two points and calls _convert_units."""
    # Splits area_extent into two lists so that its units can be converted
    if area_extent is None:
        area_extent_ll = None
        area_extent_ur = None
    else:
        area_extent_ll = area_extent[:2]
        area_extent_ur = area_extent[2:]

    center, radius, top_left_extent, pixel_size, area_extent_ll, area_extent_ur =\
        [_convert_units(var, name, units, p) for var, name in zip(*[[center, radius, top_left_extent, pixel_size,
                                                                     area_extent_ll, area_extent_ur],
                                                                    ['center', 'radius', 'top_left_extent',
                                                                     'pixel_size', 'area_extent', 'area_extent']])]
    # Recombine area_extent.
    if area_extent is not None:
        area_extent = area_extent_ll + area_extent_ur
    return center, radius, top_left_extent, pixel_size, area_extent


def _validate_variable(var, new_var, var_name, input_list):
    """Makes sure data given does not conflict with itself."""
    if var is not None and not np.allclose(var, new_var):
        raise ValueError('CONFLICTING DATA: {0} given does not match {0} found from {1}'.format(
            var_name, ', '.join(input_list)) + ':\ngiven: {0}\nvs\nfound: {1}'.format(var, new_var, var_name,
                                                                                      input_list))
    return new_var


def _convert_units(var, name, units, p, inverse=False):
    """Converts units from lon/lat to projection coordinates (meters). The inverse does the opposite."""
    if var is None:
        return None
    if hasattr(var, 'units'):
        units = var.units
    if p.is_latlong() and 'm' in units:
        raise ValueError('latlon/latlong projection cannot take meters as units: {0}'.format(name))
    if isinstance(var, DataArray):
        var = tuple(var.data.tolist())
    if not (units and ('\xB0' in units or 'deg' in units or 'rad' in units or 'm' in units)):
        raise ValueError("{0}'s units must be in degrees, radians, or meters. Given units were: {1}".format(name,
                                                                                                            units))
    # Return either degrees or meters depending on if the inverse is true or not.
    # Don't convert if inverse is True: Already in degrees/radians.
    if ('\xB0' in units or 'deg' in units or 'rad' in units) and not inverse:
        # Converts list-like from degrees/radians to meters. Lists must be within
        # [-180, 180] degrees or [-pi, pi] radians.
        var = p(*var, inverse=inverse, radians='rad' in units, errcheck=True)
    # Don't convert if inverse is False: Already in meters.
    elif inverse:
        # Converts list-like from meters to degrees. (list[0]^2 + list[1]^2)^.5 must be within 12742456 meters.
        var = p(*var, inverse=inverse, errcheck=True)
    return var


def _extrapolate_information(area_extent, shape, center, radius, pixel_size, top_left_extent):
    """Attempts to find shape and area_extent based on data provided. Note: order does matter."""
    # Input unaffected by data below: When area extent is calcuated, it's either with
    # shape (giving you an area definition) or with center/radius/top_left_extent (which this produces).
    # Yet output (center/radius/top_left_extent) is essential for data below.
    if area_extent is not None:
        # Function 1-A
        new_radius = [(area_extent[2] - area_extent[0]) / 2, (area_extent[3] - area_extent[1]) / 2]
        radius = _validate_variable(radius, new_radius, 'radius', ['area_extent'])
        new_center = [(area_extent[2] + area_extent[0]) / 2, (area_extent[3] + area_extent[1]) / 2]
        center = _validate_variable(center, new_center, 'center', ['area_extent'])
        new_top_left_extent = [area_extent[0], area_extent[3]]
        top_left_extent = _validate_variable(top_left_extent, new_top_left_extent, 'top_left_extent',
                                                 ['area_extent'])
    # Output used below, but nowhere else is top_left_extent made. Thus it should go as early as possible.
    elif None not in (top_left_extent, center):
        # Function 1-B
        new_radius = [center[0] - top_left_extent[0], top_left_extent[1] - center[1]]
        radius = _validate_variable(radius, new_radius, 'radius', ['top_left_extent', 'center'])
    # Inputs unaffected by data below: area_extent is not an input. However, utput is used below.
    if None not in (radius, pixel_size):
        # Function 2-A
        new_shape = [2 * radius[1] / pixel_size[1], 2 * radius[0] / pixel_size[0]]
        shape = _validate_variable(shape, new_shape, 'shape', ['radius', 'pixel_size'])
    elif None not in (pixel_size, shape):
        # Function 2-B
        new_radius = [pixel_size[0] * shape[1] / 2, pixel_size[1] * shape[0] / 2]
        radius = _validate_variable(radius, new_radius, 'radius', ['shape', 'pixel_size'])

    # Input determined from above functions, but output does not affect above functions: area_extent can be
    # used to find center/top_left_extent which are used to find each other, which is redundant.
    if None not in (center, radius):
        # Function 1-C
        new_area_extent = [center[0] - radius[0], center[1] - radius[1], center[0] + radius[0],
                           center[1] + radius[1]]
        area_extent = _validate_variable(area_extent, new_area_extent, 'area_extent', ['center', 'radius'])
    elif None not in (top_left_extent, radius):
        # Function 1-D
        new_area_extent = (
            top_left_extent[0], top_left_extent[1] - 2 * radius[1], top_left_extent[0] + 2 * radius[0],
            top_left_extent[1])
        area_extent = _validate_variable(area_extent, new_area_extent, 'area_extent',
                                             ['top_left_extent', 'radius'])
    return area_extent, shape


def _verify_list(name, var, length):
    """ Checks that every piece of data that should be list-like (converts lists/tuples to xarrays) is list-like,
        makes sure shapes are accurate, and checks to make sure their values are numbers."""
    # Make list-like data into tuples (or leave as xarrays). If not list-like, throw a ValueError unless it is None.
    if var is None:
        return None
    # Verify that list is made of numbers and list-like.
    try:
        if hasattr(var, 'units') and name != 'shape':
            var = DataArray([float(num) for num in var.data.tolist()], attrs=var.attrs)
        elif isinstance(var, DataArray) and name != 'shape':
            var = tuple(float(num) for num in var.data.tolist())
        else:
            var = tuple(float(num) for num in var)
    except TypeError:
        raise ValueError('{0} is not list-like:\n{1}'.format(name, var))
    except ValueError:
        raise ValueError('{0} is not composed purely of numbers:\n{1}'.format(name, var))
    # Confirm correct shape
    if len(var) != length:
        raise ValueError('{0} should have length {1}, but instead has length {2}:\n{3}'.format(name, length,
                                                                                               len(var), var))
    return var
