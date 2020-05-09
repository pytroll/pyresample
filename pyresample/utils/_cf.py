#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Pyresample developers
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

import pyproj


def _load_crs_from_cf(nc_handle, grid_mapping_varname):
    """ use pyproj to parse the content of the grid_mapping variable and initialize a crs object """
    # here we assume the grid_mapping_varname exists (checked by caller)
    return pyproj.CRS.from_cf(vars(nc_handle[grid_mapping_varname]))


def _is_valid_coordinate_variable(nc_handle, coord_varname, axis, type_of_grid_mapping):
    """ check if a coord_varname is a valid CF coordinate variable """

    valid = False

    if axis not in ('x', 'y'):
        raise ValueError("axis= parameter must be 'x' or 'y'")


    coord_var = nc_handle[coord_varname]
    try:
        if type_of_grid_mapping == 'latitude_longitude':
            # specific name for the latitude_longitude grid mapping
            valid = getattr(coord_var, 'standard_name') == {'x': 'longitude', 'y': 'latitude'}[axis]
        elif type_of_grid_mapping == 'rotated_latitude_longitude':
            # specific name for the rotated_latitude_longitude grid mapping
            valid = getattr(coord_var, 'standard_name') == 'grid_'+{'x': 'longitude', 'y': 'latitude'}[axis]
        elif type_of_grid_mapping == 'geostationary':
            # specific name for the geostationary grid mapping
            # CF-1.9 introduces projection_(x|y)_angular_coordinate for the geostationary projection
            valid_cf19 = getattr(coord_var, 'standard_name') == 'projection_'+axis+'_angular_coordinate'
            valid_oldercf = getattr(coord_var, 'standard_name') == 'projection_'+axis+'_coordinate'
            valid = valid_cf19 + valid_oldercf
        else:
            # default and most common naming: projection_(x|y)_coordinate
            valid = getattr(coord_var, 'standard_name') == 'projection_'+axis+'_coordinate'
    except AttributeError:
        # if the coordinate variable is missing a standard_name, it cannot be a valid CF coordinate axis
        valid = False
    return valid


def _load_cf_axis_info(nc_handle, coord_varname):
    """ load first value, last value, sign, spacing, and length for the axis held in coord_varname """

    # this requires reading the data, we only read first and last
    first = (nc_handle[coord_varname][0]).item()
    last = (nc_handle[coord_varname][-1]).item()
    nb = len(nc_handle[coord_varname])

    # spacing and sign of the axis
    delta = float(last - first) / (nb - 1)
    spacing = abs(delta)
    sign = delta / spacing

    # get the unit information
    try:
        unit = nc_handle[coord_varname].units
    except AttributeError:
        unit = None

    # some units that are valid in CF are not valid to pass to proj
    if unit.startswith('rad') or \
       unit.startswith('deg'):
        unit = None

    # return in a dictionnary structure
    ret = {'first': first, 'last': last, 'spacing': spacing,\
           'nb': nb, 'sign': sign, 'unit': unit}
    print(coord_varname, ret)

    return ret


def _get_area_extent_from_cf_axis(x, y):
    """ combine the 'info' about x and y axis into an extent """

    # find the ll: lower-left and ur: upper-right.
    # x['first'], y['first'] is always the Upper Left corner
    #   (think of numpy's convention for a 2D image with index 0,0 in top left).
    ll_x, ll_y = x['first'], y['last']
    ur_x, ur_y = x['last'], y['first']

    # handle the half-pixel offset between the center of corner cell (what we have in the axis info)
    #   and the corner of corner cell (what AreaDefinition expects)
    ll_x -= x['sign'] * 0.5 * x['spacing']
    ur_x += x['sign'] * 0.5 * x['spacing']
    ll_y += y['sign'] * 0.5 * y['spacing']
    ur_y -= y['sign'] * 0.5 * y['spacing']

    # return as tuple
    ret = (ll_x, ll_y, ur_x, ur_y)
    print('extent:', ret)

    return ret

def _guess_cf_axis_varname(nc_handle, variable, axis, type_of_grid_mapping):
    """ guess the name of the coordinate variable holding the axis. although Y and X are recommended to
          be placed last in the list of coordinates, this is not required. """

    ret = None

    if axis not in ('x', 'y'):
        raise ValueError("axis= parameter must be 'x' or 'y'")

    # the name of y and x are in the dimensions of the variable=
    for dim in nc_handle[variable].dimensions:
        # test if each dim is a valid CF coordinate variable
        if _is_valid_coordinate_variable(nc_handle, dim, axis, type_of_grid_mapping):
            ret = dim
            break


    return ret

def _guess_cf_lonlat_varname(nc_handle, variable, lonlat):
    """ guess the name of the variable holding the latitude (or longitude)
            corresponding to 'variable' """

    ret = None

    if lonlat not in ('lon', 'lat'):
        raise ValueError("lonlat= parameter must be 'lon' or 'lat'")

    # lat/lon are either directly a dimension,...
    search_list = list(nc_handle[variable].dimensions)
    try:
        # ...  or listed in the ':coordinates' attribute.
        search_list += nc_handle[variable].coordinates.split(' ')
    except AttributeError:
        # no ':coordinates' attribute, this is fine
        pass

    # go through the list of variables and check if one of them is lat / lon
    for v in search_list:
        try:
            # this allows for both 'latitude' and 'rotated_latitude'...
            if {'lat':'latitude','lon':'longitude'}[lonlat] in nc_handle[v].standard_name:
                ret = v
                break
        except AttributeError:
            # no 'standard_name'. this is not what we are looking for.
            pass

    return ret


def load_cf_area(nc_file, variable=None, y=None, x=None, with_cf_info=False):
    """ Load an area def object from a netCDF/CF file. """

    from netCDF4 import Dataset
    from pyresample import geometry
    from pyresample.utils import proj4_str_to_dict

    # Prepare cf_info
    #  cf_info holds information about the structure of the cf grid information
    #     (like the name of the coordinate axes, the type of projection, the name of
    #     the lat and lon variables, etc...
    cf_info = {}

    # basic check on the default values of the parameters.
    if (x is not None and y is None) or (x is None and y is not None):
        raise ValueError("You must specify either all or none of x= and y=")

    # the nc_file can be either the path to a netCDF/CF file, or directly an opened netCDF4.Dataset()
    #   if the path to a file, open the Dataset access to it
    if not isinstance(nc_file, Dataset):
        try:
            nc_file = Dataset(nc_file)
        except FileNotFoundError:
            raise FileNotFoundError('File not found: {}'.format(nc_file))
        except OSError:
            raise ValueError('File is not a netCDF file {}'.format(nc_file))

    # if the variable=None, we search for a good variable
    if variable is None:
        variable = 'good variable'
        raise NotImplementedError("search for a good variable is not implemented yet!")
    else:
        # the variable= must exist in the netCDF file
        if variable not in nc_file.variables.keys():
            raise ValueError("Variable '{}' does not exist in netCDF file".format(variable))

    cf_info['variable'] = variable

    # Load a CRS object
    # =================
    grid_mapping_variable = None
    variable_is_itself_gridmapping = False
    # test if the variable has a grid_mapping attribute
    if hasattr(nc_file[variable], 'grid_mapping'):
        # good. attempt to load the grid_mapping information into a pyproj object
        crs = _load_crs_from_cf(nc_file, nc_file[variable].grid_mapping)
        grid_mapping_variable = nc_file[variable].grid_mapping
    else:
        # the variable doesn't have a grid_mapping attribute.
        # ... maybe it is the grid_mapping variable itself?
        try:
            crs = _load_crs_from_cf(nc_file, variable)
            grid_mapping_variable = variable
            variable_is_itself_gridmapping = True
        except pyproj.exceptions.CRSError as ex:
            # ... not a valid grid_mapping either
            # we assume the crs is 'latitude_longitude' with a WGS84 datum.
            # note: there is no default CRS in CF, we choose WGS84
            grid_mapping_variable = "latlon_default"
            crs = pyproj.CRS.from_string('+proj=latlon +datum=WGS84 +ellps=WGS84')

    # the type of grid_mapping (its grid_mapping_name) impacts several aspects of the CF reader
    if grid_mapping_variable == 'latlon_default':
        type_of_grid_mapping = 'latitude_longitude'
    else:
        try:
            type_of_grid_mapping = nc_file[grid_mapping_variable].grid_mapping_name
        except AttributeError:
            raise ValueError(
                "Not a valid CF grid_mapping variable ({}): it lacks a :grid_mapping_name attribute".format(grid_mapping_variable))

    cf_info['grid_mapping_variable'] = grid_mapping_variable
    cf_info['type_of_grid_mapping']  = type_of_grid_mapping

    # identify and load the x/y axis
    # ==============================

    # test if we can allow None for y and x
    if variable_is_itself_gridmapping and (y is None or x is None):
        raise ValueError("When variable= points to the grid_mapping variable itself, y= and x= must be provided")

    # if y= or x= are None, guess the variable names for the axis
    xy = dict()
    if y is None and x is None:
        for axis in ('x', 'y'):
            xy[axis] = _guess_cf_axis_varname(nc_file, variable, axis, type_of_grid_mapping)
            if xy[axis] is None:
                raise ValueError("Could not guess the name of the {} axis of the {}".format(axis, grid_mapping_variable))
    else:
        # y= and x= are provided by the caller. Check they are valid CF coordinate variables
        #   The order is always (y,x)
        xy['y'] = y
        xy['x'] = x
        for axis in ('x', 'y'):
            if not _is_valid_coordinate_variable(nc_file, xy[axis], axis, type_of_grid_mapping):
                raise ValueError(
                    "Variable x='{}' is not a valid CF coordinate variable for the {} axis".format(xy[axis], axis))


    # we now have the names for the x= and y= coordinate variables: load the info of each axis separately
    axis_info = dict()
    for axis in ('x', 'y'):
        axis_info[axis] = _load_cf_axis_info(nc_file, xy[axis],)

    # there are few cases where the x/y values loaded from the CF files cannot be
    #   used directly in pyresample.
    if type_of_grid_mapping == 'geostationary':
        # for geostationary projection, the values stored as x/y are not directly
        #  the x/y along the projection axes, but are rather the scanning angles from
        #  the satellite. We must multiply them by the height of the satellite.
        satellite_height = crs.to_dict()['h']
        for axis in ('x', 'y'):
            for k in ('first','last','spacing'):
                axis_info[axis][k] *= satellite_height
            # the unit is now the default (meters)
            axis_info[axis]['units'] = None

    # transfer information on the axis to the cf_info dict()
    for axis in ('x', 'y'):
        cf_info[axis]  = dict()
        cf_info[axis]['varname'] = xy[axis]
        for k in axis_info[axis].keys():
            cf_info[axis][k] = axis_info[axis][k]

    # sanity check: we cannot have different units for x and y
    unit = axis_info['x']['unit']
    if axis_info['x']['unit'] != axis_info['y']['unit']:
        raise ValueError("Cannot have different units for 'x' ({}) and 'y' ({}) axis.".format(axis_info['x']['unit'],axis_info['y']['unit']))

    # create shape
    shape = (axis_info['y']['nb'], axis_info['x']['nb'])

    # get area extent from the x and y info
    extent = _get_area_extent_from_cf_axis(axis_info['x'], axis_info['y'])

    # transform the crs objecto a proj_dict (might not be needed in future versions of pyresample)
    proj_dict = crs.to_dict()

    # finally prepare the AreaDefinition object
    area_def = geometry.AreaDefinition.from_extent('from_cf', proj_dict, shape, extent,
                                               units=unit, )

    # return
    if with_cf_info:

        # also guess the name of the latitude and longitude variables
        for ll in ('lon', 'lat'):
            cf_info[ll] = _guess_cf_lonlat_varname(nc_file, variable, ll)
            # this can be None, in which case there was no good lat/lon candidate variable
            #   in the file.

        return area_def, cf_info
    else:
        return area_def
