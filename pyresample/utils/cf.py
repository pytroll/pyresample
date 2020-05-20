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
"""Load an AreaDefinition object from a netCDF/CF file."""

import pyproj

# list of valid CF grid mappings:
_valid_cf_type_of_grid_mapping = \
    ('albers_conical_equal_area',
     'azimuthal_equidistant',
     'geostationary',
     'lambert_azimuthal_equal_area',
     'lambert_conformal_conic',
     'lambert_cylindrical_equal_area',
     'latitude_longitude',
     'mercator',
     'oblique_mercator',
     'orthographic',
     'polar_stereographic',
     'rotated_latitude_longitude',
     'sinusoidal',
     'stereographic',
     'transverse_mercator',
     'vertical_perspective')

# dictionnary with the standard_names accepted by CF per projection type
#   this can be used for reading from and writing to CF files
_valid_cf_coordinate_standardnames = {}
# specific name for most grid mappings
_valid_cf_coordinate_standardnames['default'] = dict()
_valid_cf_coordinate_standardnames['default']['x'] = ('projection_x_coordinate',)
_valid_cf_coordinate_standardnames['default']['y'] = ('projection_y_coordinate',)
# specific name for the latitude_longitude grid mapping
_valid_cf_coordinate_standardnames['latitude_longitude'] = dict()
_valid_cf_coordinate_standardnames['latitude_longitude']['x'] = ('longitude',)
_valid_cf_coordinate_standardnames['latitude_longitude']['y'] = ('latitude',)
# specific name for the rotated_latitude_longitude grid mapping
_valid_cf_coordinate_standardnames['rotated_latitude_longitude'] = dict()
_valid_cf_coordinate_standardnames['rotated_latitude_longitude']['x'] = ('grid_longitude',)
_valid_cf_coordinate_standardnames['rotated_latitude_longitude']['y'] = ('grid_latitude',)
# specific name for the geostationary grid mapping (we support two flavors)
_valid_cf_coordinate_standardnames['geostationary'] = dict()
_valid_cf_coordinate_standardnames['geostationary']['x'] = (
    'projection_x_angular_coordinate', 'projection_x_coordinate',)
_valid_cf_coordinate_standardnames['geostationary']['y'] = (
    'projection_y_angular_coordinate', 'projection_y_coordinate',)


def _convert_XY_CF_to_Proj(crs, axis_info):
    """Convert XY values from CF to PROJ convention. With CF =< 1.9 only affects geostrationary projection."""
    crs_dict = crs.to_dict()
    if crs_dict['proj'] == 'geos':
        # for geostationary projection, the values stored as x/y in CF are not directly
        #  the x/y along the projection axes, but are rather the scanning angles from
        #  the satellite. We must multiply them by the height of the satellite.
        satellite_height = crs_dict['h']
        for k in ('first', 'last', 'spacing'):
            axis_info[k] *= satellite_height
        # the unit is now the default (meters)
        axis_info['units'] = None

    return axis_info


def _load_crs_from_cf_gridmapping(nc_handle, grid_mapping_varname):
    """Initialize a CRS object from a CF grid_mapping variable."""
    # check the variable exists
    try:
        v = nc_handle[grid_mapping_varname]
    except KeyError:
        raise KeyError("Variable '{}' does not exist in netCDF file".format(grid_mapping_varname))

    # check this indeed is a supported grid mapping variable
    try:
        if v.grid_mapping_name not in _valid_cf_type_of_grid_mapping:
            raise ValueError("Not a valid CF grid_mapping variable ({})".format(grid_mapping_varname))
    except AttributeError:
        # no :grid_mapping_name thus it cannot be a valid grid_mapping variable
        raise ValueError("Not a valid CF grid_mapping variable ({})".format(grid_mapping_varname))

    # use pyproj to load the CRS
    return pyproj.CRS.from_cf(v.attrs)


def _is_valid_coordinate_standardname(coord_standard_name, axis, type_of_grid_mapping):
    """Check that a CF coordinate variable has the expected CF standard_name with regard to the typw of grid mapping."""
    valid = False

    if axis not in ('x', 'y'):
        raise ValueError("axis= parameter must be 'x' or 'y'")

    if type_of_grid_mapping != 'default' and type_of_grid_mapping not in _valid_cf_type_of_grid_mapping:
        raise ValueError("grid_mapping_name {} is not a valid CF one".format(type_of_grid_mapping))

    # access the valid standard_names (also handle the 'default')
    try:
        valid_coord_standard_names = _valid_cf_coordinate_standardnames[type_of_grid_mapping][axis]
    except KeyError:
        valid_coord_standard_names = _valid_cf_coordinate_standardnames['default'][axis]

    # test for validity
    valid = coord_standard_name in valid_coord_standard_names

    return valid


def _is_valid_coordinate_variable(nc_handle, coord_varname, axis, type_of_grid_mapping):
    """Check if a variable is a valid CF coordinate variable."""
    valid = False

    if axis not in ('x', 'y'):
        raise ValueError("axis= parameter must be 'x' or 'y'")

    try:
        coord_var = nc_handle[coord_varname]
    except KeyError:
        raise KeyError("Variable '{}' does not exist in netCDF file".format(coord_varname))

    try:
        coord_standard_name = getattr(coord_var, 'standard_name')
        valid = _is_valid_coordinate_standardname(coord_standard_name, axis, type_of_grid_mapping)
    except AttributeError:
        # if the coordinate variable is missing a standard_name, it cannot be a valid CF coordinate axis
        valid = False

    return valid


def _load_cf_axis_info(nc_handle, coord_varname):
    """Load and compute information for a coordinate axis (e.g. first & last values, spacing, length, etc...)."""
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
        unit = getattr(nc_handle[coord_varname], 'units')
    except AttributeError:
        unit = None

    # some units that are valid in CF are not valid to pass to proj
    if unit.startswith('rad') or \
       unit.startswith('deg'):
        unit = None

    # return in a dictionnary structure
    ret = {'first': first, 'last': last, 'spacing': spacing,
           'nb': nb, 'sign': sign, 'unit': unit}

    return ret


def _get_area_extent_from_cf_axis(x, y):
    """Compute the area_extent of the AreaDefinition object from the information on the x and y axes."""
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

    return ret


def _guess_cf_axis_varname(nc_handle, variable, axis, type_of_grid_mapping):
    """Guess the name of the netCDF variable holding the coordinate axis of a netCDF field."""
    ret = None

    if axis not in ('x', 'y'):
        raise ValueError("axis= parameter must be 'x' or 'y'")

    # the name of y and x are in the dimensions of the variable=
    try:
        dims = nc_handle[variable].dims
    except KeyError:
        raise KeyError("variable {} not found in file".format(variable))

    for dim in dims:
        # test if each dim is a valid CF coordinate variable
        if _is_valid_coordinate_variable(nc_handle, dim, axis, type_of_grid_mapping):
            ret = dim
            break

    return ret


def _guess_cf_lonlat_varname(nc_handle, variable, lonlat):
    """Guess the name of the netCDF variable holding the longitude (or latitude) of a netCDF field."""
    ret = None

    if lonlat not in ('lon', 'lat'):
        raise ValueError("lonlat= parameter must be 'lon' or 'lat'")

    # lat/lon are either directly a dimension, or a :coordinates.

    # By default (decode_cf=True) xarray puts all dims and :coordinates in .coords
    #   and remove the :coordinates attribute
    try:
        search_list = list(nc_handle[variable].coords)
    except KeyError:
        raise KeyError("variable {} not found in file".format(variable))

    # if decode_cf=False was used, the look at the :coordinates attribute
    if 'coordinates' in nc_handle[variable].attrs.keys():
        search_list += (nc_handle[variable].attrs['coordinates']).split()

    # go through the list of variables and check if one of them is lat / lon
    for v in search_list:
        try:
            # this allows for both 'latitude' and 'rotated_latitude'...
            if {'lat': 'latitude', 'lon': 'longitude'}[lonlat] in getattr(nc_handle[v], 'standard_name'):
                ret = v
                break
        except AttributeError:
            # no 'standard_name'. this is not what we are looking for.
            pass

    return ret


def _load_cf_area_one_variable_crs(nc_handle, variable):
    """Load the CRS corresponding to variable."""
    grid_mapping_variable = None
    variable_is_itself_gridmapping = False
    # test if the variable has a grid_mapping attribute
    if hasattr(nc_handle[variable], 'grid_mapping'):
        # good. attempt to load the grid_mapping information into a pyproj object
        crs = _load_crs_from_cf_gridmapping(nc_handle, nc_handle[variable].grid_mapping)
        grid_mapping_variable = nc_handle[variable].grid_mapping
    elif hasattr(nc_handle[variable], 'grid_mapping_name') and \
            nc_handle[variable].grid_mapping_name in _valid_cf_type_of_grid_mapping:
        # this looks like a valid grid_mapping variable
        try:
            # try to load it
            crs = _load_crs_from_cf_gridmapping(nc_handle, variable)
            grid_mapping_variable = variable
            variable_is_itself_gridmapping = True
        except pyproj.exceptions.CRSError as ex:
            raise ValueError("ERROR: pyproj didn't manage to load the CRS: {}".format(ex))
    else:
        # fallback position: maybe the variable is on a basic lat/lon grid with no
        #   grid_mapping. Note: there is no default CRS in CF, we choose WGS84
        grid_mapping_variable = "latlon_default"
        crs = pyproj.CRS.from_string('+proj=latlon +datum=WGS84 +ellps=WGS84')

    # return
    return crs, grid_mapping_variable, variable_is_itself_gridmapping


def _load_cf_area_one_variable_axis(nc_handle, variable, type_of_grid_mapping, y=None, x=None):
    """Identidy and load axis x and y."""
    # if y= or x= are None, guess the variable names for the axis
    xy = dict()
    if y is None and x is None:
        for axis in ('x', 'y'):
            xy[axis] = _guess_cf_axis_varname(nc_handle, variable, axis, type_of_grid_mapping)
            if xy[axis] is None:
                raise ValueError("Could not guess the name of the '{}' axis for {}".format(
                    axis, variable))
    else:
        # y= and x= are provided by the caller. Check they are valid CF coordinate variables
        #   The order is always (y,x)
        xy['y'] = y
        xy['x'] = x
        for axis in ('x', 'y'):
            _valid_axis = _is_valid_coordinate_variable(nc_handle, xy[axis], axis, type_of_grid_mapping)
            if not _valid_axis:
                ve = "Variable x='{}' is not a valid CF coordinate variable for the {} axis".format(xy[axis], axis)
                raise ValueError(ve)

    # we now have the names for the x= and y= coordinate variables: load the info of each axis separately
    axis_info = dict()
    for axis in ('x', 'y'):
        axis_info[axis] = _load_cf_axis_info(nc_handle, xy[axis],)

    return xy, axis_info


def _load_cf_area_one_variable_areadef(axis_info, crs, unit, grid_mapping_variable):
    """Prepare the AreaDefinition object."""
    from pyresample import geometry
    # create shape
    shape = (axis_info['y']['nb'], axis_info['x']['nb'])

    # get area extent from the x and y info
    extent = _get_area_extent_from_cf_axis(axis_info['x'], axis_info['y'])

    # transform the crs objecto a proj_dict (might not be needed in future versions of pyresample)
    proj_dict = crs.to_dict()

    # finally prepare the AreaDefinition object
    return geometry.AreaDefinition.from_extent(grid_mapping_variable, proj_dict, shape, extent, units=unit)


def _load_cf_area_one_variable(nc_handle, variable, y=None, x=None):
    """Load the AreaDefinition corresponding to one netCDF variable/field."""
    if variable not in nc_handle.variables.keys():
        raise KeyError("Variable '{}' does not exist in netCDF file".format(variable))

    # the routine always prepares a cf_info
    cf_info = dict()
    cf_info['variable'] = variable

    # Load a CRS object
    # =================
    crs, grid_mapping_variable, variable_is_itself_gridmapping = _load_cf_area_one_variable_crs(nc_handle, variable)

    # the type of grid_mapping (its grid_mapping_name) impacts several aspects of the CF reader
    if grid_mapping_variable == 'latlon_default':
        type_of_grid_mapping = 'latitude_longitude'
    else:
        try:
            type_of_grid_mapping = nc_handle[grid_mapping_variable].grid_mapping_name
        except AttributeError:
            raise ValueError(
                ("Not a valid CF grid_mapping variable ({}):"
                 "it lacks a :grid_mapping_name attribute").format(grid_mapping_variable))

    cf_info['grid_mapping_variable'] = grid_mapping_variable
    cf_info['type_of_grid_mapping'] = type_of_grid_mapping

    # test if we can allow None for y and x
    if variable_is_itself_gridmapping and (y is None or x is None):
        raise ValueError("When variable= points to the grid_mapping variable itself, y= and x= must be provided")

    # identify and load the x/y axis
    # ==============================
    xy, axis_info = _load_cf_area_one_variable_axis(nc_handle, variable, type_of_grid_mapping, y=y, x=x)

    # there are few cases where the x/y values loaded from the CF files cannot be
    #   used directly in pyresample. We need a conversion:
    for axis in ('x', 'y'):
        axis_info[axis] = _convert_XY_CF_to_Proj(crs, axis_info[axis])

    # transfer information on the axis to the cf_info dict()
    for axis in ('x', 'y'):
        cf_info[axis] = dict()
        cf_info[axis]['varname'] = xy[axis]
        for k in axis_info[axis].keys():
            cf_info[axis][k] = axis_info[axis][k]

    # sanity check: we cannot have different units for x and y
    unit = axis_info['x']['unit']
    if axis_info['x']['unit'] != axis_info['y']['unit']:
        raise ValueError("Cannot have different units for 'x' ({}) and 'y' ({}) axis.".format(
            axis_info['x']['unit'], axis_info['y']['unit']))

    # prepare the AreaDefinition object
    # =================================
    area_def = _load_cf_area_one_variable_areadef(axis_info, crs, unit, grid_mapping_variable)

    return area_def, cf_info


def _load_cf_area_several_variables(nc_handle):
    """Load the AreaDefinition corresponding to several netCDF variables/fields."""
    def _indices_unique_AreaDefs(adefs):
        """Find the indices of unique AreaDefinitions in a list."""
        uniqs = dict()
        for i, adef in enumerate(adefs):
            if adef not in uniqs:  # this uses AreaDefinition.__eq__()
                uniqs[adef] = i

        # return only the indices
        return uniqs.values()

    adefs = []
    infos = []

    # go through all the variables
    for v in nc_handle.variables.keys():

        # skip variables that are less than 2D: they cannot
        #   possibly sustain an AreaDefinition
        if nc_handle[v].ndim < 2:
            continue

        try:
            # try and load an AreaDefinition from this variable
            adef, info = _load_cf_area_one_variable(nc_handle, v)
            # store
            adefs.append(adef)
            infos.append(info)
            # break the loop, we have all we need
            break
        except ValueError:
            # this is not a problem: variable v simply doesn't define an AreaDefinition
            continue

    # go through the loaded AreaDefinitions and find the unique ones.
    indices = _indices_unique_AreaDefs(adefs)
    uniq_adefs = [adefs[ui] for ui in indices]
    uniq_infos = [infos[ui] for ui in indices]

    return uniq_adefs, uniq_infos


def load_cf_area(nc_file, variable=None, y=None, x=None):
    """Load an AreaDefinition object from a netCDF/CF file.

    Parameters
    ----------
    nc_file : string or object
        path to a netCDF/CF file, or opened xarray.Dataset object
    variable : string, optional
        name of the variable to load the AreaDefinition from.
        If the variable is not a CF grid_mapping container variable,
        it should be a variable having a :grid_mapping attribute.
        If variable is None the file will be searched for valid CF
        area definitions
    y : string, optional
        name of the variable to use as 'y' axis of the CF area definition
        If y is None an appropriate 'y' axis will be deduced from the CF file
    x : string, optional
        name of the variable to use as 'x' axis of the CF area definition
        If x is None an appropriate 'x' axis will be deduced from the CF file

    Returns
    -------
    are_def, cf_info : geometry.AreaDefinition object, dict
       cf_info holds info about how the AreaDefinition was defined in the CF file.

    """
    import xarray as xr
    # basic check on the default values of the parameters.
    if (x is not None and y is None) or (x is None and y is not None):
        raise ValueError("You must specify either both or none of x= and y=")

    # the nc_file can be either the path to a netCDF/CF file, or directly an opened xarray.Dataset()
    if isinstance(nc_file, xr.Dataset):
        nc_handle = nc_file
    else:
        #   if the path to a file, open the Dataset access to it
        try:
            nc_handle = xr.open_dataset(nc_file)
        except FileNotFoundError as ex:
            raise FileNotFoundError("This file does not exist ({})".format(ex))
        except (OSError, TypeError) as ex:
            raise OSError("This file is probably not a valid netCDF file ({}).".format(ex))

    if variable is None:
        # if the variable=None, we search through all variables
        area_def, cf_info = _load_cf_area_several_variables(nc_handle)
        if len(area_def) == 0:
            raise ValueError("Found no AreaDefinitions in this netCDF/CF file.")
        elif len(area_def) > 1:
            # there were several area_definitions defined in this file. For now bark.
            raise ValueError("The CF file holds several different AreaDefinitions. Use the variable= keyword.")
        else:
            area_def = area_def[0]
            cf_info = cf_info[0]
    else:
        # the variable= is known, call appropriate routine
        try:
            area_def, cf_info = _load_cf_area_one_variable(nc_handle, variable, y=y, x=x, )
        except ValueError as ve:
            raise ValueError("Found no AreaDefinition associated with variable {} ({})".format(variable, ve))

    # also guess the name of the latitude and longitude variables
    for ll in ('lon', 'lat'):
        cf_info[ll] = _guess_cf_lonlat_varname(nc_handle, cf_info['variable'], ll)
        # this can be None, in which case there was no good lat/lon candidate variable
        #   in the file.

    return area_def, cf_info
