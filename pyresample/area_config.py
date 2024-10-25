#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019-2021 Pyresample developers
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
"""Area config handling and creation utilities."""
from __future__ import annotations

import io
import logging
import math
import os
import pathlib
import warnings
from typing import Any, Iterable, List, Union

import numpy as np
import yaml
from pyproj import Proj, Transformer
from pyproj.crs import CRS, CRSError

import pyresample
from pyresample._formatting_html import area_repr
from pyresample.utils import proj4_str_to_dict

try:
    from xarray import DataArray
except ImportError:
    class DataArray:  # type: ignore
        """Stand-in for DataArray for holding units information."""

        def __init__(self, data, attrs=None):
            """Initialize 'attrs' and 'data' properties."""
            self.attrs = attrs or {}
            self.data = np.array(data)

        def __getitem__(self, item):
            """Get a subset of the data contained in a DataArray."""
            return DataArray(self.data[item], attrs=self.attrs)

        def __getattr__(self, item):
            """Get metadata property from 'attrs'."""
            return self.attrs[item]

        def __len__(self):
            """Get size of the data."""
            return len(self.data)


class AreaNotFound(KeyError):
    """Exception raised when specified are is no found in file."""


def load_area(area_file_name, *regions):
    """Load area(s) from area file.

    Parameters
    ----------
    area_file_name : str, pathlib.Path, stream, or list thereof
        List of paths or streams.  Any str or pathlib.Path will be
        interpreted as a path to a file.  Any stream will be interpreted
        as containing a yaml definition file.  To read directly from a string,
        use :func:`load_area_from_string`.
    regions : str argument list
        Regions to parse. If no regions are specified all
        regions in the file are returned

    Returns
    -------
    area_defs : pyresample.geometry.AreaDefinition or list
        If one area name is specified a single AreaDefinition object is returned.
        If several area names are specified a list of AreaDefinition objects is returned

    Raises
    ------
    AreaNotFound:
        If a specified area name is not found
    """
    area_list = parse_area_file(area_file_name, *regions)
    if len(area_list) == 1:
        return area_list[0]
    return area_list


def load_area_from_string(area_strs, *regions):
    """Load area(s) from area strings.

    Like :func:`~pyresample.area_config.load_area`, but load from string
    directly.

    For the opposite (i.e. to create a YAML string from an area), use
    :meth:`~pyresample.geometry.AreaDefinition.dump`.

    Parameters
    ----------
    area_strs : str or List[str]
        Strings containing yaml definitions.
    regions : str
        Regions to parse.

    Returns
    -------
    area_defs : pyresample.geometry.AreaDefinition or list
        If one area name is specified a single AreaDefinition object is returned.
        If several area names are specified a list of AreaDefinition objects is returned
    """
    if isinstance(area_strs, str):
        area_strs = [area_strs]
    return load_area([io.StringIO(area_str) for area_str in area_strs],
                     *regions)


def parse_area_file(area_file_name, *regions):
    """Parse area information from area file.

    Parameters
    -----------
    area_file_name : str or list
        One or more paths to area definition files
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
    except (yaml.scanner.ScannerError, yaml.parser.ParserError):
        return _parse_legacy_area_file(area_file_name, *regions)


def _read_yaml_area_file_content(area_file_name):
    """Read one or more area files in to a single dict object."""
    from pyresample.utils import recursive_dict_update

    if isinstance(area_file_name, (str, pathlib.Path, io.IOBase)):
        area_file_name = [area_file_name]

    area_dict = {}
    for area_file_obj in area_file_name:
        if isinstance(area_file_obj, io.IOBase):
            # already a stream
            tmp_dict = yaml.safe_load(area_file_obj)
        else:
            # hopefully a path to a file, but in the past a yaml string could
            # be passed directly, assume any string with a newline must be
            # a yaml file and not a path
            if isinstance(area_file_obj, str) and "\n" in area_file_obj:
                warnings.warn("It looks like you passed a YAML string "
                              "directly.  This is deprecated since pyresample "
                              "1.14.1, please use load_area_from_string or "
                              "pass a stream or a path to a file instead",
                              DeprecationWarning, stacklevel=3)
                tmp_dict = yaml.safe_load(area_file_obj)
            else:
                with open(area_file_obj) as area_file_obj:
                    tmp_dict = yaml.safe_load(area_file_obj)
        area_dict = recursive_dict_update(area_dict, tmp_dict)

    return area_dict


def _parse_yaml_area_file(area_file_name, *regions):
    """Parse area information from a yaml area file.

    Args:
        area_file_name: filename, file-like object, yaml string, or list of these.
        regions (str): Names of areas to parse. Optional and defaults to all areas
            in the file.

    The result of loading multiple area files is the combination of all
    the files, using the first file as the "base", replacing things after
    that.
    """
    area_dict = _read_yaml_area_file_content(area_file_name)
    area_list = regions or area_dict.keys()
    res = []
    for area_name in area_list:
        params = area_dict.get(area_name)
        if params is None:
            raise AreaNotFound('Area "{0}" not found in file "{1}"'.format(area_name, area_file_name))
        area_def = _create_area_def_from_dict(area_name, params)
        res.append(area_def)
    return res


def _create_area_def_from_dict(area_name, params):
    """Create an area definition from a string of parameters."""
    params.setdefault('area_id', area_name)
    # Optional arguments.
    params['shape'] = _capture_subarguments(params, 'shape', ['height', 'width'])
    params['upper_left_extent'] = _capture_subarguments(params, 'upper_left_extent', ['upper_left_extent', 'x', 'y',
                                                                                      'units'])
    params['center'] = _capture_subarguments(params, 'center', ['center', 'x', 'y', 'units'])
    params['area_extent'] = _capture_subarguments(params, 'area_extent', ['area_extent', 'lower_left_xy',
                                                                          'upper_right_xy', 'units'])
    params['resolution'] = _capture_subarguments(params, 'resolution', ['resolution', 'dx', 'dy', 'units'])
    params['radius'] = _capture_subarguments(params, 'radius', ['radius', 'dx', 'dy', 'units'])
    area_id = params.pop("area_id", area_name)
    kwargs = {}
    if "description" in params:
        kwargs["description"] = params.pop("description")
    if "proj_id" in params:
        kwargs["proj_id"] = params.pop("proj_id")
    area_def = create_area_def(
        area_id,
        params.pop("projection"),
        area_extent=params.pop("area_extent"),
        shape=params.pop("shape"),
        upper_left_extent=params.pop("upper_left_extent"),
        center=params.pop("center"),
        resolution=params.pop("resolution"),
        radius=params.pop("radius"),
        optimize_projection=params.pop("optimize_projection", False),
        **kwargs,
    )
    if params:
        warnings.warn(f"Unused/unexpected area definition parameter(s) for {area_id}: {params=}", stacklevel=4)
    return area_def


def _capture_subarguments(params: dict, arg_name: str, sub_arg_list: list[str]) -> Any:
    """Capture :func:`~pyresample.utils.create_area_def` sub-arguments (i.e. units, height, dx, etc) from a yaml file.

    Example:
        resolution:
          dx: 11
          dy: 22
          units: meters
        # returns DataArray((11, 22), attrs={'units': 'meters})
    """
    # Check if argument is in yaml.
    argument = params.get(arg_name)
    if not isinstance(argument, dict):
        return argument
    _validate_sub_arg_list(argument, arg_name, sub_arg_list)
    units = argument.pop('units', None)
    list_of_values = argument.pop(arg_name, [])
    for sub_arg in sub_arg_list:
        sub_arg_value = argument.get(sub_arg)
        # Don't append units to the argument.
        if sub_arg_value is None:
            continue
        if sub_arg in ('lower_left_xy', 'upper_right_xy') and isinstance(sub_arg_value, list):
            list_of_values.extend(sub_arg_value)
        else:
            list_of_values.append(sub_arg_value)
    # If units are provided, convert to xarray.
    if units is not None:
        return DataArray(list_of_values, attrs={'units': units})
    return list_of_values


def _validate_sub_arg_list(argument, arg_name, sub_arg_list):
    argument_keys = argument.keys()
    for sub_arg in argument_keys:
        # Verify that provided sub-arguments are valid.
        if sub_arg not in sub_arg_list:
            raise ValueError(f"Invalid area definition: {sub_arg} is not a valid sub-argument for {arg_name}")
        if arg_name in argument_keys:
            # If the arg_name is provided as a sub_arg, then it contains all the data and does not need other sub_args.
            if sub_arg != arg_name and sub_arg != "units":
                raise ValueError(
                    f"Invalid area definition: {arg_name} has too many sub-arguments: "
                    f"Both {arg_name} and {sub_arg} were specified.")
            # If the arg_name is provided, it's expected that units is also provided.
            if 'units' not in argument_keys:
                raise ValueError(f"Invalid area definition: {arg_name} has the sub-argument {arg_name} without units")


def _read_legacy_area_file_lines(area_file_name):
    if isinstance(area_file_name, str):
        area_file_name = [area_file_name]

    for area_file_obj in area_file_name:
        if (isinstance(area_file_obj, str) and
           not os.path.isfile(area_file_obj)):
            # file content string
            for line in area_file_obj.splitlines():
                yield line
            continue
        elif isinstance(area_file_obj, str):
            # filename
            with open(area_file_obj, 'r') as area_file:
                for line in area_file.readlines():
                    yield line


def _parse_legacy_area_file(area_file_name, *regions):
    """Parse area information from a legacy area file."""
    area_file = _read_legacy_area_file_lines(area_file_name)
    area_list = list(regions)
    select_all_areas = bool(not area_list)
    area_defs = [] if select_all_areas else [None for area_id in area_list]

    # Extract area from file
    for line in area_file:
        if "REGION" not in line or line.strip().startswith("#"):
            continue

        area_id = line.replace('REGION:', '').replace('{', '').strip()
        if area_id not in area_list and not select_all_areas:
            continue

        area_def = _parse_one_legacy_area_lines(area_file, area_id)
        if select_all_areas:
            area_defs.append(area_def)
        else:
            area_defs[area_list.index(area_id)] = area_def

    # Check if all specified areas were found
    if not select_all_areas:
        for i, area in enumerate(area_defs):
            if area is None:
                raise AreaNotFound('Area "%s" not found in file "%s"' %
                                   (area_list[i], area_file_name))
    return area_defs


def _parse_one_legacy_area_lines(area_file: Iterable[str], area_id: str):
    area_content = ""
    for line in area_file:
        if '};' in line:
            try:
                return _create_area(area_id, area_content)
            except KeyError as err:
                raise ValueError('Invalid area definition: %s, %s' % (area_id, area_content)) from err
        else:
            area_content += line


def _create_area(area_id, area_content):
    """Parse area configuration."""
    from configobj import ConfigObj
    config_obj = area_content.replace('{', '').replace('};', '')
    config_obj = ConfigObj([line.replace(':', '=', 1)
                            for line in config_obj.splitlines()])
    config = config_obj.dict()
    config['REGION'] = area_id

    if not isinstance(config['NAME'], str):
        config['NAME'] = ', '.join(config['NAME'])

    config['XSIZE'] = int(config['XSIZE'])
    config['YSIZE'] = int(config['YSIZE'])
    config['AREA_EXTENT'][0] = config['AREA_EXTENT'][0].replace('(', '')
    config['AREA_EXTENT'][3] = config['AREA_EXTENT'][3].replace(')', '')

    for i, val in enumerate(config['AREA_EXTENT']):
        config['AREA_EXTENT'][i] = float(val)

    config['PCS_DEF'] = _get_proj4_args(config['PCS_DEF'])
    return create_area_def(config['REGION'], config['PCS_DEF'], description=config['NAME'], proj_id=config['PCS_ID'],
                           shape=(config['YSIZE'], config['XSIZE']), area_extent=config['AREA_EXTENT'],
                           )


def get_area_def(area_id, area_name, proj_id, proj4_args, width, height, area_extent):
    """Construct AreaDefinition object from arguments.

    Parameters
    -----------
    area_id : str
        ID of area
    area_name :str
        Description of area
    proj_id : str
        ID of projection
    proj4_args : dict, CRS, or str
        Projection information passed to pyproj's CRS object
    width : int
        Number of pixel in x dimension
    height : int
        Number of pixel in y dimension
    area_extent : list | tuple
        Area extent as a list of ints (LL_x, LL_y, UR_x, UR_y)

    Returns
    -------
    area_def : object
        AreaDefinition object
    """
    return create_area_def(area_id, proj4_args, description=area_name, proj_id=proj_id,
                           shape=(height, width), area_extent=area_extent)


def _get_proj4_args(proj4_args):
    """Create dict from proj4 args."""
    from pyresample.utils.proj4 import convert_proj_floats
    if isinstance(proj4_args, str):
        # float conversion is done in `proj4_str_to_dict` already
        return proj4_str_to_dict(str(proj4_args))

    from configobj import ConfigObj
    proj_config = ConfigObj(proj4_args)
    return convert_proj_floats(proj_config.items())


def create_area_def(area_id, projection, width=None, height=None, area_extent=None, shape=None, upper_left_extent=None,
                    center=None, resolution=None, radius=None, units=None, optimize_projection=False, **kwargs):
    """Create AreaDefinition from whatever information is known.

    Parameters
    ----------
    area_id : str
        ID of area
    projection : pyproj CRS object, dict, str, int, tuple, object
        Projection parameters.  This can be in any format understood by
        :func:`pyproj.crs.CRS.from_user_input`, such as a pyproj CRS object,
        proj4 dict, proj4 string, EPSG integer code, or others.
    description : str, optional
        Description/name of area. Defaults to area_id
    proj_id : str, optional
        ID of projection (deprecated)
    units : str, optional
        Units that provided arguments should be interpreted as. This can be
        one of 'deg', 'degrees', 'meters', 'metres', and any
        parameter supported by the
        `cs2cs -lu <https://proj4.org/apps/cs2cs.html#cmdoption-cs2cs-lu>`_
        command. Units are determined in the following priority:

        1. units expressed with each variable through a DataArray's attrs attribute.
        2. units passed to ``units``
        3. units used in ``projection``
        4. meters

    width : str, optional
        Number of pixels in the x direction
    height : str, optional
        Number of pixels in the y direction
    area_extent : list, optional
        Area extent as a list (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    shape : list, optional
        Number of pixels in the y and x direction (height, width)
    upper_left_extent : list, optional
        Upper left corner of upper left pixel (x, y)
    center : list, optional
        Center of projection (x, y)
    resolution : list or float, optional
        Size of pixels: (dx, dy)
    radius : list or float, optional
        Length from the center to the edges of the projection (dx, dy)
    nprocs : int, optional
        Number of processor cores to be used
    lons : numpy array, optional
        Grid lons
    lats : numpy array, optional
        Grid lats
    optimize_projection:
        Whether the projection parameters have to be optimized for a DynamicAreaDefinition.

    Returns
    -------
    area : pyresample.geometry.AreaDefinition or pyresample.geometry.DynamicAreaDefinition
        If shape and area_extent are found, an AreaDefinition object is returned.
        If only shape or area_extent can be found, a DynamicAreaDefinition object is returned

    Raises
    ------
    ValueError:
        If neither shape nor area_extent could be found

    Notes
    -----
    * ``resolution`` and ``radius`` can be specified with one value if dx == dy
    * If ``resolution`` and ``radius`` are provided as angles, center must be given or findable. In such a case,
      they represent [projection x distance from center[0] to center[0]+dx, projection y distance from center[1] to
      center[1]+dy]
    """
    description = kwargs.pop('description', area_id)
    proj_id = kwargs.pop('proj_id', None)

    # convert EPSG dictionaries to projection string
    # (hold on to EPSG code as much as possible)
    if isinstance(projection, dict) and 'EPSG' in projection:
        projection = "EPSG:{}".format(projection['EPSG'])

    try:
        crs = _get_proj_data(projection)
        p = Proj(crs, preserve_units=True)
    except (RuntimeError, CRSError):
        # Assume that an invalid projection will be "fixed" by a dynamic area definition later
        return _make_area(area_id, description, proj_id, projection, shape, area_extent,
                          resolution=resolution, optimize_projection=optimize_projection,
                          **kwargs)

    # If no units are provided, try to get units used in proj_dict. If still none are provided, use meters.
    if units is None:
        units = _get_proj_units(crs)

    # Allow height and width to be provided for more consistency across functions in pyresample.
    if height is not None or width is not None:
        shape = _validate_variable(shape, (height, width), 'shape', ['height', 'width'])

    # Makes sure list-like objects are list-like, have the right shape, and contain only numbers.
    center = _verify_list('center', center, 2)
    radius = _verify_list('radius', radius, 2)
    upper_left_extent = _verify_list('upper_left_extent', upper_left_extent, 2)
    resolution = _verify_list('resolution', resolution, 2)
    shape = _verify_list('shape', shape, 2)
    area_extent = _verify_list('area_extent', area_extent, 4)

    # Converts from lat/lon to projection coordinates (x,y) if not in projection coordinates. Returns tuples.
    center = _convert_units(center, 'center', units, p, crs)
    upper_left_extent = _convert_units(upper_left_extent, 'upper_left_extent', units, p, crs)
    if area_extent is not None:
        # convert area extent, pass as (X, Y)
        area_extent_ll = area_extent[:2]
        area_extent_ur = area_extent[2:]
        area_extent_ll = _convert_units(area_extent_ll, 'area_extent', units, p, crs)
        area_extent_ur = _convert_units(area_extent_ur, 'area_extent', units, p, crs)
        area_extent = area_extent_ll + area_extent_ur

    # Fills in missing information to attempt to create an area definition.
    if area_extent is None or shape is None:
        area_extent, shape, resolution = \
            _extrapolate_information(area_extent, shape, center, radius,
                                     resolution, upper_left_extent, units,
                                     p, crs)
    return _make_area(area_id, description, proj_id, projection, shape,
                      area_extent, resolution=resolution, optimize_projection=optimize_projection,
                      **kwargs)


def _make_area(
        area_id: str,
        description: str,
        proj_id: str,
        projection: Union[dict, CRS],
        shape: tuple[int, ...] | None,
        area_extent: tuple[float, float, float, float] | None,
        optimize_projection: bool = False,
        resolution: tuple[float, float] | float | None = None,
        **kwargs):
    """Handle the creation of an area definition for create_area_def."""
    from pyresample.future.geometry import AreaDefinition
    from pyresample.geometry import DynamicAreaDefinition

    # If enough data is provided, create an AreaDefinition. If only shape or area_extent are found, make a
    # DynamicAreaDefinition. If not enough information was provided, raise a ValueError.
    if area_extent is not None and shape is not None:
        attrs = {
            "name": area_id,
            "description": description,
            "proj_id": proj_id,
        }
        # FUTURE: Don't add kwargs to attrs, switch to explicit "attrs"
        attrs.update(kwargs)
        area_def = AreaDefinition(projection, shape, area_extent, attrs=attrs)
        return area_def if pyresample.config.get("features.future_geometries", False) else area_def.to_legacy()

    height, width = (None, None) if shape is None else shape
    return DynamicAreaDefinition(area_id=area_id, description=description, projection=projection, width=width,
                                 height=height, area_extent=area_extent,
                                 resolution=resolution, optimize_projection=optimize_projection)


def _get_proj_data(projection: Any) -> CRS:
    """Take projection information and returns a proj CRS.

    Takes projection information in any format understood by
    :func:`pyproj.crs.CRS.from_user_input`.  There is special
    handling for the "EPSG:XXXX" case where "XXXX" is an EPSG
    number code. It can be provided as a string `"EPSG:XXXX"` or
    as a dictionary (when provided via YAML) as `{'EPSG': XXXX}`.
    If it is passed as a string ("EPSG:XXXX") then the rules of
    :func:`~pyresample.utils._proj.proj4_str_to_dict` are followed.  If a
    dictionary and pyproj 2.0+ is installed then the string `"EPSG:XXXX"`
    is passed to ``proj4_str_to_dict``. If pyproj<2.0 is installed then
    the string ``+init=EPSG:XXXX`` is passed to ``proj4_str_to_dict``
    which provides limited information to area config operations.
    """
    if isinstance(projection, dict) and 'EPSG' in projection:
        projection = "EPSG:{}".format(projection['EPSG'])
    return CRS.from_user_input(projection)


def _get_proj_units(crs):
    if crs.is_geographic:
        unit_name = 'degrees'
    else:
        unit_name = crs.axis_info[0].unit_name
    return {
        'metre': 'm',
        'meter': 'm',
        'kilometre': 'km',
        'kilometer': 'km',
    }.get(unit_name, unit_name)


def _sign(num):
    """Return the sign of the number provided.

    Returns:
        1 if number is greater than 0, -1 otherwise
    """
    return -1 if num < 0 else 1


def _round_poles(center, units, p):
    """Round center to the nearest pole if it is extremely close to said pole.

    Used to work around floating point precision issues .
    """
    # For a laea projection, this allows for an error of 11 meters around the pole.
    error = .0001
    if 'deg' in units:
        if abs(abs(center[1]) - 90) < error:
            center = (center[0], _sign(center[1]) * 90)
    else:
        center = p(*center, inverse=True, errcheck=True)
        if abs(abs(center[1]) - 90) < error:
            center = (center[0], _sign(center[1]) * 90)
        center = p(*center, errcheck=True)
    return center


def _distance_from_center_forward(
        var: tuple,
        center: tuple | None,
        p: Proj):
    """Convert distances in degrees to projection units."""
    # Interprets radius and resolution as distances between latitudes/longitudes.
    # Since the distance between longitudes and latitudes is not constant in
    # most projections, there must be reference point to start from.
    if center is None:
        center = (0, 0)

    center_as_angle = p(*center, inverse=True, errcheck=True)
    pole = 90
    # If on a pole, use northern/southern latitude for both height and width.
    if abs(abs(center_as_angle[1]) - pole) < 1e-3:
        direction_of_poles = _sign(center_as_angle[1])
        var = (center[1] - p(0, center_as_angle[1] - direction_of_poles * abs(var[0]),
                             errcheck=True)[1],
               center[1] - p(0, center_as_angle[1] - direction_of_poles * abs(var[1]),
                             errcheck=True)[1])
    # Uses southern latitude and western longitude if radius is positive. Uses northern latitude and
    # eastern longitude if radius is negative.
    else:
        var = (center[0] - p(center_as_angle[0] - var[0], center_as_angle[1], errcheck=True)[0],
               center[1] - p(center_as_angle[0], center_as_angle[1] - var[1], errcheck=True)[1])
    return var


def _convert_units(
        var,
        name: str,
        units: str,
        p: Proj,
        crs: CRS,
        inverse: bool = False,
        center=None):
    """Convert units from lon/lat to projection coordinates (meters).

    If `inverse` it True then the inverse calculation is done.
    """
    if var is None:
        return None
    # Convert from var projection units to projection units given by projection from user.
    var, units = _extract_and_validate_units(var, units, crs)
    is_angle = units == "degrees"
    if not is_angle:
        var = _convert_coordinate_for_metered_units(var, units, crs, p.crs)
    if name == "center":
        var = _round_poles(var, units, p)
    # Return either degrees or meters depending on if the inverse is true or not.
    # Don't convert if inverse is True: Want degrees.
    # Converts list-like from degrees to meters.
    if is_angle and not inverse:
        if name in ('radius', 'resolution'):
            var = _distance_from_center_forward(var, center, p)
        elif not crs.is_geographic:
            # only convert to meters
            # this allows geographic projections to use coordinates outside
            # normal lon/lat ranges (ex. -90/90)
            var = p(*var, errcheck=True)
    # Don't convert if inverse is False: Want meters.
    elif not is_angle and inverse:
        # Converts list-like from meters to degrees.
        var = p(*var, inverse=True, errcheck=True)
    if name in ['radius', 'resolution']:
        var = (abs(var[0]), abs(var[1]))
    return var


def _extract_and_validate_units(
        var: DataArray | tuple[float, float],
        units: str,
        crs: CRS
) -> tuple[tuple[float, float], str]:
    if isinstance(var, DataArray):
        units = var.attrs["units"]
        var = tuple(var.data.tolist())
    if "deg" == units or "degrees" == units:
        return var, "degrees"
    if crs.is_geographic:
        raise ValueError(f"latlon/latlong projection cannot take '{units}' as units")
    if "deg" in units:
        raise ValueError(f"Invalid degrees-like units: {units}")
    if units == 'meters' or units == 'metres':
        return var, "m"
    return var, units


def _convert_coordinate_for_metered_units(var, units: str, src_crs: CRS, dst_crs: CRS):
    if _get_proj_units(src_crs) == units:
        return var
    tmp_proj_dict = src_crs.to_dict()
    tmp_proj_dict['units'] = units
    transformer = Transformer.from_crs(tmp_proj_dict, dst_crs)
    return transformer.transform(*var)


def _round_shape(shape, radius=None, resolution=None):
    """Make sure shape is an integer.

    Rounds down if shape is less than .01 above nearest whole number to
    handle floating point precision issues. Otherwise the number is
    round up.
    """
    # Used for area definition to prevent indexing None.
    if shape is None:
        return None
    incorrect_shape = False
    height, width = shape
    if abs(width - round(width)) > 1e-8:
        incorrect_shape = True
        if width - math.floor(width) >= .01:
            width = math.ceil(width)
    width = int(round(width))
    if abs(height - round(height)) > 1e-8:
        incorrect_shape = True
        if height - math.floor(height) >= .01:
            height = math.ceil(height)
    height = int(round(height))
    if incorrect_shape:
        if radius is not None and resolution is not None:
            new_resolution = (2 * radius[0] / width, 2 * radius[1] / height)
            logging.warning('shape found from radius and resolution does not contain only '
                            'integers: {0}\nRounding shape to {1} and resolution from {2} meters to '
                            '{3} meters'.format(shape, (height, width), resolution, new_resolution))
        else:
            logging.warning('shape provided does not contain only integers: {0}\n'
                            'Rounding shape to {1}'.format(shape, (height, width)))
    return height, width


def _validate_variable(var, new_var, var_name, input_list):
    """Make sure data given by the user does not conflict with itself.

    If a variable that was given by the user contradicts other data provided, an exception is raised.
    Example: upper_left_extent is (-10, 10), but area_extent is (-20, -20, 20, 20).
    """
    if var is not None and not np.allclose(np.array(var, dtype=float), np.array(new_var, dtype=float), equal_nan=True):
        raise ValueError('CONFLICTING DATA: {0} given does not match {0} found from {1}'.format(
            var_name, ', '.join(input_list)) + ':\ngiven: {0}\nvs\nfound: {1}'.format(var, new_var))
    return new_var


def _extrapolate_information(area_extent, shape, center, radius, resolution, upper_left_extent, units, p, crs):
    """Attempt to find shape and area_extent based on data provided.

    Parameters are used in a specific order to determine area_extent and shape.
    The area_extent and shape are later used to create an `AreaDefinition`.
    Providing some parameters may have no effect if other parameters could be
    used to determine area_extent and shape. The order of the parameters used
    is:

    1. area_extent
    2. upper_left_extent and center
    3. radius and resolution
    4. resolution and shape
    5. radius and center
    6. upper_left_extent and radius
    """
    # Input unaffected by data below: When area extent is calculated, it's either with
    # shape (giving you an area definition) or with center/radius/upper_left_extent (which this produces).
    # Yet output (center/radius/upper_left_extent) is essential for data below.
    if area_extent is not None:
        # Function 1-A
        new_center = ((area_extent[2] + area_extent[0]) / 2, (area_extent[3] + area_extent[1]) / 2)
        center = _validate_variable(center, new_center, 'center', ['area_extent'])
        # If radius is given in an angle without center it will raise an exception, and to verify, it must be in meters.
        radius = _convert_units(radius, 'radius', units, p, crs, center=center)
        new_radius = ((area_extent[2] - area_extent[0]) / 2, (area_extent[3] - area_extent[1]) / 2)
        radius = _validate_variable(radius, new_radius, 'radius', ['area_extent'])
        new_upper_left_extent = (area_extent[0], area_extent[3])
        upper_left_extent = _validate_variable(
            upper_left_extent, new_upper_left_extent, 'upper_left_extent', ['area_extent'])
    # Output used below, but nowhere else is upper_left_extent made. Thus it should go as early as possible.
    elif None not in (upper_left_extent, center):
        # Function 1-B
        radius = _convert_units(radius, 'radius', units, p, crs, center=center)
        new_radius = (center[0] - upper_left_extent[0], upper_left_extent[1] - center[1])
        radius = _validate_variable(radius, new_radius, 'radius', ['upper_left_extent', 'center'])
    else:
        radius = _convert_units(radius, 'radius', units, p, crs, center=center)
    # Convert resolution to meters if given as an angle. If center is not found, an exception is raised.
    resolution = _convert_units(resolution, 'resolution', units, p, crs, center=center)
    # Inputs unaffected by data below: area_extent is not an input. However, output is used below.
    if radius is not None and resolution is not None:
        # Function 2-A
        new_shape = _round_shape((2 * radius[1] / resolution[1], 2 * radius[0] / resolution[0]), radius=radius,
                                 resolution=resolution)
        shape = _validate_variable(shape, new_shape, 'shape', ['radius', 'resolution'])
    elif resolution is not None and shape is not None:
        # Function 2-B
        new_radius = (resolution[0] * shape[1] / 2, resolution[1] * shape[0] / 2)
        radius = _validate_variable(radius, new_radius, 'radius', ['shape', 'resolution'])
    # Input determined from above functions, but output does not affect above functions: area_extent can be
    # used to find center/upper_left_extent which are used to find each other, which is redundant.
    if center is not None and radius is not None:
        # Function 1-C
        new_area_extent = (center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1])
        area_extent = _validate_variable(area_extent, new_area_extent, 'area_extent', ['center', 'radius'])
    elif upper_left_extent is not None and radius is not None:
        # Function 1-D
        new_area_extent = (
            upper_left_extent[0], upper_left_extent[1] - 2 * radius[1], upper_left_extent[0] + 2 * radius[0],
            upper_left_extent[1])
        area_extent = _validate_variable(area_extent, new_area_extent, 'area_extent', ['upper_left_extent', 'radius'])
    return area_extent, shape, resolution


def _format_list(var, name):
    """Ensure that parameter is list-like of numbers.

    Used to let resolution and radius be single numbers if their elements are equal.

    """
    # Single-number format.
    if not isinstance(var, (list, tuple)) and name in ('resolution', 'radius'):
        var = (float(var), float(var))
    elif name == 'shape':
        var = _round_shape(var)
    else:
        var = tuple(float(num) for num in var)
    return var


def _verify_list(name, var, length):
    """Check that list-like variables are list-like, shapes are accurate, and values are numbers."""
    # Make list-like data into tuples (or leave as xarrays). If not list-like, throw a ValueError unless it is None.
    if var is None:
        return None
    # Verify that list is made of numbers and is list-like.
    try:
        if 'units' in getattr(var, 'attrs', {}) and name != 'shape':
            # For len(var) to work, DataArray must contain a list, not a tuple
            var = DataArray(list(_format_list(var.data.tolist(), name)), attrs=var.attrs)
        elif isinstance(var, DataArray):
            if name == 'shape':
                logging.warning("{0} is unitless, but was passed as a DataArray".format(name))
            else:
                logging.warning("{0} is a DataArray but does not have the attribute 'units',"
                                "but instead has attribute(s): {1}".format(name, var.attrs))
            var = _format_list(var.data.tolist(), name)
        else:
            var = _format_list(var, name)
    except TypeError as err:
        raise ValueError('{0} is not list-like:\n{1}'.format(name, var)) from err
    except ValueError as err:
        raise ValueError('{0} is not composed purely of numbers:\n{1}'.format(name, var)) from err
    # Confirm correct shape
    if len(var) != length:
        raise ValueError('{0} should have length {1}, but instead has length {2}:\n{3}'.format(name, length,
                                                                                               len(var), var))
    return var


def convert_def_to_yaml(def_area_file, yaml_area_file):
    """Convert a legacy area def file to the yaml counter partself.

    *yaml_area_file* will be overwritten by the operation.
    """
    areas = _parse_legacy_area_file(def_area_file)
    with open(yaml_area_file, 'w') as yaml_file:
        for area in areas:
            yaml_file.write(area.create_areas_def())


def generate_area_def_rst_list(area_file: str) -> str:
    """Create rst list of available area definitions with overview plot.

    Args:
        area_file : Path to area yaml file.

    Returns:
        rst list formatted string.
    """
    area_list: List[str] = []

    template = ("{area_name}\n"
                "{n:^>{header_title_length}}\n\n"
                ".. raw:: html\n\n"
                "{content}\n\n"
                "     <hr>\n\n")

    for aname, params in _read_yaml_area_file_content(area_file).items():
        area = _create_area_def_from_dict(aname, params)
        if not hasattr(area, "_repr_html_"):
            continue

        area_rep = area_repr(area, include_header=False, include_static_files=not bool(area_list))

        content = "\n".join([x.rjust(len(x) + 5) for x in area_rep.split("\n")])
        area_list.append(template.format(area_name=aname, n="", header_title_length=len(aname),
                                         content=content))

    return "".join(area_list)
