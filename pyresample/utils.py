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

import numpy as np
from configobj import ConfigObj

import pyresample as pr
import six


class AreaNotFound(Exception):

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

    area_file = open(area_file_name, 'r')
    area_list = list(regions)
    if len(area_list) == 0:
        select_all_areas = True
        area_defs = []
    else:
        select_all_areas = False
        area_defs = [None for i in area_list]

    # Extract area from file
    in_area = False
    for line in area_file.readlines():
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

    area_file.close()

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
    config['AREA_EXTENT'][0] = config['AREA_EXTENT'][0].replace('(', '')
    config['AREA_EXTENT'][3] = config['AREA_EXTENT'][3].replace(')', '')

    for i, val in enumerate(config['AREA_EXTENT']):
        config['AREA_EXTENT'][i] = float(val)

    config['PCS_DEF'] = _get_proj4_args(config['PCS_DEF'])

    return pr.geometry.AreaDefinition(config['REGION'], config['NAME'],
                                      config['PCS_ID'], config['PCS_DEF'],
                                      config['XSIZE'], config['YSIZE'],
                                      config['AREA_EXTENT'])


def get_area_def(area_id, area_name, proj_id, proj4_args, x_size, y_size,
                 area_extent):
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
    area_extent : list 
        Area extent as a list of ints (LL_x, LL_y, UR_x, UR_y)

    Returns
    -------
    area_def : object
        AreaDefinition object
    """

    proj_dict = _get_proj4_args(proj4_args)
    return pr.geometry.AreaDefinition(area_id, area_name, proj_id, proj_dict, x_size,
                                      y_size, area_extent)


def generate_quick_linesample_arrays(source_area_def, target_area_def, nprocs=1):
    """Generate linesample arrays for quick grid resampling

    Parameters
    -----------
    source_area_def : object 
        Source area definition as AreaDefinition object
    target_area_def : object 
        Target area definition as AreaDefinition object
    nprocs : int, optional 
        Number of processor cores to be used

    Returns
    -------
    (row_indices, col_indices) : tuple of numpy arrays
    """
    if not (isinstance(source_area_def, pr.geometry.AreaDefinition) and
            isinstance(target_area_def, pr.geometry.AreaDefinition)):
        raise TypeError('source_area_def and target_area_def must be of type '
                        'geometry.AreaDefinition')

    lons, lats = target_area_def.get_lonlats(nprocs)

    source_pixel_y, source_pixel_x = pr.grid.get_linesample(lons, lats,
                                                            source_area_def,
                                                            nprocs=nprocs)

    source_pixel_x = _downcast_index_array(source_pixel_x,
                                           source_area_def.shape[1])
    source_pixel_y = _downcast_index_array(source_pixel_y,
                                           source_area_def.shape[0])

    return source_pixel_y, source_pixel_x


def generate_nearest_neighbour_linesample_arrays(source_area_def, target_area_def,
                                                 radius_of_influence, nprocs=1):
    """Generate linesample arrays for nearest neighbour grid resampling

    Parameters
    -----------
    source_area_def : object 
        Source area definition as AreaDefinition object
    target_area_def : object 
        Target area definition as AreaDefinition object
    radius_of_influence : float 
        Cut off distance in meters
    nprocs : int, optional 
        Number of processor cores to be used

    Returns
    -------
    (row_indices, col_indices) : tuple of numpy arrays
    """

    if not (isinstance(source_area_def, pr.geometry.AreaDefinition) and
            isinstance(target_area_def, pr.geometry.AreaDefinition)):
        raise TypeError('source_area_def and target_area_def must be of type '
                        'geometry.AreaDefinition')

    valid_input_index, valid_output_index, index_array, distance_array = \
        pr.kd_tree.get_neighbour_info(source_area_def,
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
    lons_wrap = (lons + 180) % (360) - 180
    return lons_wrap.astype(lons.dtype)
