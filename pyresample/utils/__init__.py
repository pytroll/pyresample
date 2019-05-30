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
"""Miscellaneous utility functions for pyresample."""
from collections import Mapping
import numpy as np
import pyproj
import warnings

from ._proj4 import (proj4_dict_to_str, proj4_str_to_dict, convert_proj_floats, proj4_radius_parameters)  # noqa
from ._rasterio import get_area_def_from_raster  # noqa


def get_area_def(*args, **kwargs):
    from pyresample.area_config import get_area_def
    warnings.warn("'get_area_def' has moved, import it with 'from pyresample import get_area_def'")
    return get_area_def(*args, **kwargs)


def create_area_def(*args, **kwargs):
    from pyresample.area_config import create_area_def
    warnings.warn("'create_area_def' has moved, import it with 'from pyresample import create_area_def'")
    return create_area_def(*args, **kwargs)


def load_area(*args, **kwargs):
    from pyresample.area_config import load_area
    warnings.warn("'load_area' has moved, import it with 'from pyresample import load_area'")
    return load_area(*args, **kwargs)


def convert_def_to_yaml(*args, **kwargs):
    from pyresample.area_config import convert_def_to_yaml
    warnings.warn("'convert_def_to_yaml' has moved, import it with 'from pyresample import convert_def_to_yaml'")
    return convert_def_to_yaml(*args, **kwargs)


def parse_area_file(*args, **kwargs):
    from pyresample.area_config import parse_area_file
    warnings.warn("'parse_area_file' has moved, import it with 'from pyresample import parse_area_file'")
    return parse_area_file(*args, **kwargs)


def generate_quick_linesample_arrays(source_area_def, target_area_def, nprocs=1):
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


def is_pyproj2():
    """Determine whether the current pyproj version is >= 2.0"""
    return pyproj.__version__ >= '2'
