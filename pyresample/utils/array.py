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
"""Utilities for converting array."""

import dask.array as da
import numpy as np
import xarray as xr

VALID_FORMAT = ['Numpy', 'Dask', 'DataArray_Numpy', 'DataArray_Dask']


def _numpy_conversion(arr, to, dims=None):
    if not isinstance(to, str):
        raise TypeError("'to' must be a string indicating the conversion array format.")
    if not isinstance(arr, np.ndarray):
        raise TypeError("_numpy_conversion expects a np.ndarray as input.")
    if not np.isin(to.lower(), np.char.lower(VALID_FORMAT)):
        raise ValueError("Valid _numpy_conversion array formats are {}".format(VALID_FORMAT))
    if to.lower() == 'numpy':
        dst_arr = arr
    elif to.lower() == 'dask':
        dst_arr = da.from_array(arr)
    elif to.lower() == 'dataarray_numpy':
        dst_arr = xr.DataArray(arr, dims=dims)
    else:  # to.lower() == 'dataarray_dask':
        dst_arr = xr.DataArray(da.from_array(arr), dims=dims)
    return dst_arr


def _dask_conversion(arr, to, dims=None):
    if not isinstance(to, str):
        raise TypeError("'to' must be a string indicating the conversion array format.")
    if not isinstance(arr, da.Array):
        raise TypeError("_dask_conversion expects a dask.Array as input.")
    if not np.isin(to.lower(), np.char.lower(VALID_FORMAT)):
        raise ValueError("Valid _dask_conversion array formats are {}".format(VALID_FORMAT))
    if to.lower() == 'numpy':
        dst_arr = arr.compute()
    elif to.lower() == 'dask':
        dst_arr = arr
    elif to.lower() == 'dataarray_numpy':
        dst_arr = xr.DataArray(arr.compute(), dims=dims)
    else:  # to.lower() == 'dataarray_dask':
        dst_arr = xr.DataArray(arr, dims=dims)
    return dst_arr


def _xr_numpy_conversion(arr, to):
    if not isinstance(to, str):
        raise TypeError("'to' must be a string indicating the conversion array format.")
    if not isinstance(arr, xr.DataArray):
        raise TypeError("_xr_numpy_conversion expects a xr.DataArray with numpy array as input.")
        if not isinstance(arr.data, np.ndarray):
            raise TypeError("_xr_numpy_conversion expects a xr.DataArray with numpy array as input.")
    if not np.isin(to.lower(), np.char.lower(VALID_FORMAT)):
        raise ValueError("Valid _xr_numpy_conversion array formats are {}".format(VALID_FORMAT))
    if to.lower() == 'numpy':
        dst_arr = arr.data
    elif to.lower() == 'dask':
        dst_arr = da.from_array(arr.data)
    elif to.lower() == 'dataarray_numpy':
        dst_arr = arr
    else:  # to.lower() == 'dataarray_dask':
        dst_arr = xr.DataArray(da.from_array(arr.data), dims=arr.dims)
    return dst_arr


def _xr_dask_conversion(arr, to):
    if not isinstance(to, str):
        raise TypeError("'to' must be a string indicating the conversion array format.")
    if not isinstance(arr, xr.DataArray):
        raise TypeError("_xr_dask_conversion expects a xr.DataArray with dask.Array as input.")
        if not isinstance(arr.data, da.Array):
            raise TypeError("_xr_dask_conversion expects a xr.DataArray with dask.Array as input.")
    if not np.isin(to.lower(), np.char.lower(VALID_FORMAT)):
        raise ValueError("Valid _xr_dask_conversion array formats are {}".format(VALID_FORMAT))
    if to.lower() == 'numpy':
        dst_arr = arr.data.compute()
    elif to.lower() == 'dask':
        dst_arr = arr.data
    elif to.lower() == 'dataarray_numpy':
        dst_arr = arr.compute()
    else:  # to.lower() == 'dataarray_dask':
        dst_arr = arr
    return dst_arr


def _convert_2D_array(arr, to, dims=None):
    """
    Convert a 2D array to a specific format.

    Useful to return swath lons, lats in the same original format after processing.

    Parameters
    ----------
    arr : (np.ndarray, da.Array, xr.DataArray)
        The 2D array to be converted to another array format.
    to : TYPE
        The desired array output format.
        Accepted formats are: ['Numpy','Dask', 'DataArray_Numpy','DataArray_Dask']
    dims : tuple, optional
        Optional argument for the specification of xr.DataArray dimension names
        if the input array is Numpy or Dask.
        Does not have any impact if the input is already a xr.DataArray
        Provide a tuple with (y_dimname, x_dimname).
        The default is None --> (dim_0, dim_1)


    Returns
    -------
    dst_arr : (np.ndarray, da.Array, xr.DataArray)
        The converted 2D array.
    src_format: str
        The source format of the 2D array.

    """
    # Checks
    if not isinstance(to, str):
        raise TypeError("'to' must be a string indicating the conversion array format.")
    if not np.isin(to.lower(), np.char.lower(VALID_FORMAT)):
        raise ValueError("Valid conversion array formats are {}".format(VALID_FORMAT))
    if not isinstance(arr, (np.ndarray, da.Array, xr.DataArray)):
        raise TypeError("The provided array must be either a np.ndarray, a dask.Array or a xr.DataArray.")
    # Numpy
    if isinstance(arr, np.ndarray):
        return _numpy_conversion(arr, to=to, dims=dims), "numpy"
    # Dask
    elif isinstance(arr, da.Array):
        return _dask_conversion(arr, to=to, dims=dims), 'dask'
    # DataArray_Numpy
    elif isinstance(arr, xr.DataArray) and isinstance(arr.data, np.ndarray):
        return _xr_numpy_conversion(arr, to=to), 'DataArray_Numpy'
    # DataArray_Dask
    else:  # isinstance(arr, xr.DataArray) and isinstance(arr.data, da.Array):
        return _xr_dask_conversion(arr, to=to), 'DataArray_Dask'
