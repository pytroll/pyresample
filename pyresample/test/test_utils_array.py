#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2022 Pyresample developers
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
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test utils array ."""
import unittest

import dask.array as da
import numpy as np
import xarray as xr

VALID_FORMAT = ['Numpy', 'Dask', 'DataArray_Numpy', 'DataArray_Dask']

lons_np = np.arange(-179.5, -177.5, 0.5)
lats_np = np.arange(-89.5, -88.0, 0.5)
lons_np, lats_np = np.meshgrid(lons_np, lats_np)
lats_dask = da.from_array(lats_np, chunks=2)
lats_xr = xr.DataArray(lats_np, dims=['y', 'x'])
lats_xr_dask = xr.DataArray(lats_dask, dims=['y', 'x'])
dict_format = {'Numpy': lats_np,
               'Dask': lats_dask,
               'DataArray_Numpy': lats_xr,
               'DataArray_Dask': lats_xr_dask
               }


class TestArrayConversion(unittest.TestCase):
    """Unit testing the array conversion."""

    def test_numpy_conversion(self):
        from pyresample.utils.array import _numpy_conversion
        in_format = "Numpy"
        dims = None
        for out_format in VALID_FORMAT:
            out_arr = _numpy_conversion(dict_format[in_format], to=out_format, dims=dims)
            assert isinstance(out_arr, type(dict_format[out_format]))
            if out_format.lower() == "dataarray_numpy":
                assert isinstance(out_arr.data, type(dict_format['Numpy']))
            if out_format.lower() == "dataarray_dask":
                assert isinstance(out_arr.data, type(dict_format['Dask']))
        # Test raise errors
        self.assertRaises(TypeError, _numpy_conversion, dict_format[in_format], ['unvalid_type'])
        self.assertRaises(TypeError, _numpy_conversion, ['unvalid_type'], in_format)
        self.assertRaises(ValueError, _numpy_conversion, dict_format[in_format], 'unvalid_format')

    def test_dask_conversion(self):
        from pyresample.utils.array import _dask_conversion
        in_format = "Dask"
        dims = None
        for out_format in VALID_FORMAT:
            out_arr = _dask_conversion(dict_format[in_format], to=out_format, dims=dims)
            assert isinstance(out_arr, type(dict_format[out_format]))
            if out_format.lower() == "dataarray_numpy":
                assert isinstance(out_arr.data, type(dict_format['Numpy']))
            if out_format.lower() == "dataarray_dask":
                assert isinstance(out_arr.data, type(dict_format['Dask']))
        # Test raise errors
        self.assertRaises(TypeError, _dask_conversion, dict_format[in_format], ['unvalid_type'])
        self.assertRaises(TypeError, _dask_conversion, ['unvalid_type'], in_format)
        self.assertRaises(ValueError, _dask_conversion, dict_format[in_format], 'unvalid_format')

    def test_xr_numpy_conversion(self):
        from pyresample.utils.array import _xr_numpy_conversion
        in_format = "DataArray_Numpy"
        for out_format in VALID_FORMAT:
            out_arr = _xr_numpy_conversion(dict_format[in_format], to=out_format)
            assert isinstance(out_arr, type(dict_format[out_format]))
            if out_format.lower() == "dataarray_numpy":
                assert isinstance(out_arr.data, type(dict_format['Numpy']))
            if out_format.lower() == "dataarray_dask":
                assert isinstance(out_arr.data, type(dict_format['Dask']))
        # Test raise errors
        self.assertRaises(TypeError, _xr_numpy_conversion, dict_format[in_format], ['unvalid_type'])
        self.assertRaises(TypeError, _xr_numpy_conversion, ['unvalid_type'], in_format)
        self.assertRaises(ValueError, _xr_numpy_conversion, dict_format[in_format], 'unvalid_format')

    def test_xr_dask_conversion(self):
        from pyresample.utils.array import _xr_dask_conversion
        in_format = "DataArray_Dask"
        for out_format in VALID_FORMAT:
            out_arr = _xr_dask_conversion(dict_format[in_format], to=out_format)
            assert isinstance(out_arr, type(dict_format[out_format]))
            if out_format.lower() == "dataarray_numpy":
                assert isinstance(out_arr.data, type(dict_format['Numpy']))
            if out_format.lower() == "dataarray_dask":
                assert isinstance(out_arr.data, type(dict_format['Dask']))
        # Test raise errors
        self.assertRaises(TypeError, _xr_dask_conversion, dict_format[in_format], ['unvalid_type'])
        self.assertRaises(TypeError, _xr_dask_conversion, ['unvalid_type'], in_format)
        self.assertRaises(ValueError, _xr_dask_conversion, dict_format[in_format], 'unvalid_format')

    def test_convert_2D_array(self):
        """Test conversion of 2D arrays between various formats."""
        from pyresample.utils.array import _convert_2D_array
        dims = None
        for in_format in VALID_FORMAT:
            for out_format in VALID_FORMAT:
                out_arr, src_format = _convert_2D_array(dict_format[in_format], to=out_format, dims=dims)
                assert isinstance(out_arr, type(dict_format[out_format]))
                assert src_format.lower() == in_format.lower()
                if out_format.lower() == "dataarray_numpy":
                    assert isinstance(out_arr.data, type(dict_format['Numpy']))
                if out_format.lower() == "dataarray_dask":
                    assert isinstance(out_arr.data, type(dict_format['Dask']))
        # Test raise errors
        self.assertRaises(TypeError, _convert_2D_array, dict_format['Numpy'], ['unvalid_type'])
        self.assertRaises(TypeError, _convert_2D_array, [dict_format['Numpy']], 'numpy')
        self.assertRaises(ValueError, _convert_2D_array, dict_format['Numpy'], 'unvalid_format')
