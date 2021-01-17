#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test EWA Dask-based resamplers."""

import logging
import unittest
from unittest import mock

try:
    from pyproj import CRS
except ImportError:
    CRS = None

LOG = logging.getLogger(__name__)


def get_test_data(input_shape=(100, 50), output_shape=(200, 100), output_proj=None,
                  input_dims=('y', 'x')):
    """Get common data objects used in testing.

    Returns: tuple with the following elements
        input_data_on_area: DataArray with dimensions as if it is a gridded
            dataset.
        input_area_def: AreaDefinition of the above DataArray
        input_data_on_swath: DataArray with dimensions as if it is a swath.
        input_swath: SwathDefinition of the above DataArray
        target_area_def: AreaDefinition to be used as a target for resampling

    """
    from xarray import DataArray
    import dask.array as da
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from pyresample.utils import proj4_str_to_dict
    ds1 = DataArray(da.zeros(input_shape, chunks=85),
                    dims=input_dims,
                    attrs={'name': 'test_data_name', 'test': 'test'})
    if input_dims and 'y' in input_dims:
        ds1 = ds1.assign_coords(y=da.arange(input_shape[-2], chunks=85))
    if input_dims and 'x' in input_dims:
        ds1 = ds1.assign_coords(x=da.arange(input_shape[-1], chunks=85))
    if input_dims and 'bands' in input_dims:
        ds1 = ds1.assign_coords(bands=list('RGBA'[:ds1.sizes['bands']]))

    input_proj_str = ('+proj=geos +lon_0=-95.0 +h=35786023.0 +a=6378137.0 '
                      '+b=6356752.31414 +sweep=x +units=m +no_defs')
    source = AreaDefinition(
        'test_target',
        'test_target',
        'test_target',
        proj4_str_to_dict(input_proj_str),
        input_shape[1],  # width
        input_shape[0],  # height
        (-1000., -1500., 1000., 1500.))
    ds1.attrs['area'] = source
    if CRS is not None:
        crs = CRS.from_string(input_proj_str)
        ds1 = ds1.assign_coords(crs=crs)

    ds2 = ds1.copy()
    input_area_shape = tuple(ds1.sizes[dim] for dim in ds1.dims
                             if dim in ['y', 'x'])
    geo_dims = ('y', 'x') if input_dims else None
    lons = da.random.random(input_area_shape, chunks=50)
    lats = da.random.random(input_area_shape, chunks=50)
    swath_def = SwathDefinition(
        DataArray(lons, dims=geo_dims),
        DataArray(lats, dims=geo_dims))
    ds2.attrs['area'] = swath_def
    if CRS is not None:
        crs = CRS.from_string('+proj=latlong +datum=WGS84 +ellps=WGS84')
        ds2 = ds2.assign_coords(crs=crs)

    # set up target definition
    output_proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                       '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
    output_proj_str = output_proj or output_proj_str
    target = AreaDefinition(
        'test_target',
        'test_target',
        'test_target',
        proj4_str_to_dict(output_proj_str),
        output_shape[1],  # width
        output_shape[0],  # height
        (-1000., -1500., 1000., 1500.),
    )
    return ds1, source, ds2, swath_def, target


class TestLegacyDaskEWAResampler(unittest.TestCase):
    """Test Legacy Dask EWA resampler class."""

    @mock.patch('pyresample.ewa._legacy_dask_ewa.fornav')
    @mock.patch('pyresample.ewa._legacy_dask_ewa.ll2cr')
    @mock.patch('pyresample.ewa._legacy_dask_ewa.SwathDefinition.get_lonlats')
    def test_2d_ewa(self, get_lonlats, ll2cr, fornav):
        """Test EWA with a 2D dataset."""
        import numpy as np
        import xarray as xr
        from pyresample.ewa import LegacyDaskEWAResampler
        ll2cr.return_value = (100,
                              np.zeros((10, 10), dtype=np.float32),
                              np.zeros((10, 10), dtype=np.float32))
        fornav.return_value = (100 * 200,
                               np.zeros((200, 100), dtype=np.float32))
        _, _, swath_data, source_swath, target_area = get_test_data()
        get_lonlats.return_value = (source_swath.lons, source_swath.lats)
        swath_data.data = swath_data.data.astype(np.float32)
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        resampler = LegacyDaskEWAResampler(source_swath, target_area)
        new_data = resampler.resample(swath_data)
        self.assertTupleEqual(new_data.shape, (200, 100))
        self.assertEqual(new_data.dtype, np.float32)
        self.assertEqual(new_data.attrs['test'], 'test')
        self.assertIs(new_data.attrs['area'], target_area)
        # make sure we can actually compute everything
        new_data.compute()
        lonlat_calls = get_lonlats.call_count
        ll2cr_calls = ll2cr.call_count

        # resample a different dataset and make sure cache is used
        data = xr.DataArray(
            swath_data.data,
            dims=('y', 'x'), attrs={'area': source_swath, 'test': 'test2',
                                    'name': 'test2'})
        new_data = resampler.resample(data)
        new_data.compute()
        # ll2cr will be called once more because of the computation
        self.assertEqual(ll2cr.call_count, ll2cr_calls + num_chunks)
        # but we should already have taken the lonlats from the SwathDefinition
        self.assertEqual(get_lonlats.call_count, lonlat_calls)
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')
            if hasattr(target_area, 'crs'):
                self.assertIs(target_area.crs, new_data.coords['crs'].item())

    @mock.patch('pyresample.ewa._legacy_dask_ewa.fornav')
    @mock.patch('pyresample.ewa._legacy_dask_ewa.ll2cr')
    @mock.patch('pyresample.ewa._legacy_dask_ewa.SwathDefinition.get_lonlats')
    def test_3d_ewa(self, get_lonlats, ll2cr, fornav):
        """Test EWA with a 3D dataset."""
        import numpy as np
        import xarray as xr
        from pyresample.ewa import LegacyDaskEWAResampler
        _, _, swath_data, source_swath, target_area = get_test_data(
            input_shape=(3, 200, 100), input_dims=('bands', 'y', 'x'))
        swath_data.data = swath_data.data.astype(np.float32)
        ll2cr.return_value = (100,
                              np.zeros((10, 10), dtype=np.float32),
                              np.zeros((10, 10), dtype=np.float32))
        fornav.return_value = ([100 * 200] * 3,
                               [np.zeros((200, 100), dtype=np.float32)] * 3)
        get_lonlats.return_value = (source_swath.lons, source_swath.lats)
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        resampler = LegacyDaskEWAResampler(source_swath, target_area)
        new_data = resampler.resample(swath_data)
        self.assertTupleEqual(new_data.shape, (3, 200, 100))
        self.assertEqual(new_data.dtype, np.float32)
        self.assertEqual(new_data.attrs['test'], 'test')
        self.assertIs(new_data.attrs['area'], target_area)
        # make sure we can actually compute everything
        new_data.compute()
        lonlat_calls = get_lonlats.call_count
        ll2cr_calls = ll2cr.call_count

        # resample a different dataset and make sure cache is used
        swath_data = xr.DataArray(
            swath_data.data,
            dims=('bands', 'y', 'x'), coords={'bands': ['R', 'G', 'B']},
            attrs={'area': source_swath, 'test': 'test'})
        new_data = resampler.resample(swath_data)
        new_data.compute()
        # ll2cr will be called once more because of the computation
        self.assertEqual(ll2cr.call_count, ll2cr_calls + num_chunks)
        # but we should already have taken the lonlats from the SwathDefinition
        self.assertEqual(get_lonlats.call_count, lonlat_calls)
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        self.assertIn('bands', new_data.coords)
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')
            np.testing.assert_equal(new_data.coords['bands'].values,
                                    ['R', 'G', 'B'])
            if hasattr(target_area, 'crs'):
                self.assertIs(target_area.crs, new_data.coords['crs'].item())


