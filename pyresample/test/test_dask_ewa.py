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
from unittest import mock

import dask
import numpy as np
import pytest
from pyproj import CRS

import pyresample.ewa

da = pytest.importorskip("dask.array")
xr = pytest.importorskip("xarray")
dask_ewa = pytest.importorskip("pyresample.ewa.dask_ewa")
legacy_dask_ewa = pytest.importorskip("pyresample.ewa._legacy_dask_ewa")
DaskEWAResampler = pyresample.ewa.DaskEWAResampler
LegacyDaskEWAResampler = pyresample.ewa.LegacyDaskEWAResampler


LOG = logging.getLogger(__name__)


def _fill_mask(data):
    if np.issubdtype(data.dtype, np.floating):
        return np.isnan(data)
    elif np.issubdtype(data.dtype, np.integer):
        return data == np.iinfo(data.dtype).max
    else:
        raise ValueError("Not sure how to get fill mask.")


def _get_test_array(input_shape, input_dtype, chunk_size):
    if np.issubdtype(input_dtype, np.integer):
        dinfo = np.iinfo(input_dtype)
        data = da.random.randint(dinfo.min + 1, dinfo.max, size=input_shape,
                                 chunks=chunk_size, dtype=input_dtype)
    else:
        data = da.random.random(input_shape, chunks=chunk_size).astype(input_dtype)
    fill_value = 127 if np.issubdtype(input_dtype, np.integer) else np.nan
    if data.ndim in (2, 3):
        data[..., int(data.shape[-2]) * 0.7, :] = fill_value
    return data


def _get_test_swath_def(input_shape, chunk_size, geo_dims):
    from pyresample.geometry import SwathDefinition
    from pyresample.test.utils import create_test_latitude, create_test_longitude
    lon_arr = create_test_longitude(-95.0, -75.0, input_shape, dtype=np.float64)
    lat_arr = create_test_latitude(15.0, 30.0, input_shape, dtype=np.float64)
    lons = da.from_array(lon_arr, chunks=chunk_size)
    lats = da.from_array(lat_arr, chunks=chunk_size)
    swath_def = SwathDefinition(
        xr.DataArray(lons, dims=geo_dims),
        xr.DataArray(lats, dims=geo_dims))
    return swath_def


def _get_test_target_area(output_shape, output_proj=None):
    from pyresample.geometry import AreaDefinition
    if output_proj is None:
        output_proj = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                       '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
    target = AreaDefinition(
        'test_target',
        'test_target',
        'test_target',
        output_proj,
        output_shape[1],  # width
        output_shape[0],  # height
        (-100000., -150000., 100000., 150000.),
    )
    return target


def get_test_data(input_shape=(100, 50), output_shape=(200, 100), output_proj=None,
                  input_chunks=10,
                  input_dims=('y', 'x'), input_dtype=np.float64):
    """Get common data objects used in testing.

    Returns: tuple with the following elements
        input_data_on_swath: DataArray with dimensions as if it is a swath.
        input_swath: SwathDefinition of the above DataArray
        target_area_def: AreaDefinition to be used as a target for resampling

    """
    data = _get_test_array(input_shape, input_dtype, input_chunks)
    ds1 = xr.DataArray(data,
                       dims=input_dims,
                       attrs={'name': 'test', 'test': 'test'})
    if input_dims and 'bands' in input_dims:
        ds1 = ds1.assign_coords(bands=list('RGBA'[:ds1.sizes['bands']]))

    input_area_shape = tuple(ds1.sizes[dim] for dim in ds1.dims
                             if dim in ['y', 'x'])
    geo_dims = ('y', 'x') if input_dims else None
    swath_def = _get_test_swath_def(input_area_shape, input_chunks, geo_dims)
    ds1.attrs['area'] = swath_def
    crs = CRS.from_string('+proj=latlong +datum=WGS84 +ellps=WGS84')
    ds1 = ds1.assign_coords(crs=crs)

    target_area = _get_test_target_area(output_shape, output_proj)
    return ds1, swath_def, target_area


def _create_second_test_data(swath_data):
    swath_data2 = swath_data.copy(deep=True)
    swath_data2.attrs['test'] = 'test2'
    swath_data2.attrs['name'] = 'test2'
    return swath_data2


def _data_attrs_coords_checks(new_data, output_shape, input_dtype, target_area,
                              test_attr, name_attr):
    assert new_data.shape == output_shape
    assert new_data.dtype == input_dtype
    assert new_data.attrs['test'] == test_attr
    assert new_data.attrs['name'] == name_attr
    assert new_data.attrs['area'] is target_area
    if new_data.ndim == 3:
        assert list(new_data.coords['bands']) == ['R', 'G', 'B']


def _coord_and_crs_checks(new_data, target_area, has_bands=False):
    assert 'y' in new_data.coords
    assert 'x' in new_data.coords
    if has_bands:
        assert 'bands' in new_data.coords
    assert 'crs' in new_data.coords
    assert isinstance(new_data.coords['crs'].item(), CRS)
    assert "Lambert" in new_data.coords['crs'].item().coordinate_operation.method_name
    assert new_data.coords['y'].attrs['units'] == 'meter'
    assert new_data.coords['x'].attrs['units'] == 'meter'
    assert target_area.crs == new_data.coords['crs'].item()
    if has_bands:
        np.testing.assert_equal(new_data.coords['bands'].values,
                                ['R', 'G', 'B'])


def _get_num_chunks(source_swath, resampler_class, rows_per_scan=10):
    if resampler_class is DaskEWAResampler:
        # ignore column-wise chunks because DaskEWA should rechunk to use whole scans
        num_chunks = len(source_swath.lons.chunks[0]) if rows_per_scan == 10 else 1
    else:
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])
    return num_chunks


class TestDaskEWAResampler:
    """Test Dask EWA resampler class."""

    @pytest.mark.parametrize(
        ('resampler_class', 'resampler_mod'),
        [
            (DaskEWAResampler, dask_ewa),
            (LegacyDaskEWAResampler, legacy_dask_ewa),
        ])
    @pytest.mark.parametrize(
        ('input_shape', 'input_dims'),
        [
            ((100, 50), ('y', 'x')),
            ((3, 100, 50), ('bands', 'y', 'x')),
        ]
    )
    @pytest.mark.parametrize('input_dtype', [np.float32, np.float64, np.int8])
    @pytest.mark.parametrize('maximum_weight_mode', [False, True])
    @pytest.mark.parametrize('rows_per_scan', [10, 0, 100])
    def test_xarray_basic_ewa(self, resampler_class, resampler_mod,
                              input_shape, input_dims, input_dtype,
                              maximum_weight_mode, rows_per_scan):
        """Test EWA with basic xarray DataArrays."""
        is_legacy = resampler_class is LegacyDaskEWAResampler
        is_int = np.issubdtype(input_dtype, np.integer)
        if is_legacy and is_int:
            pytest.skip("Legacy dask resampler does not properly support "
                        "integer inputs.")
        if is_legacy and rows_per_scan == 0:
            pytest.skip("Legacy dask resampler does not support rows_per_scan "
                        "of 0.")
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims, input_dtype=input_dtype,
        )
        num_chunks = _get_num_chunks(source_swath, resampler_class, rows_per_scan)

        with mock.patch.object(resampler_mod, 'll2cr', wraps=resampler_mod.ll2cr) as ll2cr, \
                mock.patch.object(source_swath, 'get_lonlats', wraps=source_swath.get_lonlats) as get_lonlats, \
                dask.config.set(scheduler='sync'):
            resampler = resampler_class(source_swath, target_area)
            new_data = resampler.resample(swath_data, rows_per_scan=rows_per_scan,
                                          weight_delta_max=40,
                                          maximum_weight_mode=maximum_weight_mode)
            _data_attrs_coords_checks(new_data, output_shape, input_dtype, target_area,
                                      'test', 'test')
            # make sure we can actually compute everything
            new_data.compute()
            lonlat_calls = get_lonlats.call_count
            ll2cr_calls = ll2cr.call_count

            # resample a different dataset and make sure cache is used
            swath_data2 = _create_second_test_data(swath_data)
            new_data = resampler.resample(swath_data2, rows_per_scan=rows_per_scan,
                                          weight_delta_max=40,
                                          maximum_weight_mode=maximum_weight_mode)
            _data_attrs_coords_checks(new_data, output_shape, input_dtype, target_area,
                                      'test2', 'test2')
            _coord_and_crs_checks(new_data, target_area,
                                  has_bands='bands' in input_dims)
            result = new_data.compute()

            # ll2cr will be called once more because of the computation
            assert ll2cr.call_count == ll2cr_calls + num_chunks
            # but we should already have taken the lonlats from the SwathDefinition
            assert get_lonlats.call_count == lonlat_calls

            # check how many valid pixels we have
            band_mult = 3 if 'bands' in result.dims else 1
            fill_mask = _fill_mask(result.values)
            # without NaNs:
            # exp_valid = 13939 if rows_per_scan == 10 else 14029
            # with NaNs but no fix:
            exp_valid = 13817 if rows_per_scan == 10 else 13913
            assert np.count_nonzero(~fill_mask) == exp_valid * band_mult

    @pytest.mark.parametrize(
        ('input_chunks', 'input_shape', 'input_dims'),
        [
            (10, (100, 50), ('y', 'x')),
            ((100, 50), (100, 50), ('y', 'x')),
            (10, (3, 100, 50), ('bands', 'y', 'x')),
        ]
    )
    @pytest.mark.parametrize('input_dtype', [np.float32, np.float64, np.int8])
    @pytest.mark.parametrize('maximum_weight_mode', [False, True])
    def test_xarray_ewa_empty(self, input_chunks, input_shape, input_dims,
                              input_dtype, maximum_weight_mode):
        """Test EWA with xarray DataArrays where the result is all fills."""
        # projection that should result in no output pixels
        output_proj = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                       '+lon_0=-55. +lat_0=25 +lat_1=25 +units=m +no_defs')
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
        # different chunk sizes produces different behaviors for dask reduction
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_chunks=input_chunks,
            input_dims=input_dims, input_dtype=input_dtype,
            output_proj=output_proj
        )

        resampler = DaskEWAResampler(source_swath, target_area)
        new_data = resampler.resample(swath_data, rows_per_scan=10,
                                      maximum_weight_mode=maximum_weight_mode)
        _data_attrs_coords_checks(new_data, output_shape, input_dtype, target_area,
                                  'test', 'test')
        # make sure we can actually compute everything
        computed_data = new_data.compute()
        fill_value = 127 if np.issubdtype(input_dtype, np.integer) else np.nan
        np.testing.assert_array_equal(computed_data, fill_value)

    @pytest.mark.parametrize(
        ('input_shape', 'input_dims', 'maximum_weight_mode'),
        [
            ((100, 50), ('y', 'x'), False),
            # ((3, 100, 50), ('bands', 'y', 'x'), False),
            ((100, 50), ('y', 'x'), True),
            # ((3, 100, 50), ('bands', 'y', 'x'), True),
        ]
    )
    def test_numpy_basic_ewa(self, input_shape, input_dims, maximum_weight_mode):
        """Test EWA with basic xarray DataArrays."""
        from pyresample.geometry import SwathDefinition
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims,
        )
        swath_data = swath_data.data.astype(np.float32).compute()
        source_swath = SwathDefinition(*source_swath.get_lonlats())

        resampler = DaskEWAResampler(source_swath, target_area)
        new_data = resampler.resample(swath_data, rows_per_scan=10,
                                      weight_delta_max=40,
                                      maximum_weight_mode=maximum_weight_mode)
        assert new_data.shape == output_shape
        assert new_data.dtype == np.float32
        assert isinstance(new_data, np.ndarray)

        # check how many valid pixels we have
        band_mult = 3 if len(output_shape) == 3 else 1
        assert np.count_nonzero(~np.isnan(new_data)) == 13817 * band_mult

    @pytest.mark.parametrize(
        ('input_shape', 'input_dims', 'maximum_weight_mode'),
        [
            ((100, 50), ('y', 'x'), False),
            ((3, 100, 50), ('bands', 'y', 'x'), False),
            ((100, 50), ('y', 'x'), True),
            ((3, 100, 50), ('bands', 'y', 'x'), True),
        ]
    )
    def test_compare_to_legacy(self, input_shape, input_dims, maximum_weight_mode):
        """Make sure new and legacy EWA algorithms produce the same results."""
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims,
        )
        swath_data.data = swath_data.data.astype(np.float32)

        resampler = DaskEWAResampler(source_swath, target_area)
        new_data = resampler.resample(swath_data, rows_per_scan=10,
                                      maximum_weight_mode=maximum_weight_mode)
        new_arr = new_data.compute()

        legacy_resampler = LegacyDaskEWAResampler(source_swath, target_area)
        legacy_data = legacy_resampler.resample(swath_data, rows_per_scan=10,
                                                maximum_weight_mode=maximum_weight_mode)
        legacy_arr = legacy_data.compute()

        np.testing.assert_allclose(new_arr, legacy_arr, atol=1e-6)

    @pytest.mark.parametrize(
        ('input_shape', 'input_dims', 'as_np'),
        [
            ((100,), ('y',), False),
            ((4, 100, 50, 25), ('bands', 'y', 'x', 'time'), False),
            ((100,), ('y',), True),
            ((4, 100, 50, 25), ('bands', 'y', 'x', 'time'), True),
        ]
    )
    def test_bad_input(self, input_shape, input_dims, as_np):
        """Check that 1D array inputs are not currently supported."""
        output_shape = (200, 100)
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape,
            input_dims=input_dims,
        )
        swath_data.data = swath_data.data.astype(np.float32)

        resampler = DaskEWAResampler(source_swath, target_area)

        exp_exc = ValueError if len(input_shape) != 4 else NotImplementedError
        with pytest.raises(exp_exc):
            resampler.resample(swath_data, rows_per_scan=10)

    def test_multiple_targets(self):
        """Test that multiple targets produce unique results."""
        input_shape = (100, 50)
        output_shape = (200, 100)
        swath_data, source_swath, target_area1 = get_test_data(
            input_shape=input_shape, output_shape=output_shape,
        )
        target_area2 = _get_test_target_area((250, 150))

        resampler1 = DaskEWAResampler(source_swath, target_area1)
        res1 = resampler1.resample(swath_data, rows_per_scan=10)
        resampler2 = DaskEWAResampler(source_swath, target_area2)
        res2 = resampler2.resample(swath_data, rows_per_scan=10)

        assert res1.name != res2.name
        assert res1.compute().shape != res2.compute().shape
