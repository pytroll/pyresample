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
import numpy as np
from unittest import mock
import pytest

try:
    from pyproj import CRS
except ImportError:
    CRS = None

LOG = logging.getLogger(__name__)


def _fill_mask(data):
    if np.issubdtype(data.dtype, np.floating):
        return np.isnan(data)
    elif np.issubdtype(data.dtype, np.integer):
        return data == np.iinfo(data.dtype).min
    else:
        raise ValueError("Not sure how to get fill mask.")


def get_test_data(input_shape=(100, 50), output_shape=(200, 100), output_proj=None,
                  input_dims=('y', 'x'), input_dtype=np.float64):
    """Get common data objects used in testing.

    Returns: tuple with the following elements
        input_data_on_swath: DataArray with dimensions as if it is a swath.
        input_swath: SwathDefinition of the above DataArray
        target_area_def: AreaDefinition to be used as a target for resampling

    """
    from xarray import DataArray
    import dask.array as da
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from pyresample.utils import proj4_str_to_dict
    from pyresample.test.utils import create_test_longitude, create_test_latitude
    if np.issubdtype(input_dtype, np.integer):
        dinfo = np.iinfo(input_dtype)
        data = da.random.randint(dinfo.min + 1, dinfo.max, size=input_shape,
                                 chunks=85, dtype=input_dtype)
    else:
        data = da.random.random(input_shape, chunks=85).astype(input_dtype)
    ds1 = DataArray(data,
                    dims=input_dims,
                    attrs={'name': 'test_data_name', 'test': 'test'})
    if input_dims and 'bands' in input_dims:
        ds1 = ds1.assign_coords(bands=list('RGBA'[:ds1.sizes['bands']]))

    input_area_shape = tuple(ds1.sizes[dim] for dim in ds1.dims
                             if dim in ['y', 'x'])
    geo_dims = ('y', 'x') if input_dims else None
    lon_arr = create_test_longitude(-95.0, -75.0, input_area_shape, dtype=np.float64)
    lat_arr = create_test_latitude(15.0, 30.0, input_area_shape, dtype=np.float64)
    lons = da.from_array(lon_arr, chunks=50)
    lats = da.from_array(lat_arr, chunks=50)
    swath_def = SwathDefinition(
        DataArray(lons, dims=geo_dims),
        DataArray(lats, dims=geo_dims))
    ds1.attrs['area'] = swath_def
    if CRS is not None:
        crs = CRS.from_string('+proj=latlong +datum=WGS84 +ellps=WGS84')
        ds1 = ds1.assign_coords(crs=crs)

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
        (-100000., -150000., 100000., 150000.),
    )
    return ds1, swath_def, target


def _coord_and_crs_checks(new_data, target_area, has_bands=False):
    assert 'y' in new_data.coords
    assert 'x' in new_data.coords
    if has_bands:
        assert 'bands' in new_data.coords
    if CRS is not None:
        assert 'crs' in new_data.coords
        assert isinstance(new_data.coords['crs'].item(), CRS)
        assert 'lcc' in new_data.coords['crs'].item().to_proj4()
        assert new_data.coords['y'].attrs['units'] == 'meter'
        assert new_data.coords['x'].attrs['units'] == 'meter'
        if hasattr(target_area, 'crs'):
            assert target_area.crs is new_data.coords['crs'].item()
        if has_bands:
            np.testing.assert_equal(new_data.coords['bands'].values,
                                    ['R', 'G', 'B'])


class TestLegacyDaskEWAResampler:
    """Test Legacy Dask EWA resampler class."""

    @pytest.mark.parametrize(
        ('input_shape', 'input_dims', 'input_dtype'),
        [
            ((100, 50), ('y', 'x'), np.float32),
            ((3, 100, 50), ('bands', 'y', 'x'), np.float32),
            ((100, 50), ('y', 'x'), np.float64),
            ((3, 100, 50), ('bands', 'y', 'x'), np.float64),
        ]
    )
    def test_basic_ewa(self, input_shape, input_dims, input_dtype):
        """Test EWA with basic xarray DataArrays."""
        import xarray as xr
        from pyresample.ewa import LegacyDaskEWAResampler, _legacy_dask_ewa
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_coords = {'bands': ['R', 'G', 'B']}
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
            output_dims = ('bands', 'y', 'x')
        else:
            output_coords = {}
            output_dims = ('y', 'x')
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims, input_dtype=input_dtype,
        )
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        with mock.patch.object(_legacy_dask_ewa, 'll2cr', wraps=_legacy_dask_ewa.ll2cr) as ll2cr, \
                mock.patch.object(source_swath, 'get_lonlats', wraps=source_swath.get_lonlats) as get_lonlats:
            resampler = LegacyDaskEWAResampler(source_swath, target_area)
            new_data = resampler.resample(swath_data, rows_per_scan=10)
            assert new_data.shape == output_shape
            assert new_data.dtype == input_dtype
            assert new_data.attrs['test'] == 'test'
            assert new_data.attrs['area'] is target_area
            # make sure we can actually compute everything
            new_data.compute()
            lonlat_calls = get_lonlats.call_count
            ll2cr_calls = ll2cr.call_count

            # resample a different dataset and make sure cache is used
            data = xr.DataArray(
                swath_data.data,
                coords=output_coords,
                dims=output_dims, attrs={'area': source_swath, 'test': 'test2',
                                         'name': 'test2'})
            new_data = resampler.resample(data, rows_per_scan=10)
            new_data.compute()
            # ll2cr will be called once more because of the computation
            assert ll2cr.call_count == ll2cr_calls + num_chunks
            # but we should already have taken the lonlats from the SwathDefinition
            assert get_lonlats.call_count == lonlat_calls
            _coord_and_crs_checks(new_data, target_area,
                                  has_bands='bands' in input_dims)


class TestDaskEWAResampler:
    """Test Dask EWA resampler class."""

    @pytest.mark.parametrize(
        ('input_shape', 'input_dims', 'input_dtype', 'maximum_weight_mode'),
        [
            ((100, 50), ('y', 'x'), np.float32, False),
            ((3, 100, 50), ('bands', 'y', 'x'), np.float32, False),
            ((100, 50), ('y', 'x'), np.float32, True),
            ((3, 100, 50), ('bands', 'y', 'x'), np.float32, True),
            ((100, 50), ('y', 'x'), np.float64, False),
            ((3, 100, 50), ('bands', 'y', 'x'), np.float64, False),
            ((100, 50), ('y', 'x'), np.float64, True),
            ((3, 100, 50), ('bands', 'y', 'x'), np.float64, True),
        ]
    )
    def test_xarray_basic_ewa(self, input_shape, input_dims, input_dtype,
                              maximum_weight_mode):
        """Test EWA with basic xarray DataArrays."""
        import numpy as np
        import xarray as xr
        from pyresample.ewa import DaskEWAResampler, dask_ewa
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_coords = {'bands': ['R', 'G', 'B']}
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
            output_dims = ('bands', 'y', 'x')
        else:
            output_coords = {}
            output_dims = ('y', 'x')
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims, input_dtype=input_dtype,
        )
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        with mock.patch.object(dask_ewa, 'll2cr', wraps=dask_ewa.ll2cr) as ll2cr, \
                mock.patch.object(source_swath, 'get_lonlats', wraps=source_swath.get_lonlats) as get_lonlats:
            resampler = DaskEWAResampler(source_swath, target_area)
            new_data = resampler.resample(swath_data, rows_per_scan=10,
                                          maximum_weight_mode=maximum_weight_mode)
            assert new_data.shape == output_shape
            assert new_data.dtype == input_dtype
            assert new_data.attrs['test'] == 'test'
            assert new_data.attrs['area'] is target_area
            # make sure we can actually compute everything
            new_data.compute()
            lonlat_calls = get_lonlats.call_count
            ll2cr_calls = ll2cr.call_count

            # resample a different dataset and make sure cache is used
            data = xr.DataArray(
                swath_data.data,
                coords=output_coords,
                dims=output_dims, attrs={'area': source_swath, 'test': 'test2',
                                         'name': 'test2'})
            new_data = resampler.resample(data, rows_per_scan=10,
                                          maximum_weight_mode=maximum_weight_mode)
            result = new_data.compute()
            # ll2cr will be called once more because of the computation
            assert ll2cr.call_count == ll2cr_calls + num_chunks
            # but we should already have taken the lonlats from the SwathDefinition
            assert get_lonlats.call_count == lonlat_calls
            # check how many valid pixels we have
            band_mult = 3 if 'bands' in output_dims else 1
            assert np.count_nonzero(~np.isnan(result.values)) == 468 * band_mult
            _coord_and_crs_checks(new_data, target_area,
                                  has_bands='bands' in input_dims)

    @pytest.mark.parametrize(
        ('input_shape', 'input_dims', 'input_dtype'),
        [
            ((100, 50), ('y', 'x'), np.int8),
        ]
    )
    def test_xarray_basic_ewa_int(self, input_shape, input_dims, input_dtype):
        """Test EWA with basic xarray DataArrays of integer type."""
        import numpy as np
        import xarray as xr
        from pyresample.ewa import DaskEWAResampler, dask_ewa
        output_shape = (200, 100)
        output_coords = {}
        output_dims = ('y', 'x')
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims, input_dtype=input_dtype,
        )
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        with mock.patch.object(dask_ewa, 'll2cr', wraps=dask_ewa.ll2cr) as ll2cr, \
                mock.patch.object(source_swath, 'get_lonlats', wraps=source_swath.get_lonlats) as get_lonlats:
            resampler = DaskEWAResampler(source_swath, target_area)
            new_data = resampler.resample(swath_data, rows_per_scan=10)
            assert new_data.shape == output_shape
            assert new_data.dtype == input_dtype
            assert new_data.attrs['test'] == 'test'
            assert new_data.attrs['area'] is target_area
            # make sure we can actually compute everything
            new_data.compute()
            lonlat_calls = get_lonlats.call_count
            ll2cr_calls = ll2cr.call_count

            # resample a different dataset and make sure cache is used
            data = xr.DataArray(
                swath_data.data,
                coords=output_coords,
                dims=output_dims, attrs={'area': source_swath, 'test': 'test2',
                                         'name': 'test2'})
            new_data = resampler.resample(data, rows_per_scan=10)
            result = new_data.compute()
            # ll2cr will be called once more because of the computation
            assert ll2cr.call_count == ll2cr_calls + num_chunks
            # but we should already have taken the lonlats from the SwathDefinition
            assert get_lonlats.call_count == lonlat_calls
            # check how many valid pixels we have
            band_mult = 3 if 'bands' in output_dims else 1
            fill_mask = _fill_mask(result.values)
            assert np.count_nonzero(~fill_mask) == 468 * band_mult
            _coord_and_crs_checks(new_data, target_area,
                                  has_bands='bands' in input_dims)

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
        import numpy as np
        from pyresample.ewa import DaskEWAResampler
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
                                      maximum_weight_mode=maximum_weight_mode)
        assert new_data.shape == output_shape
        assert new_data.dtype == np.float32
        assert isinstance(new_data, np.ndarray)

        # check how many valid pixels we have
        band_mult = 3 if len(output_shape) == 3 else 1
        assert np.count_nonzero(~np.isnan(new_data)) == 468 * band_mult

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
        import numpy as np
        from pyresample.ewa import DaskEWAResampler, LegacyDaskEWAResampler
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

        np.testing.assert_allclose(new_arr, legacy_arr)

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
        import numpy as np
        from pyresample.ewa import DaskEWAResampler
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

