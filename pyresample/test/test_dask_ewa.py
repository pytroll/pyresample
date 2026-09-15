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
from pyresample.test.utils import assert_maximum_dask_computes

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


OUT_CHUNKS_2X2 = ((2, 2), (2, 2))
# with rows_per_scan=10 the default get_test_data swath has 10 row blocks, 3 of which overlap the target area
NUM_NONEMPTY_LL2CR_BLOCKS = 3


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
    @pytest.mark.parametrize('persist', [False, True])
    def test_xarray_basic_ewa(self, resampler_class, resampler_mod,
                              input_shape, input_dims, input_dtype,
                              maximum_weight_mode, rows_per_scan, persist):
        """Test EWA with basic xarray DataArrays."""
        is_legacy = resampler_class is LegacyDaskEWAResampler
        is_int = np.issubdtype(input_dtype, np.integer)
        if is_legacy and is_int:
            pytest.skip("Legacy dask resampler does not properly support "
                        "integer inputs.")
        if is_legacy and rows_per_scan == 0:
            pytest.skip("Legacy dask resampler does not support rows_per_scan "
                        "of 0.")
        if is_legacy and persist:
            pytest.skip("Legacy dask resampler does not support persist.")
        # small output chunks so the persisted path prunes input/output chunk pairs
        extra_kwargs = {} if is_legacy else {"persist": persist, "chunks": (50, 50)}
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
                                          maximum_weight_mode=maximum_weight_mode,
                                          **extra_kwargs)
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
                                          maximum_weight_mode=maximum_weight_mode,
                                          **extra_kwargs)
            _data_attrs_coords_checks(new_data, output_shape, input_dtype, target_area,
                                      'test2', 'test2')
            _coord_and_crs_checks(new_data, target_area,
                                  has_bands='bands' in input_dims)
            result = new_data.compute()

            # ll2cr will be called once more because of the computation
            # unless the persisted ll2cr result is reused
            assert ll2cr.call_count == ll2cr_calls + (0 if persist else num_chunks)
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
    @pytest.mark.parametrize('persist', [False, True])
    def test_xarray_ewa_empty(self, input_chunks, input_shape, input_dims,
                              input_dtype, maximum_weight_mode, persist):
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
                                      maximum_weight_mode=maximum_weight_mode,
                                      persist=persist)
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
    @pytest.mark.parametrize('persist', [False, True])
    def test_numpy_basic_ewa(self, input_shape, input_dims, maximum_weight_mode, persist):
        """Test EWA with basic numpy arrays."""
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
                                      maximum_weight_mode=maximum_weight_mode,
                                      persist=persist, chunks=(50, 50))
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
    @pytest.mark.parametrize('persist', [False, True])
    def test_compare_to_legacy(self, input_shape, input_dims, maximum_weight_mode, persist):
        """Make sure new and legacy EWA algorithms produce the same results.

        The default ``weight_delta_max`` of 10 with 25x25 output chunks on
        a 200x100 target area means that the persisted path prunes
        input/output chunk pairs that cannot overlap.

        """
        output_shape = (200, 100)
        if len(input_shape) == 3:
            output_shape = (input_shape[0], output_shape[0], output_shape[1])
        swath_data, source_swath, target_area = get_test_data(
            input_shape=input_shape, output_shape=output_shape[-2:],
            input_dims=input_dims,
        )
        swath_data.data = swath_data.data.astype(np.float32)
        out_chunks = (25, 25)

        resampler = DaskEWAResampler(source_swath, target_area)
        # mock call counting is not thread safe
        with mock.patch.object(dask_ewa, 'fornav_weights_and_sums_wrapper',
                               wraps=dask_ewa.fornav_weights_and_sums_wrapper) as fornav_mock, \
                dask.config.set(scheduler='sync'):
            new_data = resampler.resample(swath_data, rows_per_scan=10,
                                          maximum_weight_mode=maximum_weight_mode,
                                          persist=persist, chunks=out_chunks)
            new_arr = new_data.compute()
        num_bands = 1 if len(input_shape) == 2 else input_shape[0]
        num_out_chunks = (output_shape[-2] // out_chunks[0]) * (output_shape[-1] // out_chunks[1])
        max_fornav_calls = num_bands * NUM_NONEMPTY_LL2CR_BLOCKS * num_out_chunks
        assert fornav_mock.call_count > 0
        if persist:
            # some input/output chunk pairs were pruned
            assert fornav_mock.call_count < max_fornav_calls
        else:
            assert fornav_mock.call_count == max_fornav_calls

        legacy_resampler = LegacyDaskEWAResampler(source_swath, target_area)
        legacy_data = legacy_resampler.resample(swath_data, rows_per_scan=10,
                                                maximum_weight_mode=maximum_weight_mode)
        legacy_arr = legacy_data.compute()

        # small output chunks cause float32 rounding differences near chunk
        # boundaries compared to the legacy single-chunk fornav
        np.testing.assert_allclose(new_arr, legacy_arr, atol=1e-4)

    @pytest.mark.parametrize('maximum_weight_mode', [False, True])
    def test_persist_matches_non_persist(self, maximum_weight_mode):
        """Pruning input/output chunk pairs must not change the result."""
        swath_data, source_swath, target_area = get_test_data(
            input_shape=(100, 50), output_shape=(200, 100),
            input_dtype=np.float32,
        )
        results = []
        for persist in (False, True):
            resampler = DaskEWAResampler(source_swath, target_area)
            new_data = resampler.resample(swath_data, rows_per_scan=10,
                                          maximum_weight_mode=maximum_weight_mode,
                                          persist=persist, chunks=(25, 25))
            results.append(new_data.compute().values)
        np.testing.assert_array_equal(results[0], results[1])

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


@pytest.mark.parametrize('persist', [False, True])
def test_precompute_ll2cr_extents(persist):
    """Persisted precompute computes clipped extents for every ll2cr block in one pass, otherwise nothing."""
    swath_data, source_swath, target_area = get_test_data(
        input_shape=(100, 50), output_shape=(200, 100),
        input_dims=('y', 'x'), input_dtype=np.float32,
    )
    margin = 10.0
    resampler = DaskEWAResampler(source_swath, target_area)
    # persisting ll2cr is one computation, the extents of the persisted blocks another
    with assert_maximum_dask_computes(2 if persist else 0):
        resampler.precompute(rows_per_scan=10, persist=persist, weight_delta_max=margin)

    assert resampler.cache['extent_margin'] == margin
    assert isinstance(resampler.cache['ll2cr_result'], da.Array)
    extents = resampler.cache['extents']
    if not persist:
        assert extents is None
        return

    num_row_blocks, num_col_blocks = resampler.cache['ll2cr_result'].numblocks
    assert extents.shape == (num_row_blocks, num_col_blocks, 4)
    empty = np.isnan(extents).all(axis=-1)
    assert np.count_nonzero(~empty) == NUM_NONEMPTY_LL2CR_BLOCKS
    # empty blocks are entirely NaN, non-empty blocks are entirely finite
    np.testing.assert_array_equal(np.isnan(extents).any(axis=-1), empty)
    row_min, row_max, col_min, col_max = np.moveaxis(extents[~empty], -1, 0)
    grid_rows, grid_cols = target_area.shape
    assert (row_min <= row_max).all()
    assert (col_min <= col_max).all()
    assert (-margin <= row_min).all() and (row_max <= grid_rows + margin).all()
    assert (-margin <= col_min).all() and (col_max <= grid_cols + margin).all()


def test_fornav_block_meta_filters_non_overlapping_pairs():
    """Only overlapping input/output chunk pairs are flagged for computation."""
    block_extents = (
        (0.1, 1.8, 0.1, 1.8),
        (0.1, 1.8, 2.1, 3.8),
        (2.1, 3.8, 0.1, 1.8),
        (2.1, 3.8, 2.1, 3.8),
    )
    block_meta = dask_ewa._fornav_block_meta(OUT_CHUNKS_2X2, block_extents, 0.0)
    assert len(block_meta) == 16
    assert block_meta[(0, 0, 0)] == (slice(0, 2), slice(0, 2), True)
    assert block_meta[(3, 1, 1)] == (slice(2, 4), slice(2, 4), True)
    overlapping = {key for key, (_, _, overlaps) in block_meta.items() if overlaps}
    assert overlapping == {(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)}


@pytest.mark.parametrize(
    ("block_extent", "overlap_margin", "expected_count"),
    [
        pytest.param((1.2, 1.9, 1.2, 1.9), 0.0, 1, id="no-padding"),
        pytest.param((1.2, 1.9, 1.2, 1.9), 1.0, 4, id="padding"),
        pytest.param(None, 0.0, 4, id="no-extent"),
    ],
)
def test_fornav_block_meta_overlap_padding(block_extent, overlap_margin, expected_count):
    """Overlap padding expands to neighboring output chunks and an unknown extent overlaps everything."""
    block_meta = dask_ewa._fornav_block_meta(OUT_CHUNKS_2X2, [block_extent], overlap_margin)
    assert len(block_meta) == 4
    assert sum(overlaps for _, _, overlaps in block_meta.values()) == expected_count


def test_row_block_extent():
    """Extents of the column blocks of one input row are merged, ignoring empty blocks."""
    nan = np.nan
    extents = np.array([[nan, nan, nan, nan], [5.0, 20.0, 1.0, 8.0], [2.0, 15.0, 3.0, 12.0]])
    assert dask_ewa._row_block_extent(extents) == (2.0, 20.0, 1.0, 12.0)
    assert dask_ewa._row_block_extent(extents[:1]) is None


def test_ll2cr_block_extent_returns_none_for_all_non_finite():
    ll2cr_block = np.stack(
        [np.full((2, 2), np.nan, dtype=np.float64), np.full((2, 2), np.nan, dtype=np.float64)],
        axis=0,
    )
    assert dask_ewa._ll2cr_block_extent(ll2cr_block, (200, 100), 10.0) is None


def test_ll2cr_block_extent_returns_none_for_empty_sentinel():
    empty = (((2, 2), np.nan, np.float64), ((2, 2), np.nan, np.float64))
    assert dask_ewa._ll2cr_block_extent(empty, (200, 100), 10.0) is None


@pytest.mark.parametrize(
    ("cols", "rows", "margin", "expected"),
    [
        # far-off points on both sides are excluded, -0.5 is inside [-10, 110]
        pytest.param([-5000.0, -0.5, 3.0, 250.0], [10.0, 10.0, 10.0, 10.0], 10.0, (10.0, 10.0, -0.5, 3.0),
                     id="far-off-cols"),
        # all points outside the padded grid
        pytest.param([-50.0, -50.0], [10.0, 10.0], 10.0, None, id="all-outside"),
        # points past the grid edge but within the margin are kept
        pytest.param([105.0, 106.0], [10.0, 20.0], 10.0, (10.0, 20.0, 105.0, 106.0), id="inside-margin"),
        # rows are clipped the same way as columns
        pytest.param([10.0, 10.0, 10.0], [-11.0, 5.0, 210.5], 10.0, (5.0, 5.0, 10.0, 10.0), id="far-off-rows"),
        # NaNs are ignored
        pytest.param([np.nan, 3.0], [10.0, 10.0], 10.0, (10.0, 10.0, 3.0, 3.0), id="nan"),
    ],
)
def test_ll2cr_block_extent_clips_to_padded_grid(cols, rows, margin, expected):
    """Extents only consider points that can reach the target grid."""
    ll2cr_block = np.stack([np.array([cols]), np.array([rows])], axis=0)
    assert dask_ewa._ll2cr_block_extent(ll2cr_block, (200, 100), margin) == expected


def test_average_fornav_empty_keepdims_returns_fill():
    empty = (((2, 2), 0, np.float32), ((2, 2), 0, np.float32))
    out = dask_ewa._average_fornav([empty], axis=(0,), keepdims=True, dtype=np.float32, fill_value=np.nan)
    assert out.shape == (2, 2)
    assert np.all(np.isnan(out))


def test_persisted_ll2cr_blocks_are_reused_between_resample_calls():
    """Persisted ll2cr blocks should not be recomputed for subsequent calls."""
    swath_data, source_swath, target_area = get_test_data(
        input_shape=(100, 50), output_shape=(200, 100),
        input_dims=('y', 'x'), input_dtype=np.float32,
    )

    with mock.patch.object(dask_ewa, 'll2cr', wraps=dask_ewa.ll2cr) as ll2cr_mock, \
            dask.config.set(scheduler='sync'):
        resampler = DaskEWAResampler(source_swath, target_area)

        out1 = resampler.resample(
            swath_data,
            rows_per_scan=10,
            persist=True,
            chunks=(50, 50),
            weight_delta_max=40,
        )
        out1.compute()
        calls_after_first = ll2cr_mock.call_count

        out2 = resampler.resample(
            swath_data,
            rows_per_scan=10,
            persist=True,
            chunks=(50, 50),
            weight_delta_max=40,
        )
        out2.compute()
        calls_after_second = ll2cr_mock.call_count

    assert calls_after_first > 0
    assert calls_after_second == calls_after_first


@pytest.mark.parametrize(
    ("first_kwargs", "second_kwargs"),
    [
        pytest.param(
            {"rows_per_scan": 10, "persist": False},
            {"rows_per_scan": 100, "persist": False},
            id="rows-per-scan-change",
        ),
        pytest.param(
            {"rows_per_scan": 10, "persist": False},
            {"rows_per_scan": 10, "persist": True},
            id="persist-change",
        ),
        pytest.param(
            {"rows_per_scan": 10, "persist": True, "weight_delta_max": 10},
            {"rows_per_scan": 10, "persist": True, "weight_delta_max": 40},
            id="margin-change",
        ),
    ],
)
def test_ll2cr_cache_recomputes_when_precompute_mode_changes(first_kwargs, second_kwargs):
    """Changing precompute mode should invalidate the cached ll2cr block layout."""
    swath_data, source_swath, target_area = get_test_data(
        input_shape=(100, 50), output_shape=(200, 100),
        input_dims=('y', 'x'), input_dtype=np.float32,
    )

    with mock.patch.object(source_swath, 'get_lonlats', wraps=source_swath.get_lonlats) as get_lonlats_mock, \
            dask.config.set(scheduler='sync'):
        resampler = DaskEWAResampler(source_swath, target_area)

        resampler.resample(
            swath_data,
            chunks=(50, 50),
            **first_kwargs,
        ).compute()
        calls_after_first = get_lonlats_mock.call_count

        resampler.resample(
            swath_data,
            chunks=(50, 50),
            **second_kwargs,
        ).compute()
        calls_after_second = get_lonlats_mock.call_count

    assert calls_after_first > 0
    assert calls_after_second > calls_after_first
