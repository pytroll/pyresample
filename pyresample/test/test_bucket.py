#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Pyresample developers
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
"""Test the bucket resampler."""

from unittest.mock import MagicMock, patch

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from pyresample import bucket, create_area_def
from pyresample.bucket import _get_invalid_mask
from pyresample.geometry import AreaDefinition
from pyresample.test.utils import CustomScheduler

CHUNKS = 2
WIDTH = 2560
HEIGHT = 2048


@pytest.fixture(scope="module")
def adef():
    """Get AreaDefinition for tests."""
    return AreaDefinition('eurol',
                          'description',
                          '',
                          {'ellps': 'WGS84',
                           'lat_0': '90.0',
                           'lat_ts': '60.0',
                           'lon_0': '0.0',
                           'proj': 'stere'},
                          2560,
                          2048,
                          (-3780000.0, -7644000.0, 3900000.0, -1500000.0))


@pytest.fixture(scope="module")
def lons():
    """Get longitudes for tests."""
    return da.from_array(np.array([[25., 25.], [25., 25.]]), chunks=CHUNKS)


@pytest.fixture(scope="module")
def lats():
    """Get latitudes for tests."""
    return da.from_array(np.array([[60., 60.00001], [60.2, 60.3]]), chunks=CHUNKS)


@pytest.fixture(scope="module")
def resampler(adef, lons, lats):
    """Get initialised resampler for tests."""
    return bucket.BucketResampler(adef, lons, lats)


@patch('pyresample.bucket.Proj')
@patch('pyresample.bucket.BucketResampler._get_indices')
def test_init(get_indices, prj, adef, lons, lats):
    """Test the init method of the BucketResampler."""
    resampler = bucket.BucketResampler(adef, lons, lats)

    get_indices.assert_called_once()
    prj.assert_called_once_with(adef.proj_dict)

    assert hasattr(resampler, 'target_area')
    assert hasattr(resampler, 'source_lons')
    assert hasattr(resampler, 'source_lats')
    assert hasattr(resampler, 'x_idxs')
    assert hasattr(resampler, 'y_idxs')
    assert hasattr(resampler, 'idxs')
    assert hasattr(resampler, 'get_sum')
    assert hasattr(resampler, 'get_count')
    assert hasattr(resampler, 'get_min')
    assert hasattr(resampler, 'get_max')
    assert hasattr(resampler, 'get_abs_max')
    assert hasattr(resampler, 'get_average')
    assert hasattr(resampler, 'get_fractions')
    assert resampler.counts is None


def test_round_to_resolution():
    """Test rounding to given resolution."""
    # Scalar, integer resolution
    assert bucket.round_to_resolution(5.5, 2.) == 6
    # Scalar, non-integer resolution
    assert bucket.round_to_resolution(5.5, 1.7) == 5.1
    # List
    assert np.all(bucket.round_to_resolution([4.2, 5.6], 2) == np.array([4., 6.]))
    # Numpy array
    assert np.all(bucket.round_to_resolution(np.array([4.2, 5.6]), 2) == np.array([4., 6.]))
    # Dask array
    assert np.all(bucket.round_to_resolution(da.array([4.2, 5.6]), 2) == np.array([4., 6.]))


def test_get_proj_coordinates(adef, lons, lats):
    """Test calculation of projection coordinates."""
    resampler = bucket.BucketResampler(source_lats=lats, source_lons=lons, target_area=adef)
    prj = MagicMock()
    prj.return_value = ([3.1, 3.1, 3.1], [4.8, 4.8, 4.8])
    lons = [1., 1., 1.]
    lats = [2., 2., 2.]
    resampler.prj = prj

    result = resampler._get_proj_coordinates(lons, lats)

    prj.assert_called_once_with(lons, lats)
    assert isinstance(result, np.ndarray)
    np.testing.assert_equal(result, np.array([[3.1, 3.1, 3.1],
                                              [4.8, 4.8, 4.8]]))


def test_get_bucket_indices(resampler):
    """Test calculation of array indices."""
    # Ensure nothing is calculated
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        resampler._get_indices()
    x_idxs, y_idxs = da.compute(resampler.x_idxs, resampler.y_idxs)
    np.testing.assert_equal(x_idxs, np.array([1710, 1710, 1707, 1705]))
    np.testing.assert_equal(y_idxs, np.array([465, 465, 459, 455]))


def test_get_bucket_indices_on_latlong():
    """Test calculation of array indices on latlong grid."""
    adef = create_area_def(
        area_id='test',
        projection={'proj': 'latlong'},
        width=2, height=2,
        center=(0, 0),
        resolution=10)
    lons = da.from_array(np.array([-10.0, -9.9, -0.1, 0, 0.1, 9.9, 10.0, -10.1, 0]), chunks=CHUNKS)
    lats = da.from_array(np.array([-10.0, -9.9, -0.1, 0, 0.1, 9.9, 10.0, 0, 10.1]), chunks=CHUNKS)
    resampler = bucket.BucketResampler(source_lats=lats, source_lons=lons, target_area=adef)
    resampler._get_indices()

    np.testing.assert_equal(resampler.x_idxs, np.array([-1, 0, 0, 1, 1, 1, -1, -1, -1]))
    np.testing.assert_equal(resampler.y_idxs, np.array([-1, 1, 1, 1, 0, 0, -1, -1, -1]))


def _get_sum_result(resampler, data, **kwargs):
    """Compute the bucket average with kwargs and check that no dask computation is performed."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_sum(data, **kwargs)
    return result.compute()


def test_get_sum_valid_data(resampler, adef):
    """Test drop-in-a-bucket sum for valid data input."""
    data = da.from_array(np.array([[2., 3.], [7., 16.]]), chunks=CHUNKS)
    result = _get_sum_result(resampler, data)

    # first two values are in same bin
    assert np.count_nonzero(result == 5) == 1
    # others are in separate bins
    assert np.count_nonzero(result == 7) == 1
    assert np.count_nonzero(result == 16) == 1
    assert result.shape == adef.shape

    # Test that also xarray.DataArrays work (same output)
    data = xr.DataArray(data)
    np.testing.assert_array_equal(result, _get_sum_result(resampler, data))


def _equal_or_both_nan(val1, val2):
    return val1 == val2 or (np.isnan(val1) and np.isnan(val2))


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("fill_value", [np.nan, 255, -1])
@pytest.mark.parametrize("empty_bucket_value", [0, 4095, np.nan, -1])
def test_get_sum_skipna_fillvalue_empty_bucket_value(resampler, skipna, fill_value, empty_bucket_value):
    """Test drop-in-a-bucket sum for invalid data input and according arguments."""
    data = da.from_array(np.array([[2., fill_value], [5., fill_value]]), chunks=CHUNKS)
    result = _get_sum_result(resampler, data,
                             skipna=skipna,
                             fill_value=fill_value,
                             empty_bucket_value=empty_bucket_value)
    n_target_bkt = WIDTH * HEIGHT

    # 5 is untouched in a single bin, in any case
    n_bkt_with_val_5 = 1

    if skipna:
        # 2 + fill_value is 2 (nansum)
        n_bkt_with_val_2 = 1
        # and fill_value+fill_value is empty_bucket_value,
        # hence no fill_value bkt are left
        n_bkt_with_val_fill_value = 0
    else:
        # 2 + fill_value is fill_value (sum)
        n_bkt_with_val_2 = 0
        # and fill_value + fill_value is fill_value, so
        n_bkt_with_val_fill_value = 2

    n_bkt_with_empty_value = n_target_bkt - n_bkt_with_val_fill_value - n_bkt_with_val_5 - n_bkt_with_val_2

    # special case
    if _equal_or_both_nan(fill_value, empty_bucket_value):
        # the fill and empty values are equal, so they should be added up
        n_bkt_with_empty_value = n_bkt_with_val_fill_value = n_bkt_with_empty_value + n_bkt_with_val_fill_value

    assert np.count_nonzero(result == 5.) == n_bkt_with_val_5
    assert np.count_nonzero(result == 2.) == n_bkt_with_val_2
    assert np.count_nonzero(_get_invalid_mask(result, fill_value)) == n_bkt_with_val_fill_value
    assert np.count_nonzero(_get_invalid_mask(result, empty_bucket_value)) == n_bkt_with_empty_value


def test_get_count(resampler):
    """Test drop-in-a-bucket sum."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_count()
    result = result.compute()
    assert np.max(result) == 2
    assert np.sum(result == 1) == 2
    assert np.sum(result == 2) == 1
    assert resampler.counts is not None


def _get_min_result(resampler, data, **kwargs):
    """Compute the bucket average with kwargs and check that no dask computation is performed."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_min(data, **kwargs)
    return result.compute()


def test_get_min(resampler):
    """Test min bucket resampling."""
    data = da.from_array(np.array([[2, 11], [5, np.nan]]), chunks=CHUNKS)
    result = _get_min_result(resampler, data)
    # test multiple entries minimum
    assert np.count_nonzero(result == 2) == 1
    # test single entry minimum
    assert np.count_nonzero(result == 5) == 1
    # test that minimum of bucket with only nan is nan, and empty buckets are nan
    assert np.count_nonzero(~np.isnan(result)) == 2


def _get_max_result(resampler, data, **kwargs):
    """Compute the bucket max with kwargs and check that no dask computation is performed."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_max(data, **kwargs)
    return result.compute()


def test_get_max(resampler):
    """Test max bucket resampling."""
    data = da.from_array(np.array([[2, 11], [5, np.nan]]), chunks=CHUNKS)
    result = _get_max_result(resampler, data)
    # test multiple entries maximum
    assert np.count_nonzero(result == 11) == 1
    # test single entry maximum
    assert np.count_nonzero(result == 5) == 1
    # test that minimum of bucket with only nan is nan, and empty buckets are nan
    assert np.count_nonzero(~np.isnan(result)) == 2


def _get_abs_max_result(resampler, data, **kwargs):
    """Compute the bucket abs max with kwargs and check that no dask computation is performed."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_abs_max(data, **kwargs)
    return result.compute()


def test_get_abs_max(resampler):
    """Test abs max bucket resampling."""
    data = da.from_array(np.array([[2, -11], [5, np.nan]]), chunks=CHUNKS)
    result = _get_abs_max_result(resampler, data)
    # test multiple entries absolute maximum
    assert np.count_nonzero(result == -11) == 1
    # test single entry maximum
    assert np.count_nonzero(result == 5) == 1
    # test that minimum of bucket with only nan is nan, and empty buckets are nan
    assert np.count_nonzero(~np.isnan(result)) == 2


def _get_average_result(resampler, data, **kwargs):
    """Compute the bucket average with kwargs and check that no dask computation is performed."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_average(data, **kwargs)
    return result.compute()


def test_get_average_basic(resampler):
    """Test averaging bucket resampling."""
    data = da.from_array(np.array([[2, 11], [5, np.nan]]), chunks=CHUNKS)
    result = _get_average_result(resampler, data)
    # test multiple entries average
    assert np.count_nonzero(result == 6.5) == 1
    # test single entry average
    assert np.count_nonzero(result == 5) == 1
    # test that average of bucket with only nan is nan, and empty buckets are nan
    assert np.count_nonzero(~np.isnan(result)) == 2


def test_get_average_with_fillvalue_for_output(resampler):
    """Test averaging bucket resampling with defined fill_value for output."""
    data = da.from_array(np.array([[2, 11], [5, np.nan]]), chunks=CHUNKS)
    # test fill_value other than np.nan
    result = _get_average_result(resampler, data, fill_value=-1)
    # check that all empty buckets are fill_value
    assert np.count_nonzero(result != -1) == 2


def test_get_average_skipna_true(resampler):
    """Test averaging bucket resampling with skipna True."""
    # test skipna
    data = da.from_array(np.array([[2, np.nan], [np.nan, np.nan]]), chunks=CHUNKS)
    result = _get_average_result(resampler, data, skipna=True)
    # test that average of 2 and np.nan is 2 for skipna=True
    assert np.count_nonzero(result == 2) == 1


def test_get_average_skipna_false(resampler):
    """Test averaging bucket resampling with skipna False."""
    data = da.from_array(np.array([[2, np.nan], [np.nan, np.nan]]), chunks=CHUNKS)
    result = _get_average_result(resampler, data, skipna=False)
    # test that average of 2 and np.nan is nan for skipna=False
    assert np.all(np.isnan(result))


def test_get_average_only_nan_input(resampler):
    """Test averaging bucket resampling with only NaN as input."""
    data = da.from_array(np.array([[np.nan, np.nan], [np.nan, np.nan]]), chunks=CHUNKS)
    result = _get_average_result(resampler, data, skipna=True)
    # test that average of np.nan and np.nan is np.nan for both skipna
    assert np.all(np.isnan(result))
    np.testing.assert_array_equal(result, _get_average_result(resampler, data, skipna=False))


def test_get_average_with_fill_value_in_input(resampler):
    """Test averaging bucket resampling with fill_value in input and skipna True."""
    # test that fill_value in input is recognised as missing value
    data = da.from_array(np.array([[2, -1], [-1, np.nan]]), chunks=CHUNKS)
    result = _get_average_result(resampler, data, fill_value=-1, skipna=True)
    # test that average of 2 and -1 (missing value) is 2
    assert np.count_nonzero(result == 2) == 1
    # test that all other buckets are -1
    assert np.count_nonzero(result != -1) == 1


def test_resample_bucket_fractions(resampler):
    """Test fraction calculations for categorical data."""
    data = da.from_array(np.array([[2, 4], [2, 2]]), chunks=CHUNKS)
    categories = [1, 2, 3, 4]
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_fractions(data, categories=categories)
    assert set(categories) == set(result.keys())

    res = result[1].compute()
    assert np.nanmax(res) == 0.

    res = result[2].compute()
    assert np.nanmax(res) == 1.
    assert np.nanmin(res) == 0.5

    res = result[3].compute()
    assert np.nanmax(res) == 0.

    res = result[4].compute()
    assert np.nanmax(res) == 0.5
    assert np.nanmin(res) == 0.
    # There should be NaN values
    assert np.any(np.isnan(res))

    # Use a fill value
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        result = resampler.get_fractions(data, categories=categories, fill_value=-1)

    # There should not be any NaN values
    for i in categories:
        res = result[i].compute()
        assert not np.any(np.isnan(res))
        assert np.min(res) == -1

    # No categories given, need to compute the data once to get
    # the categories
    with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        _ = resampler.get_fractions(data, categories=None)
