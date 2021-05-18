#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""Tests for the gradien search resampling."""

import unittest
from unittest import mock
from pyresample.geometry import AreaDefinition, SwathDefinition
import pyresample
import numpy as np
import dask.array as da
import xarray as xr
import pytest


class TestGradientResamplerChunked(unittest.TestCase):
    """Test case for the gradient resampling."""

    def setUp(self):
        """Set up the test case."""
        import pyresample.gradient
        from pyresample.gradient import GradientSearchResampler
        self.src_area = AreaDefinition('dst', 'dst area', None,
                                       {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                       100, 100,
                                       (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

        self.resampler = GradientSearchResampler(self.src_area, self.dst_area)

        swath_area = AreaDefinition(
            'omerc_otf',
            'On-the-fly omerc area',
            None,
            {'alpha': '8.99811271718795',
             'ellps': 'sphere',
             'gamma': '0',
             'k': '1',
             'lat_0': '0',
             'lonc': '13.8096029486222',
             'proj': 'omerc',
             'units': 'm'},
            50, 100,
            (-1461111.3603, 3440088.0459, 1534864.0322, 9598335.0457)
        )

        lons, lats = swath_area.get_lonlats(chunks=30)
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        self.swath = SwathDefinition(lons, lats)

        self.swath_resampler = GradientSearchResampler(self.swath,
                                                       self.dst_area)
        self.area2swath_resampler = GradientSearchResampler(self.src_area, self.swath)
        self.original_chunk_size = pyresample.CHUNK_SIZE
        pyresample.gradient.CHUNK_SIZE = 30

    def tearDown(self):
        """Tear down the test case."""
        pyresample.gradient.CHUNK_SIZE = self.original_chunk_size

    def test_resample_area_to_area_2d_with_chunks(self):
        """Resample area to area, 2d."""
        data = xr.DataArray(da.ones(self.src_area.shape, dtype=np.float64, chunks=40),
                            dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        np.testing.assert_allclose(res, 1)

    def test_resample_area_to_area_2d_fill_value(self):
        """Resample area to area, 2d, use fill value."""
        data = xr.DataArray(da.full(self.src_area.shape, np.nan, dtype=np.float64, chunks=40),
                            dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bilinear',
            fill_value=2.0).compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        np.testing.assert_allclose(res, 2.0)

    def test_resample_area_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.src_area.shape, dtype=np.float64, chunks=40) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        res = self.resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        np.testing.assert_allclose(res[0, :, :], 1.0)
        np.testing.assert_allclose(res[1, :, :], 2.0)
        np.testing.assert_allclose(res[2, :, :], 3.0)

    def test_resample_swath_to_area_2d(self):
        """Resample swath to area, 2d."""
        data = xr.DataArray(da.ones(self.swath.shape, dtype=np.float64, chunks=40),
                            dims=['y', 'x'])
        res = self.swath_resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert not np.all(np.isnan(res))

    def test_resample_swath_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.swath.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        res = self.swath_resampler.compute(
            data, method='bilinear').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        for i in range(res.shape[0]):
            arr = np.ravel(res[i, :, :])
            assert np.allclose(arr[np.isfinite(arr)], float(i + 1))

    def test_resampler_only_works_on_dataarrays_for_3d(self):
        """Test that the resampler only works on dataarrays for the 3d case."""
        data = da.ones(self.src_area.shape + (1,), dtype=np.float64, chunks=40)
        with pytest.raises(TypeError):
            self.resampler.compute(data, method='bilinear').compute(scheduler='single-threaded')

    def test_resampler_works_on_2d_dask_arrays(self):
        """Test that the resampler works on 2d dask arrays."""
        data = da.ones(self.src_area.shape, dtype=np.float64, chunks=40)
        res = self.resampler.compute(data, method='bilinear').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert not np.all(np.isnan(res))

    # def test_resample_area_to_swath_2d_with_chunks(self):
    #     """Resample area to swath, 2d."""
    #     data = xr.DataArray(da.ones(self.src_area.shape, dtype=np.float64, chunks=40),
    #                         dims=['y', 'x'])
    #     res = self.area2swath_resampler.compute(
    #         data, method='bilinear').compute(scheduler='single-threaded')
    #     assert res.shape == self.dst_area.shape
    #     np.testing.assert_allclose(res, 1)


class TestEnsureDataArray(unittest.TestCase):
    """Test the ensure_data_array decorator."""

    def test_decorator_converts_2d_array_to_dataarrays_if_needed(self):
        """Test that the decorator converts numpy or dask 2d arrays to dataarrays."""
        from pyresample.gradient import ensure_chunked_data_array
        data = da.ones((10, 10), dtype=np.float64, chunks=40)

        def fake_compute(arg1, data):
            assert isinstance(data, xr.DataArray)

        decorated = ensure_chunked_data_array(fake_compute)
        decorated('bla', data)

    def test_decorator_rechunks_other_dimensions_to_one_chunk(self):
        """Test that the decorator rechunks dimensions other than x and y to one chunk."""
        from pyresample.gradient import ensure_chunked_data_array
        data = xr.DataArray(da.ones((10, 10, 10), dtype=np.float64, chunks=2), dims=["band", "y", "x"])

        def fake_compute(arg1, data):
            assert data.data.chunksize[0] == 10

        decorated = ensure_chunked_data_array(fake_compute)
        decorated('bla', data)


def test_check_overlap():
    """Test overlap check returning correct results."""
    from shapely.geometry import Polygon
    from pyresample.gradient import check_overlap

    # If either of the polygons is False, True is returned
    assert check_overlap(False, 3) is True
    assert check_overlap('eggs', False) is True
    assert check_overlap(False, False) is True

    # If either the polygons is None, False is returned
    assert check_overlap(None, 'bacon') is False
    assert check_overlap('spam', None) is False
    assert check_overlap(None, None) is False

    # If the polygons overlap, True is returned
    poly1 = Polygon(((0, 0), (0, 1), (1, 1), (1, 0)))
    poly2 = Polygon(((-1, -1), (-1, 1), (1, 1), (1, -1)))
    assert check_overlap(poly1, poly2) is True

    # If the polygons do not overlap, False is returned
    poly2 = Polygon(((5, 5), (6, 5), (6, 6), (5, 6)))
    assert check_overlap(poly1, poly2) is False


@mock.patch('pyresample.gradient.get_geostationary_bounding_box')
def test_get_border_lonlats(get_geostationary_bounding_box):
    """Test that correct methods are called in get_border_lonlats()."""
    from pyresample.gradient import get_border_lonlats
    geo_def = mock.MagicMock(proj_dict={'proj': 'geos'})
    get_geostationary_bounding_box.return_value = 1, 2
    res = get_border_lonlats(geo_def)
    assert res == (1, 2)
    get_geostationary_bounding_box.assert_called_with(geo_def, 3600)
    geo_def.get_boundary_lonlats.assert_not_called()

    lon_sides = mock.MagicMock(side1=np.array([1]), side2=np.array([2]),
                               side3=np.array([3]), side4=np.array([4]))
    lat_sides = mock.MagicMock(side1=np.array([1]), side2=np.array([2]),
                               side3=np.array([3]), side4=np.array([4]))
    geo_def = mock.MagicMock()
    geo_def.get_boundary_lonlats.return_value = lon_sides, lat_sides
    lon_b, lat_b = get_border_lonlats(geo_def)
    assert np.all(lon_b == np.array([1, 2, 3, 4]))
    assert np.all(lat_b == np.array([1, 2, 3, 4]))


@mock.patch('pyresample.gradient.Polygon')
@mock.patch('pyresample.gradient.get_border_lonlats')
def test_get_polygon(get_border_lonlats, Polygon):
    """Test polygon creation."""
    from pyresample.gradient import get_polygon

    # Valid polygon
    get_border_lonlats.return_value = (1, 2)
    geo_def = mock.MagicMock()
    prj = mock.MagicMock()
    x_borders = [0, 0, 1, 1]
    y_borders = [0, 1, 1, 0]
    boundary = [(0, 0), (0, 1), (1, 1), (1, 0)]
    prj.return_value = (x_borders, y_borders)
    poly = mock.MagicMock(area=2.0)
    Polygon.return_value = poly
    res = get_polygon(prj, geo_def)
    get_border_lonlats.assert_called_with(geo_def)
    prj.assert_called_with(1, 2)
    Polygon.assert_called_with(boundary)
    assert res is poly

    # Some border points are invalid, those should have been removed
    x_borders = [np.inf, 0, 0, 0, 1, np.nan, 2]
    y_borders = [-1, 0, np.nan, 1, 1, np.nan, -1]
    boundary = [(0, 0), (0, 1), (1, 1), (2, -1)]
    prj.return_value = (x_borders, y_borders)
    res = get_polygon(prj, geo_def)
    Polygon.assert_called_with(boundary)
    assert res is poly

    # Polygon area is NaN
    poly.area = np.nan
    res = get_polygon(prj, geo_def)
    assert res is None

    # Polygon area is 0.0
    poly.area = 0.0
    res = get_polygon(prj, geo_def)
    assert res is None


@mock.patch('pyresample.gradient.one_step_gradient_search')
def test_gradient_resample_data(one_step_gradient_search):
    """Test that one_step_gradient_search() is called with proper array shapes."""
    from pyresample.gradient import _gradient_resample_data

    ndim_3 = np.zeros((3, 3, 4))
    ndim_2a = np.zeros((3, 4))
    ndim_2b = np.zeros((8, 10))

    # One of the source arrays has wrong shape
    try:
        _ = _gradient_resample_data(ndim_3, ndim_2a, ndim_2b, ndim_2a, ndim_2a,
                                    ndim_2a, ndim_2a, ndim_2b, ndim_2b)
        raise IndexError
    except AssertionError:
        pass
    one_step_gradient_search.assert_not_called()

    # Data array has wrong shape
    try:
        _ = _gradient_resample_data(ndim_2a, ndim_2a, ndim_2a, ndim_2a, ndim_2a,
                                    ndim_2a, ndim_2a, ndim_2b, ndim_2b)
        raise IndexError
    except AssertionError:
        pass
    one_step_gradient_search.assert_not_called()

    # The destination x and y arrays have different shapes
    try:
        _ = _gradient_resample_data(ndim_3, ndim_2a, ndim_2a, ndim_2a, ndim_2a,
                                    ndim_2a, ndim_2a, ndim_2b, ndim_2a)
        raise IndexError
    except AssertionError:
        pass
    one_step_gradient_search.assert_not_called()

    # Correct shapes are given
    _ = _gradient_resample_data(ndim_3, ndim_2a, ndim_2a, ndim_2a, ndim_2a,
                                ndim_2a, ndim_2a, ndim_2b, ndim_2b)
    one_step_gradient_search.assert_called_once()


@mock.patch('pyresample.gradient.dask.delayed')
@mock.patch('pyresample.gradient._concatenate_chunks')
@mock.patch('pyresample.gradient.da')
def test_parallel_gradient_search(dask_da, _concatenate_chunks, delayed):
    """Test calling parallel_gradient_search()."""
    from pyresample.gradient import parallel_gradient_search

    def mock_cc(chunks):
        """Return the input."""
        return chunks

    _concatenate_chunks.side_effect = mock_cc

    # Mismatch in number of bands raises ValueError
    data = [np.zeros((1, 5, 5)), np.zeros((2, 5, 5))]
    try:
        parallel_gradient_search(data, None, None, None, None,
                                 None, None, None, None, None, None)
        raise
    except ValueError:
        pass

    data = [np.zeros((1, 5, 4)), np.ones((1, 5, 4)), None, None]
    src_x, src_y = [1, 2, 3, 4], [4, 5, 6, 4]
    # dst_x is used to check the target area shape, so needs "valid"
    # data.  The last values shouldn't matter as data[-2:] are None
    # and should be skipped.
    dst_x = [np.zeros((5, 5)), np.zeros((5, 5)), 'foo', 'bar']
    dst_y = [1, 2, 3, 4]
    src_gradient_xl, src_gradient_xp = [1, 2, None, None], [1, 2, None, None]
    src_gradient_yl, src_gradient_yp = [1, 2, None, None], [1, 2, None, None]
    # Destination slices are used only for padding, so the first two
    # None values shouldn't raise errors
    dst_slices = [None, None, [1, 2, 1, 3], [1, 3, 1, 4]]
    # The first two chunks have the same target location, same for the two last
    dst_mosaic_locations = [(0, 0), (0, 0), (0, 1), (0, 1)]

    res = parallel_gradient_search(data, src_x, src_y, dst_x, dst_y,
                                   src_gradient_xl, src_gradient_xp,
                                   src_gradient_yl, src_gradient_yp,
                                   dst_mosaic_locations, dst_slices,
                                   method='foo')
    assert len(res[(0, 0)]) == 2
    # The second padding shouldn't be in the chunks[(0, 1)] list
    assert len(res[(0, 1)]) == 1
    _concatenate_chunks.assert_called_with(res)
    # Two padding arrays
    assert dask_da.full.call_count == 2
    assert mock.call((1, 1, 2), np.nan) in dask_da.full.mock_calls
    assert mock.call((1, 2, 3), np.nan) in dask_da.full.mock_calls
    # Two resample calls
    assert dask_da.from_delayed.call_count == 2
    # The _gradient_resample_data() function has been delayed twice
    assert '_gradient_resample_data' in str(delayed.mock_calls[0])
    assert '_gradient_resample_data' in str(delayed.mock_calls[2])
    assert str(mock.call()(data[0],
                           src_x[0], src_y[0],
                           src_gradient_xl[0], src_gradient_xp[0],
                           src_gradient_yl[0], src_gradient_yp[0],
                           dst_x[0], dst_y[0],
                           method='foo')) == str(delayed.mock_calls[1])
    assert str(mock.call()(data[1],
                           src_x[1], src_y[1],
                           src_gradient_xl[1], src_gradient_xp[1],
                           src_gradient_yl[1], src_gradient_yp[1],
                           dst_x[1], dst_y[1],
                           method='foo')) == str(delayed.mock_calls[3])


def test_concatenate_chunks():
    """Test chunk concatenation for correct results."""
    from pyresample.gradient import _concatenate_chunks

    # 1-band image
    chunks = {(0, 0): [np.ones((1, 5, 4)), np.zeros((1, 5, 4))],
              (1, 0): [np.zeros((1, 5, 2))],
              (1, 1): [np.full((1, 3, 2), 0.5)],
              (0, 1): [np.full((1, 3, 4), -1)]}
    res = _concatenate_chunks(chunks).compute(scheduler='single-threaded')
    assert np.all(res[:5, :4] == 1.0)
    assert np.all(res[:5, 4:] == 0.0)
    assert np.all(res[5:, :4] == -1.0)
    assert np.all(res[5:, 4:] == 0.5)
    assert res.shape == (8, 6)

    # 3-band image
    chunks = {(0, 0): [np.ones((3, 5, 4)), np.zeros((3, 5, 4))],
              (1, 0): [np.zeros((3, 5, 2))],
              (1, 1): [np.full((3, 3, 2), 0.5)],
              (0, 1): [np.full((3, 3, 4), -1)]}
    res = _concatenate_chunks(chunks).compute(scheduler='single-threaded')
    assert np.all(res[:, :5, :4] == 1.0)
    assert np.all(res[:, :5, 4:] == 0.0)
    assert np.all(res[:, 5:, :4] == -1.0)
    assert np.all(res[:, 5:, 4:] == 0.5)
    assert res.shape == (3, 8, 6)


@mock.patch('pyresample.gradient.da')
def test_concatenate_chunks_stack_calls(dask_da):
    """Test that stacking is called the correct times in chunk concatenation."""
    from pyresample.gradient import _concatenate_chunks

    chunks = {(0, 0): [np.ones((1, 5, 4)), np.zeros((1, 5, 4))],
              (1, 0): [np.zeros((1, 5, 2))],
              (1, 1): [np.full((1, 3, 2), 0.5)],
              (0, 1): [np.full((1, 3, 4), -1)]}
    _ = _concatenate_chunks(chunks)
    dask_da.stack.assert_called_once_with(chunks[(0, 0)], axis=-1)
    dask_da.nanmax.assert_called_once()
    assert 'axis=2' in str(dask_da.concatenate.mock_calls[-2])
    assert 'squeeze' in str(dask_da.concatenate.mock_calls[-1])
