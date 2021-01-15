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
import numpy as np
import dask.array as da
import xarray as xr


class TestGradientResampler(unittest.TestCase):
    """Test case for the gradient resampling."""

    def setUp(self):
        """Set up the test case."""
        from pyresample.gradient import GradientSearchResampler
        self.src_area = AreaDefinition('dst', 'dst area', None,
                                       {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                       100, 100,
                                       (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        self.src_swath = SwathDefinition(*self.src_area.get_lonlats())
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

        self.resampler = GradientSearchResampler(self.src_area, self.dst_area)
        self.swath_resampler = GradientSearchResampler(self.src_swath,
                                                       self.dst_area)

    def test_get_projection_coordinates_area_to_area(self):
        """Check that the coordinates are initialized, for area -> area."""
        assert self.resampler.prj is None
        self.resampler._get_projection_coordinates((10, 10))
        cdst_x = self.resampler.dst_x.compute()
        cdst_y = self.resampler.dst_y.compute()
        assert np.allclose(np.min(cdst_x), -2022632.1675016289)
        assert np.allclose(np.max(cdst_x), 2196052.591296284)
        assert np.allclose(np.min(cdst_y), 3517933.413092212)
        assert np.allclose(np.max(cdst_y), 5387038.893400168)
        assert self.resampler.use_input_coords
        assert self.resampler.prj is not None

    def test_get_projection_coordinates_swath_to_area(self):
        """Check that the coordinates are initialized, for swath -> area."""
        assert self.swath_resampler.prj is None
        self.swath_resampler._get_projection_coordinates((10, 10))
        cdst_x = self.swath_resampler.dst_x.compute()
        cdst_y = self.swath_resampler.dst_y.compute()
        assert np.allclose(np.min(cdst_x), -2697103.29912692)
        assert np.allclose(np.max(cdst_x), 1358739.8381279823)
        assert np.allclose(np.min(cdst_y), -5550969.708939591)
        assert np.allclose(np.max(cdst_y), -1495126.5716846888)
        assert self.swath_resampler.use_input_coords is False
        assert self.swath_resampler.prj is not None

    def test_get_gradients(self):
        """Test that coordinate gradients are computed correctly."""
        self.resampler._get_projection_coordinates((10, 10))
        assert self.resampler.src_gradient_xl is None
        self.resampler._get_gradients()
        assert self.resampler.src_gradient_xl.compute().max() == 0.0
        assert self.resampler.src_gradient_xp.compute().max() == -111000.0
        assert self.resampler.src_gradient_yl.compute().max() == 111000.0
        assert self.resampler.src_gradient_yp.compute().max() == 0.0

    def test_get_chunk_mappings(self):
        """Test that chunk overlap, and source and target slices are correct."""
        chunks = (10, 10)
        num_chunks = np.product(chunks)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        assert self.resampler.coverage_status is None
        self.resampler.get_chunk_mappings()
        # 8 source chunks overlap the target area
        covered_src_chunks = np.array([38, 39, 48, 49, 58, 59, 68, 69])
        res = np.where(self.resampler.coverage_status)[0]
        assert np.all(res == covered_src_chunks)
        # All *num_chunks* should have values in the lists
        assert len(self.resampler.coverage_status) == num_chunks
        assert len(self.resampler.src_slices) == num_chunks
        assert len(self.resampler.dst_slices) == num_chunks
        assert len(self.resampler.dst_mosaic_locations) == num_chunks
        # There's only one output chunk, and the covered source chunks
        # should have destination locations of (0, 0)
        res = np.array(self.resampler.dst_mosaic_locations)[covered_src_chunks]
        assert all([all(loc == (0, 0)) for loc in list(res)])

    def test_get_src_poly_area(self):
        """Test defining source chunk polygon for AreaDefinition."""
        chunks = (10, 10)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        poly = self.resampler._get_src_poly(0, 40, 0, 40)
        assert np.allclose(poly.area, 12365358458842.43)

    def test_get_src_poly_swath(self):
        """Test defining source chunk polygon for SwathDefinition."""
        chunks = (10, 10)
        self.swath_resampler._get_projection_coordinates(chunks)
        self.swath_resampler._get_gradients()
        # Swath area defs can't be sliced, so False is returned
        poly = self.swath_resampler._get_src_poly(0, 40, 0, 40)
        assert poly is False

    @mock.patch('pyresample.gradient.get_polygon')
    def test_get_dst_poly(self, get_polygon):
        """Test defining destination chunk polygon."""
        chunks = (10, 10)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        # First call should make a call to get_polygon()
        self.resampler._get_dst_poly('idx1', 0, 10, 0, 10)
        assert get_polygon.call_count == 1
        assert 'idx1' in self.resampler.dst_polys
        # The second call to the same index should come from cache
        self.resampler._get_dst_poly('idx1', 0, 10, 0, 10)
        assert get_polygon.call_count == 1

        # Swath defs raise AttributeError, and False is returned
        get_polygon.side_effect = AttributeError
        self.resampler._get_dst_poly('idx2', 0, 10, 0, 10)
        assert self.resampler.dst_polys['idx2'] is False

    def test_filter_data(self):
        """Test filtering chunks that do not overlap."""
        chunks = (10, 10)
        self.resampler._get_projection_coordinates(chunks)
        self.resampler._get_gradients()
        self.resampler.get_chunk_mappings()

        # Basic filtering.  There should be 8 dask arrays that each
        # have a shape of (10, 10)
        res = self.resampler._filter_data(self.resampler.src_x)
        valid = [itm for itm in res if itm is not None]
        assert len(valid) == 8
        shapes = [arr.shape for arr in valid]
        for shp in shapes:
            assert shp == (10, 10)

        # Destination x/y coordinate array filtering.  Again, 8 dask
        # arrays each with shape (102, 102)
        res = self.resampler._filter_data(self.resampler.dst_x, is_src=False)
        valid = [itm for itm in res if itm is not None]
        assert len(valid) == 8
        shapes = [arr.shape for arr in valid]
        for shp in shapes:
            assert shp == (102, 102)

        # Add a dimension to the given dataset
        data = da.random.random(self.src_area.shape)
        res = self.resampler._filter_data(data, add_dim=True)
        valid = [itm for itm in res if itm is not None]
        assert len(valid) == 8
        shapes = [arr.shape for arr in valid]
        for shp in shapes:
            assert shp == (1, 10, 10)

        # 1D and 3+D should raise NotImplementedError
        data = da.random.random((3,))
        try:
            res = self.resampler._filter_data(data, add_dim=True)
            raise IndexError
        except NotImplementedError:
            pass
        data = da.random.random((3, 3, 3, 3))
        try:
            res = self.resampler._filter_data(data, add_dim=True)
            raise IndexError
        except NotImplementedError:
            pass

    def test_resample_area_to_area_2d(self):
        """Resample area to area, 2d."""
        data = xr.DataArray(da.ones(self.src_area.shape, dtype=np.float64),
                            dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert np.allclose(res, 1)

    def test_resample_area_to_area_2d_fill_value(self):
        """Resample area to area, 2d, use fill value."""
        data = xr.DataArray(da.full(self.src_area.shape, np.nan,
                                    dtype=np.float64), dims=['y', 'x'])
        res = self.resampler.compute(
            data, method='bil',
            fill_value=2.0).compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert np.allclose(res, 2.0)

    def test_resample_area_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.src_area.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        res = self.resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        assert np.allclose(res[0, :, :], 1.0)
        assert np.allclose(res[1, :, :], 2.0)
        assert np.allclose(res[2, :, :], 3.0)

    def test_resample_swath_to_area_2d(self):
        """Resample swath to area, 2d."""
        data = xr.DataArray(da.ones(self.src_swath.shape, dtype=np.float64),
                            dims=['y', 'x'])
        res = self.swath_resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == self.dst_area.shape
        assert not np.all(np.isnan(res))


    def test_resample_swath_to_area_3d(self):
        """Resample area to area, 3d."""
        data = xr.DataArray(da.ones((3, ) + self.src_swath.shape,
                                    dtype=np.float64) *
                            np.array([1, 2, 3])[:, np.newaxis, np.newaxis],
                            dims=['bands', 'y', 'x'])
        res = self.swath_resampler.compute(
            data, method='bil').compute(scheduler='single-threaded')
        assert res.shape == (3, ) + self.dst_area.shape
        for i in range(res.shape[0]):
            arr = np.ravel(res[i, :, :])
            assert np.allclose(arr[np.isfinite(arr)], float(i + 1))


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
