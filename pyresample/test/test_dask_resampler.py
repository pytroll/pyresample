#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021

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
"""Tests for the DaskResampler."""

import unittest
import dask.array as da
import xarray as xr
import numpy as np
from pyresample.resampler import DaskResampler, Slicer
from pyresample.geometry import AreaDefinition, SwathDefinition, IncompatibleAreas, InvalidArea
import pytest


def dummy_resampler(data, source_area, destination_area):
    """Resample by filling an array with the sum of the data."""
    return np.full(destination_area.shape, data.sum())


class TestDaskResampler(unittest.TestCase):
    """Test case for the DaskResampler class."""

    def setUp(self):
        """Set up the test case."""
        self.input_data = da.arange(100*100).reshape((100, 100)).rechunk(30).astype(float)
        self.src_area = AreaDefinition('dst', 'dst area', None,
                                       {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                       100, 100,
                                       (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        lons, lats = self.src_area.get_lonlats(chunks=self.input_data.chunks)
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        self.src_swath = SwathDefinition(lons, lats)
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))
        self.dr = DaskResampler(self.src_area, self.dst_area, dummy_resampler)

    def test_resampling_generates_a_dask_array(self):
        """Test that resampling generates a dask array."""
        res = self.dr.resample(self.input_data)
        self.assertIsInstance(res, da.Array)

    def test_resampling_has_the_size_of_the_target_area(self):
        """Test that resampling generates an array of the right size."""
        res = self.dr.resample(self.input_data)
        assert res.shape == self.dst_area.shape

    def test_resampling_keeps_the_chunk_size(self):
        """Test that resampling keeps the chunk size from the input."""
        res = self.dr.resample(self.input_data)
        assert res.chunksize == self.input_data.chunksize

    def test_resampling_result_has_no_nans_when_fully_covered(self):
        """Test that resampling does not produce nans with full coverage."""
        res = self.dr.resample(self.input_data)
        assert np.isfinite(res).all()

    def test_resampling_result_name_is_unique(self):
        """Test that resampling generates unique dask array names."""
        res1 = self.dr.resample(self.input_data)
        input_data = da.ones((100, 100))
        res2 = self.dr.resample(input_data)
        assert res1.name != res2.name
        assert res1.name.startswith('dummy_resampler')

    def test_resampling_reduces_input_data(self):
        """Test that resampling reduces the input data."""
        res = self.dr.resample(self.input_data)
        assert res.max() < 49995000  # sum of all self.input_data

    def test_gradient_resampler(self):
        """Test the gradient resampler."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        res = dr.resample(self.input_data)
        assert np.nanmin(res - 8000) > 0

    def test_gradient_resampler_3d(self):
        """Test the gradient resampler with 3d data."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        input_data = self.input_data[np.newaxis, :, :]
        res = dr.resample(input_data)
        assert res.ndim == 3
        assert res.shape[0] == 1
        assert np.nanmin(res - 8000) > 0

    def test_gradient_resampler_3d_chunked(self):
        """Test gradient resampler in 3d with chunked data."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        input_data = self.input_data[np.newaxis, :, :].rechunk(20)
        res = dr.resample(input_data)
        assert res.ndim == 3
        assert res.shape[0] == 1
        assert np.nanmin(res - 8000) > 0

    def test_gradient_resampler_2d_chunked(self):
        """Test gradient resampler in 3d with chunked data."""
        from pyresample.gradient import gradient_resampler
        dr = DaskResampler(self.src_area, self.dst_area, gradient_resampler)
        input_data = self.input_data.rechunk(20)
        res = dr.resample(input_data)
        assert res.ndim == 2
        assert np.nanmin(res - 8000) > 0


class TestAreaSlicer(unittest.TestCase):
    """Test the get_slice method for AreaSlicers."""

    def setUp(self):
        """Set up the test case."""
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

    def test_source_area_covers_dest_area(self):
        """Test source area covers dest area."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        slicer = Slicer(src_area, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start > 0 and x_slice.stop <= 100
        assert y_slice.start > 0 and y_slice.stop <= 100

    def test_source_area_does_not_cover_dest_area_entirely(self):
        """Test source area does not cover dest area entirely."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 4440000.0, -5550000.0, -6660000.0))

        slicer = Slicer(src_area, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start > 0 and x_slice.stop < 100
        assert y_slice.start > 0 and y_slice.stop >= 100

    def test_source_area_does_not_cover_dest_area_at_all(self):
        """Test source area does not cover dest area at all."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  80, 100,
                                  (5550000.0, 3330000.0, -5550000.0, -5550000.0))

        slicer = Slicer(src_area, self.dst_area)
        with pytest.raises(IncompatibleAreas):
            slicer.get_slices()

    def test_dest_area_is_outside_source_area_domain(self):
        """Test dest area is outside the source area domain (nan coordinates)."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        dst_area = AreaDefinition('merc', 'merc', None,
                                  {'proj': 'merc', 'lon_0': 120.0,
                                   'lat_0': 0,
                                   'ellps': 'bessel'},
                                  102, 102,
                                  (-100000, -100000,
                                   100000, 100000))
        slicer = Slicer(src_area, dst_area)
        with pytest.raises(IncompatibleAreas):
            slicer.get_slices()

    def test_barely_touching_chunks_intersection(self):
        """Test that barely touching chunks generate slices on intersection."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        dst_area = AreaDefinition('moll', 'moll', None,
                                  {
                                      'ellps': 'WGS84',
                                      'lon_0': '0',
                                      'proj': 'moll',
                                      'units': 'm'
                                  },
                                  102, 102,
                                  (-18040095.6961, 4369712.0686,
                                   18040095.6961, 9020047.8481))
        slicer = Slicer(src_area, dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start > 0 and x_slice.stop < 100
        assert y_slice.start > 0 and y_slice.stop >= 100

    def test_slicing_an_area_with_infinite_bounds(self):
        """Test slicing an area with infinite bounds."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'proj': 'merc'},
                                  100, 100,
                                  (-10000.0, -10000.0, 0.0, 0.0))

        dst_area = AreaDefinition('moll', 'moll', None,
                                  {
                                      'ellps': 'WGS84',
                                      'lon_0': '0',
                                      'proj': 'moll',
                                      'units': 'm'
                                  },
                                  102, 102,
                                  (-100000.0, -4369712.0686,
                                   18040096.0, 9020047.8481))

        slicer = Slicer(src_area, dst_area)
        with pytest.raises(InvalidArea):
            slicer.get_slices()


class TestSwathSlicer(unittest.TestCase):
    """Test the get_slice function when input is a swath."""

    def setUp(self):
        """Set up the test case."""
        chunks = 10
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))
        self.src_area = AreaDefinition(
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

        lons, lats = self.src_area.get_lonlats(chunks=chunks)
        lons = xr.DataArray(lons.persist())
        lats = xr.DataArray(lats.persist())
        self.src_swath = SwathDefinition(lons, lats)

    def test_slicer_init(self):
        """Test slicer initialization."""
        slicer = Slicer(self.src_area, self.dst_area)
        assert slicer.area_to_crop == self.src_area
        assert slicer.area_to_contain == self.dst_area

    def test_source_swath_slicing_does_not_return_full_dataset(self):
        """Test source area covers dest area."""
        slicer = Slicer(self.src_swath, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert (x_slice, y_slice) == (slice(0, 36), slice(14, 91))
        assert x_slice.start == 0
        assert x_slice.stop == 36
        assert y_slice.start == 14
        assert y_slice.stop == 91

    def test_source_area_slicing_does_not_return_full_dataset(self):
        """Test source area covers dest area."""
        slicer = Slicer(self.src_area, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start == 0
        assert x_slice.stop == 35
        assert y_slice.start == 16
        assert y_slice.stop == 94

    def test_area_get_polygon_returns_a_polygon(self):
        """Test getting a polygon returns a polygon."""
        from shapely.geometry import Polygon
        slicer = Slicer(self.src_area, self.dst_area)
        poly = slicer.get_polygon()
        assert isinstance(poly, Polygon)

    def test_swath_get_polygon_returns_a_polygon(self):
        """Test getting a polygon returns a polygon."""
        from shapely.geometry import Polygon
        slicer = Slicer(self.src_swath, self.dst_area)
        poly = slicer.get_polygon()
        assert isinstance(poly, Polygon)

    def test_cannot_slice_a_string(self):
        """Test that we cannot slice a string."""
        with pytest.raises(NotImplementedError):
            Slicer("my_funky_area", self.dst_area)


class TestDaskResamplerFromSwath(unittest.TestCase):
    """Test case for the DaskResampler class swath to area."""

    def setUp(self):
        """Set up the test case."""
        chunks = 30
        self.input_data = da.arange(100*50).reshape((100, 50)).rechunk(chunks).astype(float)
        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))
        self.src_area = AreaDefinition(
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

        lons, lats = self.src_area.get_lonlats(chunks=chunks)
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        self.src_swath = SwathDefinition(lons, lats)

    def test_gradient_resampler_2d_chunked(self):
        """Test gradient resampler in 2d with chunked data."""
        from pyresample.gradient import gradient_resampler
        dr_area = DaskResampler(self.src_area, self.dst_area, gradient_resampler, method='bilinear')
        res_area = dr_area.resample(self.input_data)
        dr_swath = DaskResampler(self.src_swath, self.dst_area, gradient_resampler, method='bilinear')
        res_swath = dr_swath.resample(self.input_data)
        np.testing.assert_allclose(res_area[:, 60:], res_swath[:, 60:], rtol=1e-1)

    def test_gradient_resampler_2d_via_indices(self):
        """Test gradient resample in 2d via indices."""
        from pyresample.gradient import gradient_resampler, gradient_resampler_indices
        dr_area = DaskResampler(self.src_area, self.dst_area, gradient_resampler, method='bilinear')
        res_area = dr_area.resample(self.input_data)
        dr_area2 = DaskResampler(self.src_area, self.dst_area, gradient_resampler_indices)
        res_area2 = dr_area2.resample_via_indices(self.input_data)
        np.testing.assert_allclose(res_area, res_area2)


class TestResampleBlocksArea2Area:
    """Test resample_block in an area to area resampling case."""

    def setup(self):
        """Set up the test case."""
        self.src_area = AreaDefinition(
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

        self.dst_area = AreaDefinition('euro40', 'euro40', None,
                                       {'proj': 'stere', 'lon_0': 14.0,
                                        'lat_0': 90.0, 'lat_ts': 60.0,
                                        'ellps': 'bessel'},
                                       102, 102,
                                       (-2717181.7304994687, -5571048.14031214,
                                        1378818.2695005313, -1475048.1403121399))

    def test_resample_blocks_returns_input_array_when_source_and_destination_areas_are_the_same(self):
        """Test resample_blocks returns input data when the source and destination areas are the same."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, *data):
            return data[0]

        some_array = da.random.random(self.src_area.shape)
        res = resample_blocks(self.src_area, self.src_area, fun, some_array)
        assert res is some_array

    def test_resample_blocks_returns_array_with_destination_area_shape(self):
        """Test resample_blocks returns array with the shape of the destination area."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, *data):
            return data[0]

        some_array = da.random.random(self.src_area.shape)
        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=40, dtype=float)
        assert res.shape == self.dst_area.shape

    def test_resample_blocks_works_in_chunks(self):
        """Test resample_blocks works in chunks."""
        from pyresample.resampler import resample_blocks
        self.cnt = 0

        def fun(src_area, dst_area, *data):
            self.cnt += 1
            return np.full(dst_area.shape, self.cnt)

        res = resample_blocks(self.src_area, self.dst_area, fun, chunks=40, dtype=float)
        res = res.compute()
        assert np.nanmin(res) == 1
        assert np.nanmax(res) == 6
        assert res[40, 40] != res[39, 39]

    def test_resample_blocks_can_run_without_input(self):
        """Test resample_blocks can be run without input data."""
        from pyresample.resampler import resample_blocks
        self.cnt = 0

        def fun(src_area, dst_area, *data):
            assert not data
            self.cnt += 1
            return np.full(dst_area.shape, self.cnt)

        res = resample_blocks(self.src_area, self.dst_area, fun, chunks=40, dtype=float)
        res = res.compute()
        assert np.nanmin(res) == 1
        assert np.nanmax(res) == 6

    def test_resample_blocks_uses_input(self):
        """Test resample_blocks makes use of input data."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, data):
            val = np.mean(data)
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=200)

        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=200, dtype=float)
        np.testing.assert_allclose(res, 2742)

    def test_resample_blocks_returns_float_dtype(self):
        """Test resample_blocks returns the expected dtype."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, data):
            val = np.mean(data)
            return np.full(dst_area.shape, val)

        some_array = da.arange(np.prod(self.src_area.shape)).reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=40, dtype=float)
        assert res.compute().dtype == float

    def test_resample_blocks_returns_int_dtype(self):
        """Test resample_blocks returns the expected dtype."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, data):
            val = int(np.mean(data))
            return np.full(dst_area.shape, val)

        some_array = da.arange(np.prod(self.src_area.shape)).reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=40, dtype=int)
        assert res.compute().dtype == int

    def test_resample_blocks_uses_cropped_input(self):
        """Test resample_blocks uses cropped input data."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, data):
            val = np.mean(data)
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=40, dtype=float)
        res = res.compute()
        assert not np.allclose(res[0, -1], res[-1, -1])

    def test_resample_blocks_uses_cropped_source_area(self):
        """Test resample_blocks uses cropped source area."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, data):
            val = np.mean(src_area.shape)
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=40, dtype=float)
        res = res.compute()
        assert np.allclose(res[0, -1], 25)
        assert np.allclose(res[-1, -1], 17)

    def test_resample_blocks_can_add_a_new_axis(self):
        """Test resample_blocks can add a new axis."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, data):
            val = np.mean(data)
            return np.full((2, ) + dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=200)

        res = resample_blocks(self.src_area, self.dst_area, fun, some_array, chunks=(2, 40, 40), dtype=float)
        assert res.shape == (2,) + self.dst_area.shape
        res = res.compute()
        np.testing.assert_allclose(res[:, :40, 40:80], 1609.5)
        np.testing.assert_allclose(res[:, :40, 80:], 1574)
        assert res.shape == (2,) + self.dst_area.shape

    def test_resample_blocks_can_generate_gradient_indices(self):
        """Test resample blocks can generate gradient indices."""
        from pyresample.resampler import resample_blocks
        from pyresample.gradient import gradient_resampler_indices

        chunks = 40

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=chunks)

        indices = resample_blocks(self.src_area, self.dst_area, gradient_resampler_indices,
                                  chunks=(2, chunks, chunks), dtype=float)
        np.testing.assert_allclose(gradient_resampler_indices(self.src_area, self.dst_area), indices)

    def test_resample_blocks_can_gradient_resample(self):
        """Test resample_blocks can do gradient resampling."""
        from pyresample.resampler import resample_blocks
        from pyresample.gradient import gradient_resampler_indices, gradient_resampler
        chunksize = 40

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1]).astype(float)
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=chunksize)

        indices = resample_blocks(self.src_area, self.dst_area, gradient_resampler_indices,
                                  chunks=(2, chunksize, chunksize), dtype=float)
        np.testing.assert_allclose(gradient_resampler_indices(self.src_area, self.dst_area), indices)
        from pyresample.resampler import bil2
        res = resample_blocks(self.src_area, self.dst_area, bil2, some_array, dst_arrays=[indices],
                              chunks=(chunksize, chunksize), dtype=some_array.dtype)
        np.testing.assert_allclose(gradient_resampler(some_array.compute(), self.src_area, self.dst_area), res)

    def test_resample_blocks_passes_kwargs(self):
        """Test resample_blocks passes kwargs."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, val=1):
            return np.full(dst_area.shape, val)

        value = 12
        res = resample_blocks(self.src_area, self.dst_area, fun, val=value, chunks=40, dtype=float)
        res = res.compute()
        assert np.nanmin(res) == value
        assert np.nanmax(res) == value

    def test_resample_blocks_chunks_dst_arrays(self):
        """Test resample_blocks chunks the dst_arrays."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, dst_array=None):
            assert dst_array is not None
            assert dst_area.shape == dst_array.shape
            return dst_array

        dst_array = da.arange(np.product(self.dst_area.shape)).reshape(self.dst_area.shape).rechunk(40)
        res = resample_blocks(self.src_area, self.dst_area, fun, dst_arrays=[dst_array], chunks=40, dtype=float)
        res = res.compute()
        np.testing.assert_allclose(res[:, 40:], dst_array[:, 40:])

    def test_resample_blocks_can_pass_block_info_about_source(self):
        """Test resample_blocks can pass block_info about the source chunk."""
        from pyresample.resampler import resample_blocks

        prev_block_info = []

        def fun(src_area, dst_area, dst_array=None, block_info=None):
            assert dst_array is not None
            assert dst_area.shape == dst_array.shape
            assert block_info is not None
            assert block_info[0]["shape"] == (100, 50)
            assert block_info[0]["array-location"] is not None
            assert block_info[0] not in prev_block_info
            prev_block_info.append(block_info[0])
            return dst_array

        dst_array = da.arange(np.product(self.dst_area.shape)).reshape(self.dst_area.shape).rechunk(40)
        res = resample_blocks(self.src_area, self.dst_area, fun, dst_arrays=[dst_array], chunks=40, dtype=float)
        res = res.compute()

    def test_resample_blocks_can_pass_block_info_about_target(self):
        """Test resample_blocks can pass block_info about the target chunk."""
        from pyresample.resampler import resample_blocks

        prev_block_info = []

        def fun(src_area, dst_area, dst_array=None, block_info=None):
            assert dst_array is not None
            assert dst_area.shape == dst_array.shape
            assert block_info is not None
            print(block_info)
            assert block_info[None]["shape"] == (102, 102)
            assert block_info[None]["array-location"] is not None
            assert block_info[None] not in prev_block_info
            prev_block_info.append(block_info[None])
            return dst_array

        dst_array = da.arange(np.product(self.dst_area.shape)).reshape(self.dst_area.shape).rechunk(40)
        res = resample_blocks(self.src_area, self.dst_area, fun, dst_arrays=[dst_array], chunks=40, dtype=float)
        res = res.compute()

    def test_resample_blocks_supports_3d_dst_arrays(self):
        """Test resample_blocks supports 3d dst_arrays."""
        from pyresample.resampler import resample_blocks

        def fun(src_area, dst_area, dst_array=None):
            assert dst_array is not None
            assert dst_area.shape == dst_array.shape[1:]
            return dst_array[0, :, :]

        dst_array = da.arange(np.product(self.dst_area.shape)).reshape((1, *self.dst_area.shape)).rechunk(40)
        res = resample_blocks(self.src_area, self.dst_area, fun, dst_arrays=[dst_array], chunks=40, dtype=float)
        res = res.compute()
        np.testing.assert_allclose(res[:, 40:], dst_array[0, :, 40:])

    # test_multiple_inputs
    # test output type
    # test 3d resampling
    # test chunks on non-xy dimensions
