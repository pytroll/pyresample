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

import dask.array as da
import numpy as np
import pytest

from pyresample.area_config import create_area_def
from pyresample.geometry import AreaDefinition


class TestResampleBlocksArea2Area:
    """Test resample_block in an area to area resampling case."""

    def setup_method(self):
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

    def test_resample_blocks_advises_on_using_mapblocks_when_source_and_destination_areas_are_the_same(self):
        """Test resample_blocks advises on using map_blocks when the source and destination areas are the same."""
        from pyresample.resampler import resample_blocks

        def fun(data):
            return data

        some_array = da.random.random(self.src_area.shape)
        with pytest.raises(ValueError) as excinfo:
            resample_blocks(fun, self.src_area, [some_array], self.src_area)
        assert "map_blocks" in str(excinfo.value)

    def test_resample_blocks_returns_array_with_destination_area_shape(self):
        """Test resample_blocks returns array with the shape of the destination area."""
        from pyresample.resampler import resample_blocks

        def fun(data, **kwargs):
            return data

        some_array = da.random.random(self.src_area.shape)
        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=40, dtype=float)
        assert res.shape == self.dst_area.shape

    def test_resample_blocks_works_in_chunks(self):
        """Test resample_blocks works in chunks."""
        from pyresample.resampler import resample_blocks
        self.cnt = 0

        def fun(*data, block_info=None, **kwargs):
            self.cnt += 1
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, self.cnt)

        res = resample_blocks(fun, self.src_area, [], self.dst_area, chunk_size=40, dtype=float)
        res = res.compute()
        assert np.nanmin(res) == 1
        assert np.nanmax(res) == 6
        assert res[40, 40] != res[39, 39]

    def test_resample_blocks_can_run_without_input(self):
        """Test resample_blocks can be run without input data."""
        from pyresample.resampler import resample_blocks
        self.cnt = 0

        def fun(*data, block_info=None, **kwargs):
            assert not data
            self.cnt += 1
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, self.cnt)

        res = resample_blocks(fun, self.src_area, [], self.dst_area, chunk_size=40, dtype=float)
        res = res.compute()
        assert np.nanmin(res) == 1
        assert np.nanmax(res) == 6

    def test_resample_blocks_uses_input(self):
        """Test resample_blocks makes use of input data."""
        from pyresample.resampler import resample_blocks

        def fun(data, block_info=None, **kwargs):
            val = np.mean(data)
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=200)

        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=200, dtype=float)
        np.testing.assert_allclose(res, 2742)

    def test_resample_blocks_returns_float_dtype(self):
        """Test resample_blocks returns the expected dtype."""
        from pyresample.resampler import resample_blocks

        def fun(data, block_info=None, **kwargs):
            val = np.mean(data)
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, val)

        some_array = da.arange(np.prod(self.src_area.shape)).reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=40, dtype=float)
        assert res.compute().dtype == float

    def test_resample_blocks_returns_int_dtype(self):
        """Test resample_blocks returns the expected dtype."""
        from pyresample.resampler import resample_blocks

        def fun(data, block_info=None, **kwargs):
            val = int(np.mean(data))
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, val)

        some_array = da.arange(np.prod(self.src_area.shape)).reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=40, dtype=int)
        assert res.compute().dtype == int

    def test_resample_blocks_uses_cropped_input(self):
        """Test resample_blocks uses cropped input data."""
        from pyresample.resampler import resample_blocks

        def fun(data, block_info=None, **kwargs):
            val = np.mean(data)
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=40, dtype=float)
        res = res.compute()
        assert not np.allclose(res[0, -1], res[-1, -1])

    def test_resample_blocks_uses_cropped_source_area(self):
        """Test resample_blocks uses cropped source area."""
        from pyresample.resampler import resample_blocks

        def fun(data, block_info=None, **kwargs):
            src_area = block_info[0]["area"]
            dst_area = block_info[None]["area"]
            val = np.mean(src_area.shape)
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=40)

        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=40, dtype=float)
        res = res.compute()
        assert np.allclose(res[0, -1], 25)
        assert np.allclose(res[-1, -1], 17)

    def test_resample_blocks_can_add_a_new_axis(self):
        """Test resample_blocks can add a new axis."""
        from pyresample.resampler import resample_blocks

        def fun(data, block_info=None, **kwargs):
            val = np.mean(data)
            dst_area = block_info[None]["area"]
            return np.full((2, ) + dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=200)

        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=(2, 40, 40), dtype=float)
        assert res.shape == (2,) + self.dst_area.shape
        res = res.compute()
        np.testing.assert_allclose(res[:, :40, 40:80], 1609.5)
        np.testing.assert_allclose(res[:, :40, 80:], 1574)
        assert res.shape == (2,) + self.dst_area.shape

    def test_resample_blocks_can_generate_gradient_indices(self):
        """Test resample blocks can generate gradient indices."""
        from pyresample.gradient import gradient_resampler_indices, gradient_resampler_indices_block
        from pyresample.resampler import resample_blocks

        chunks = 40
        indices = resample_blocks(gradient_resampler_indices_block, self.src_area, [], self.dst_area,
                                  chunk_size=(2, chunks, chunks), dtype=float)
        np.testing.assert_allclose(gradient_resampler_indices(self.src_area, self.dst_area), indices)

    def test_resample_blocks_can_gradient_resample(self):
        """Test resample_blocks can do gradient resampling."""
        from pyresample.gradient import (
            block_bilinear_interpolator,
            gradient_resampler,
            gradient_resampler_indices_block,
        )
        from pyresample.resampler import resample_blocks

        chunksize = 40

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1]).astype(float)
        some_array = some_array.reshape(self.src_area.shape).rechunk(chunks=chunksize)

        indices = resample_blocks(gradient_resampler_indices_block, self.src_area, [], self.dst_area,
                                  chunk_size=(2, chunksize, chunksize), dtype=float)

        res = resample_blocks(block_bilinear_interpolator, self.src_area, [some_array], self.dst_area,
                              dst_arrays=[indices], chunk_size=(chunksize, chunksize), dtype=some_array.dtype)
        np.testing.assert_allclose(gradient_resampler(some_array.compute(), self.src_area, self.dst_area), res)

    def test_resample_blocks_passes_kwargs(self):
        """Test resample_blocks passes kwargs."""
        from pyresample.resampler import resample_blocks

        def fun(val=1, block_info=None, **kwargs):
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, val)

        value = 12
        res = resample_blocks(fun, self.src_area, [], self.dst_area, val=value, chunk_size=40, dtype=float)
        res = res.compute()
        assert np.nanmin(res) == value
        assert np.nanmax(res) == value

    def test_resample_blocks_chunks_dst_arrays(self):
        """Test resample_blocks chunks the dst_arrays."""
        from pyresample.resampler import resample_blocks

        def fun(dst_array=None, block_info=None, **kwargs):
            dst_area = block_info[None]["area"]
            assert dst_array is not None
            assert dst_area.shape == dst_array.shape
            return dst_array

        dst_array = da.arange(np.prod(self.dst_area.shape)).reshape(self.dst_area.shape).rechunk(40)
        res = resample_blocks(fun, self.src_area, [], self.dst_area, dst_arrays=[dst_array], chunk_size=40, dtype=float)
        res = res.compute()
        np.testing.assert_allclose(res[:, 40:], dst_array[:, 40:])

    def test_resample_blocks_can_pass_block_info_about_source(self):
        """Test resample_blocks can pass block_info about the source chunk."""
        from pyresample.resampler import resample_blocks

        prev_block_info = []

        def fun(dst_array=None, block_info=None, **kwargs):
            assert dst_array is not None
            dst_area = block_info[None]["area"]
            assert dst_area.shape == dst_array.shape
            assert block_info is not None
            assert block_info[0]["shape"] == (100, 50)
            assert block_info[0]["array-location"] is not None
            assert block_info[0] not in prev_block_info
            assert block_info[0]["area"] is not None
            prev_block_info.append(block_info[0])
            return dst_array

        dst_array = da.arange(np.prod(self.dst_area.shape)).reshape(self.dst_area.shape).rechunk(40)
        res = resample_blocks(fun, self.src_area, [], self.dst_area, dst_arrays=[dst_array], chunk_size=40, dtype=float)
        _ = res.compute()

    def test_resample_blocks_can_pass_block_info_about_target(self):
        """Test resample_blocks can pass block_info about the target chunk."""
        from pyresample.resampler import resample_blocks

        prev_block_info = []

        def fun(dst_array=None, block_info=None, **kwargs):
            assert dst_array is not None
            dst_area = block_info[None]["area"]
            assert dst_area.shape == dst_array.shape
            assert block_info is not None
            assert block_info[None]["shape"] == (102, 102)
            assert block_info[None]["array-location"] is not None
            assert block_info[None] not in prev_block_info
            prev_block_info.append(block_info[None])
            return dst_array

        dst_array = da.arange(np.prod(self.dst_area.shape)).reshape(self.dst_area.shape).rechunk(40)
        res = resample_blocks(fun, self.src_area, [], self.dst_area, dst_arrays=[dst_array], chunk_size=40, dtype=float)
        _ = res.compute()

    def test_resample_blocks_supports_3d_dst_arrays(self):
        """Test resample_blocks supports 3d dst_arrays."""
        from pyresample.resampler import resample_blocks

        def fun(dst_array=None, block_info=None, **kwargs):
            dst_area = block_info[None]["area"]
            assert dst_array is not None
            assert dst_area.shape == dst_array.shape[1:]
            return dst_array[0, :, :]

        dst_array = da.arange(np.prod(self.dst_area.shape)).reshape((1, *self.dst_area.shape)).rechunk(40)
        res = resample_blocks(fun, self.src_area, [], self.dst_area, dst_arrays=[dst_array],
                              chunk_size=(40, 40), dtype=float)
        res = res.compute()
        np.testing.assert_allclose(res[:, 40:], dst_array[0, :, 40:])

    def test_resample_blocks_supports_multiple_input_arrays(self):
        """Test that resample_blocks supports multiple inputs."""
        from pyresample.resampler import resample_blocks

        def fun(data1, data2, block_info=None, **kwargs):
            val = np.mean((data1 + data2) / 2)
            dst_area = block_info[None]["area"]
            return np.full(dst_area.shape, val)

        some_array = da.arange(self.src_area.shape[0] * self.src_area.shape[1])
        some_array1 = some_array.reshape(self.src_area.shape).rechunk(chunks=200)
        some_array2 = some_array1.copy()

        res = resample_blocks(fun, self.src_area, [some_array1, some_array2], self.dst_area, chunk_size=200,
                              dtype=float)
        np.testing.assert_allclose(res, 2742)

    def test_resample_blocks_supports_3d_src_arrays(self):
        """Test resample_blocks supports 3d src_arrays."""
        from pyresample.resampler import resample_blocks

        def fun(src_array, block_info=None, **kwargs):
            src_area = block_info[0]["area"]
            dst_area = block_info[None]["area"]
            assert src_array.ndim == 3
            assert src_area.shape == src_array.shape[-2:]
            return np.full(src_array.shape[:-2] + dst_area.shape, 18)

        src_array = da.arange(np.prod(self.src_area.shape) * 3).reshape((3, *self.src_area.shape)).rechunk(40)
        res = resample_blocks(fun, self.src_area, [src_array], self.dst_area, chunk_size=(3, 40, 40), dtype=float)
        res = res.compute()
        assert res.ndim == 3
        assert np.nanmean(res) == 18

    def test_resample_blocks_supports_3d_src_arrays_with_multiple_chunks_on_non_xy_dims(self):
        """Test resample_blocks supports 3d src_arrays with multiple chunks on non xy dimensions."""
        from pyresample.resampler import resample_blocks

        def fun(src_array, block_info=None, **kwargs):
            src_area = block_info[0]["area"]
            dst_area = block_info[None]["area"]
            assert src_array.ndim == 3
            assert src_array.shape[-2:] == src_area.shape
            assert src_array.shape[0] == 3
            return np.full(src_array.shape[:-2] + dst_area.shape, 18)

        src_array = da.arange(np.prod(self.src_area.shape) * 3).reshape((3, *self.src_area.shape))
        src_array = src_array.rechunk((1, 40, 40))

        res = resample_blocks(fun, self.src_area, [src_array], self.dst_area, chunk_size=(3, 40, 40), dtype=float)
        res = res.compute()
        assert res.ndim == 3
        assert np.nanmean(res) == 18

    def test_resample_blocks_uses_custom_fill_value(self):
        """Test that resample_blocks uses a provided custom fill_value."""
        from pyresample.resampler import resample_blocks

        def fun(data, fill_value=np.nan, block_info=None):
            dst_area = block_info[None]["area"]
            val = int(np.mean(data))
            assert fill_value == -12
            return np.full(dst_area.shape, val)

        some_array = da.arange(np.prod(self.src_area.shape)).reshape(self.src_area.shape).rechunk(chunks=40)
        fill_value = -12
        res = resample_blocks(fun, self.src_area, [some_array], self.dst_area, chunk_size=40, dtype=int, fill_value=-12)
        assert res.compute().dtype == int
        assert res.compute()[0, 0] == fill_value

    def test_resample_blocks_supports_auto_chunks(self):
        from pyresample.resampler import resample_blocks

        def fun(src_array, block_info=None, **kwargs):
            dst_area = block_info[None]["area"]
            return np.full(src_array.shape[:-2] + dst_area.shape, 18)

        src_array = da.arange(np.prod(self.src_area.shape) * 3).reshape((3, *self.src_area.shape))
        src_array = src_array.rechunk((1, 40, 40))

        res = resample_blocks(fun, self.src_area, [src_array], self.dst_area, chunk_size=(3, "auto", "auto"),
                              dtype=float)
        res = res.compute()
        assert res.ndim == 3
        assert np.nanmean(res) == 18

    def test_resample_blocks_supports_warns_when_chunk_size_is_too_big(self, caplog):
        from pyresample.resampler import resample_blocks

        def fun(src_array, block_info=None, **kwargs):
            dst_area = block_info[None]["area"]
            return np.full(src_array.shape[:-2] + dst_area.shape, 18)

        src_area = create_area_def("epsg4326", "EPSG:4326", 20000, 20000,
                                   (20., 60., 30., 70.))

        area_id = 'Suomi_3067'
        description = 'Suomi_kansallinen, EPSG 3067'
        proj_id = 'Suomi_3067'
        projection = 'EPSG:3067'
        width = 1160
        height = 1820
        from pyproj import Proj
        pp = Proj(proj='utm', zone=35, ellps='GRS80')
        xx1, yy1 = pp(15.82308183, 55.93417040)  # LL_lon, LL_lat
        xx2, yy2 = pp(43.12029189, 72.19756918)  # UR_lon, UR_lat
        area_extent = (xx1, yy1, xx2, yy2)
        dst_area = AreaDefinition(area_id, description, proj_id,
                                  projection, width, height,
                                  area_extent)

        _ = resample_blocks(fun, src_area, [], dst_area, chunk_size=(3, 2048, 2048), dtype=float)
        assert "The input area chunks are large." in caplog.text

    def test_resample_blocks_supports_auto_chunks_and_dst_array(self):
        from pyresample.resampler import resample_blocks

        def fun(src_array, dst_array, block_info=None, **kwargs):
            assert dst_array.ndim == 3
            dst_area = block_info[None]["area"]
            return np.full(src_array.shape[:-2] + dst_area.shape, 18)

        src_array = da.arange(np.prod(self.src_area.shape) * 3).reshape((3, *self.src_area.shape))
        src_array = src_array.rechunk((1, 40, 40))

        dst_array = da.arange(np.prod(self.dst_area.shape) * 2).reshape((2, *self.dst_area.shape))
        dst_array = src_array.rechunk((1, 40, 40))

        res = resample_blocks(fun, self.src_area, [src_array], self.dst_area, [dst_array],
                              chunk_size=(3, "auto", "auto"), dtype=float)
        res = res.compute()
        assert res.ndim == 3
        assert np.nanmean(res) == 18
