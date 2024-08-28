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
"""Test the geometry objects.

DEPRECATED: Don't add new tests to this file. Put them in a module in ``test/test_geometry/`` instead.

"""
import random
from unittest.mock import MagicMock, patch

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from pyresample import geometry
from pyresample.geometry import IncompatibleAreas, combine_area_extents_vertical, concatenate_area_defs
from pyresample.test.utils import catch_warnings


class TestBaseDefinition:
    """Tests for the BaseDefinition class."""

    def test_base_type(self):
        """Test the base type."""
        from pyresample.geometry import BaseDefinition

        lons1 = np.arange(-135., +135, 50.)
        lats = np.ones_like(lons1) * 70.

        # Test dtype is preserved without longitude wrapping
        basedef = BaseDefinition(lons1, lats)
        lons, _ = basedef.get_lonlats()
        assert lons.dtype == lons1.dtype, \
            f"BaseDefinition did not maintain dtype of longitudes (in:{lons1.dtype} out:{lons.dtype})"

        lons1_ints = lons1.astype('int')
        basedef = BaseDefinition(lons1_ints, lats)
        lons, _ = basedef.get_lonlats()
        assert lons.dtype == lons1_ints.dtype, \
            f"BaseDefinition did not maintain dtype of longitudes (in:{lons1_ints.dtype} out:{lons.dtype})"

        # Test dtype is preserved with automatic longitude wrapping
        lons2 = np.where(lons1 < 0, lons1 + 360, lons1)
        with catch_warnings():
            basedef = BaseDefinition(lons2, lats)

        lons, _ = basedef.get_lonlats()
        assert lons.dtype == lons2.dtype, \
            f"BaseDefinition did not maintain dtype of longitudes (in:{lons2.dtype} out:{lons.dtype})"

        lons2_ints = lons2.astype('int')
        with catch_warnings():
            basedef = BaseDefinition(lons2_ints, lats)

        lons, _ = basedef.get_lonlats()
        assert lons.dtype == lons2_ints.dtype, \
            f"BaseDefinition did not maintain dtype of longitudes (in:{lons2_ints.dtype} out:{lons.dtype})"


def assert_np_dict_allclose(dict1, dict2):
    """Check allclose on dicts."""
    assert set(dict1.keys()) == set(dict2.keys())
    for key, val in dict1.items():
        try:
            np.testing.assert_allclose(val, dict2[key])
        except TypeError:
            assert val == dict2[key]


class TestStackedAreaDefinition:
    """Test the StackedAreaDefition."""

    def test_append(self):
        """Appending new definitions."""
        area1 = geometry.AreaDefinition("area1", 'area1', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3715498.9194295374,
                                            -1830246.0673044831, 3251436.5796920112)
                                        )

        area2 = geometry.AreaDefinition("area2", 'area2', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 4179561.259167064,
                                            -1830246.0673044831, 3715498.9194295374)
                                        )

        adef = geometry.StackedAreaDefinition(area1, area2)
        assert len(adef.defs) == 1
        assert adef.defs[0].area_extent == (3738502.0095458371,
                                            4179561.259167064,
                                            -1830246.0673044831,
                                            3251436.5796920112)

        # same
        area3 = geometry.AreaDefinition("area3", 'area3', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3251436.5796920112,
                                         -1830246.0673044831, 2787374.2399544837))
        adef.append(area3)
        assert len(adef.defs) == 1
        assert adef.defs[0].area_extent == (3738502.0095458371,
                                            4179561.259167064,
                                            -1830246.0673044831,
                                            2787374.2399544837)
        assert isinstance(adef.squeeze(), geometry.AreaDefinition)

        # transition
        area4 = geometry.AreaDefinition("area4", 'area4', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (5567747.7409681147, 2787374.2399544837,
                                         -1000.3358822065015, 2323311.9002169576))

        adef.append(area4)
        assert len(adef.defs) == 2
        assert adef.defs[-1].area_extent == (5567747.7409681147,
                                             2787374.2399544837,
                                             -1000.3358822065015,
                                             2323311.9002169576)

        assert adef.height == 4 * 464
        assert isinstance(adef.squeeze(), geometry.StackedAreaDefinition)

        adef2 = geometry.StackedAreaDefinition()
        assert len(adef2.defs) == 0

        adef2.append(adef)
        assert len(adef2.defs) == 2
        assert adef2.defs[-1].area_extent == (5567747.7409681147,
                                              2787374.2399544837,
                                              -1000.3358822065015,
                                              2323311.9002169576)

        assert adef2.height == 4 * 464

    def test_get_lonlats(self):
        """Test get_lonlats on StackedAreaDefinition."""
        area3 = geometry.AreaDefinition("area3", 'area3', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (3738502.0095458371, 3251436.5796920112,
                                         -1830246.0673044831, 2787374.2399544837))

        # transition
        area4 = geometry.AreaDefinition("area4", 'area4', "geosmsg",
                                        {'a': '6378169.0', 'b': '6356583.8',
                                         'h': '35785831.0', 'lon_0': '0.0',
                                         'proj': 'geos', 'units': 'm'},
                                        5568, 464,
                                        (5567747.7409681147, 2787374.2399544837,
                                         -1000.3358822065015, 2323311.9002169576))

        final_area = geometry.StackedAreaDefinition(area3, area4)
        assert len(final_area.defs) == 2
        lons, lats = final_area.get_lonlats()
        lons0, lats0 = final_area.defs[0].get_lonlats()
        lons1, lats1 = final_area.defs[1].get_lonlats()
        np.testing.assert_allclose(lons[:464, :], lons0)
        np.testing.assert_allclose(lons[464:, :], lons1)
        np.testing.assert_allclose(lats[:464, :], lats0)
        np.testing.assert_allclose(lats[464:, :], lats1)

        # check that get_lonlats with chunks definition doesn't cause errors and output arrays are equal
        with pytest.raises(ValueError):
            # too many chunks
            _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((200, 264, 464), (5570,)))
        # right amount of chunks, different shape
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((464, 470), (5568,)))
        # only one chunk value
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=464)

        # only one set of chunks in a tuple
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=(464, 5568))
        # too few chunks
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((464,), (5568,)))
        # right amount of chunks, same shape
        _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks=((464, 464), (5568,)))

    def test_combine_area_extents(self):
        """Test combination of area extents."""
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 6, 3, 2)
        res = combine_area_extents_vertical(area1, area2)
        assert res == [1, 6, 3, 4]

        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        res = combine_area_extents_vertical(area1, area2)
        assert res == [1, 2, 3, 6]

        # Non contiguous area extends shouldn't be combinable
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 5, 3, 7)
        pytest.raises(IncompatibleAreas, combine_area_extents_vertical,
                      area1, area2)

    def test_append_area_defs_fail(self):
        """Fail appending areas."""
        area1 = MagicMock()
        area1.proj_dict = {"proj": 'A'}
        area1.width = 4
        area1.height = 5
        area2 = MagicMock()
        area2.proj_dict = {'proj': 'B'}
        area2.width = 4
        area2.height = 6
        # res = combine_area_extents_vertical(area1, area2)
        pytest.raises(IncompatibleAreas, concatenate_area_defs, area1, area2)

    @patch('pyresample.geometry.AreaDefinition')
    def test_append_area_defs(self, adef):
        """Test appending area definitions."""
        x_size = random.randrange(6425)
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area1.crs = 'some_crs'
        area1.height = random.randrange(6425)
        area1.width = x_size

        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        area2.crs = 'some_crs'
        area2.height = random.randrange(6425)
        area2.width = x_size

        concatenate_area_defs(area1, area2)
        area_extent = [1, 2, 3, 6]
        y_size = area1.height + area2.height
        adef.assert_called_once_with(area1.area_id, area1.description, area1.proj_id,
                                     area1.crs, area1.width, y_size, area_extent)


def _check_final_area_lon_lat_with_chunks(final_area, lons, lats, chunks):
    """Compute the lons and lats with chunk definition and check that they are as expected."""
    lons_c, lats_c = final_area.get_lonlats(chunks=chunks)
    np.testing.assert_array_equal(lons, lons_c)
    np.testing.assert_array_equal(lats, lats_c)


class TestDynamicAreaDefinition:
    """Test the DynamicAreaDefinition class."""

    def test_freeze(self):
        """Test freezing the area."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'laea'})
        lons = [10, 10, 22, 22]
        lats = [50, 66, 66, 50]
        result = area.freeze((lons, lats),
                             resolution=3000,
                             proj_info={'lon_0': 16, 'lat_0': 58})

        np.testing.assert_allclose(result.area_extent, (-435000.0,
                                                        -873000.0,
                                                        435000.0,
                                                        906000.0))
        assert result.proj_dict['lon_0'] == 16
        assert result.proj_dict['lat_0'] == 58
        assert result.width == 290
        assert result.height == 593
        assert result.pixel_size_x == 3000
        assert result.pixel_size_y == 3000

        # make sure that setting `proj_info` once doesn't set it in the dynamic area
        result = area.freeze((lons, lats),
                             resolution=3000,
                             proj_info={'lon_0': 0})
        np.testing.assert_allclose(result.area_extent, (537000.0,
                                                        5379000.0,
                                                        1725000.0,
                                                        6999000.0))
        assert result.proj_dict['lon_0'] == 0
        # lat_0 could be provided or not depending on version of pyproj
        assert result.proj_dict.get('lat_0', 0) == 0
        assert result.width == 396
        assert result.height == 540
        assert result.pixel_size_x == 3000
        assert result.pixel_size_y == 3000

    def test_freeze_when_area_is_optimized_and_has_a_resolution(self):
        """Test freezing an optimized area with a resolution."""
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        swath = geometry.SwathDefinition(lons, lats)

        area10km = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                                  {'ellps': 'WGS84', 'proj': 'omerc'},
                                                  resolution=10000,
                                                  optimize_projection=True)

        result10km = area10km.freeze(swath)
        assert result10km.shape == (679, 330)

    def test_freeze_when_area_is_optimized_and_a_resolution_is_provided(self):
        """Test freezing an optimized area when provided a resolution."""
        nplats = np.array([[85.23900604248047, 62.256004333496094, 35.58000183105469],
                           [80.84000396728516, 60.74200439453125, 34.08500289916992],
                           [67.07600402832031, 54.147003173828125, 30.547000885009766]]).T
        lats = xr.DataArray(nplats)
        nplons = np.array([[-90.67900085449219, -21.565000534057617, -21.525001525878906],
                           [79.11000061035156, 7.284000396728516, -5.107000350952148],
                           [81.26400756835938, 29.672000885009766, 10.260000228881836]]).T
        lons = xr.DataArray(nplons)

        swath = geometry.SwathDefinition(lons, lats)

        area10km = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                                  {'ellps': 'WGS84', 'proj': 'omerc'},
                                                  optimize_projection=True)

        result10km = area10km.freeze(swath, 10000)
        assert result10km.shape == (679, 330)

    @pytest.mark.parametrize(
        ('lats',),
        [
            (np.linspace(-25.0, -10.0, 10),),
            (np.linspace(10.0, 25.0, 10),),
            (np.linspace(75, 90.0, 10),),
            (np.linspace(-75, -90.0, 10),),
        ],
    )
    @pytest.mark.parametrize('use_dask', [False, True])
    def test_freeze_longlat_antimeridian(self, lats, use_dask):
        """Test geographic areas over the antimeridian."""
        from pyresample.test.utils import CustomScheduler
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              'EPSG:4326')
        lons = np.linspace(175, 185, 10)
        lons[lons > 180] -= 360
        is_pole = (np.abs(lats) > 88).any()
        if use_dask:
            # if we aren't at a pole then we adjust the coordinates
            # that takes a total of 2 computations
            num_computes = 1 if is_pole else 2
            lons = da.from_array(lons, chunks=2)
            lats = da.from_array(lats, chunks=2)
            with dask.config.set(scheduler=CustomScheduler(num_computes)):
                result = area.freeze((lons, lats),
                                     resolution=0.0056)
        else:
            result = area.freeze((lons, lats),
                                 resolution=0.0056)

        extent = result.area_extent
        if is_pole:
            assert extent[0] < -178
            assert extent[2] > 178
            assert result.width == 64090
        else:
            assert extent[0] > 0
            assert extent[2] > 0
            assert result.width == 1788
        assert result.height == 2680

    def test_freeze_with_bb(self):
        """Test freezing the area with bounding box computation."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area', {'proj': 'omerc'},
                                              optimize_projection=True)
        lons = [[10, 12.1, 14.2, 16.3],
                [10, 12, 14, 16],
                [10, 11.9, 13.8, 15.7]]
        lats = [[66, 67, 68, 69.],
                [58, 59, 60, 61],
                [50, 51, 52, 53]]
        sdef = geometry.SwathDefinition(xr.DataArray(lons), xr.DataArray(lats))
        result = area.freeze(sdef)
        np.testing.assert_allclose(result.area_extent,
                                   [-335439.956533, 5502125.451125,
                                    191991.313351, 7737532.343683])

        assert result.width == 4
        assert result.height == 18
        # Test for properties and shape usage in freeze.
        area = geometry.DynamicAreaDefinition('test_area', 'A test area', {'proj': 'merc'},
                                              width=4, height=18)
        assert (18, 4) == area.shape
        result = area.freeze(sdef)
        np.testing.assert_allclose(result.area_extent,
                                   (996309.4426, 6287132.757981, 1931393.165263, 10837238.860543))
        area = geometry.DynamicAreaDefinition('test_area', 'A test area', {'proj': 'merc'},
                                              resolution=1000)
        assert 1000 == area.pixel_size_x
        assert 1000 == area.pixel_size_y

    def test_compute_domain(self):
        """Test computing size and area extent."""
        area = geometry.DynamicAreaDefinition('test_area', 'A test area',
                                              {'proj': 'laea'})
        corners = [1, 1, 9, 9]
        pytest.raises(ValueError, area.compute_domain, corners, 1, 1)

        area_extent, x_size, y_size = area.compute_domain(corners, shape=(5, 5))
        assert area_extent == (0, 0, 10, 10)
        assert x_size == 5
        assert y_size == 5

        area_extent, x_size, y_size = area.compute_domain(corners, resolution=2)
        assert area_extent == (0, 0, 10, 10)
        assert x_size == 5
        assert y_size == 5

    @pytest.mark.parametrize(
        (
            "antimeridian_mode",
            "expected_shape",
            "expected_extents",
            "include_proj_components",
            "exclude_proj_components"
        ),
        [
            (None, (22, 60), (164.5, 24.5, 194.5, 35.5), tuple(), ("+pm=180",)),
            ("modify_extents", (22, 60), (164.5, 24.5, 194.5, 35.5), tuple(), ("+pm=180",)),
            ("modify_crs", (22, 60), (164.5 - 180.0, 24.5, 194.5 - 180.0, 35.5), ("+pm=180",), tuple()),
            ("global_extents", (22, 720), (-180.0, 24.5, 180.0, 35.5), tuple(), ("+pm=180",)),
        ],
    )
    @pytest.mark.parametrize("use_dask", [False, True])
    def test_antimeridian_mode(self,
                               use_dask,
                               antimeridian_mode,
                               expected_shape,
                               expected_extents,
                               include_proj_components,
                               exclude_proj_components):
        """Test that antimeridian_mode affects the result."""
        dyn_area = geometry.DynamicAreaDefinition('test_area', '', {'proj': 'longlat'})
        lons, lats = _get_fake_antimeridian_lonlats(use_dask)
        area = dyn_area.freeze(lonslats=(lons, lats), resolution=0.5, antimeridian_mode=antimeridian_mode)
        proj_str = area.crs.to_proj4()

        assert area.shape == expected_shape
        np.testing.assert_allclose(area.area_extent, expected_extents)
        for include_comp in include_proj_components:
            assert include_comp in proj_str
        for exclude_comp in exclude_proj_components:
            assert exclude_comp not in proj_str

    def test_create_area_def_dynamic_areas(self):
        """Test certain parameter combinations produce a DynamicAreaDefinition."""
        from pyresample import create_area_def as cad
        from pyresample.geometry import DynamicAreaDefinition
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
        shape = (425, 850)
        area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

        assert isinstance(cad('ease_sh', projection, shape=shape), DynamicAreaDefinition)
        assert isinstance(cad('ease_sh', projection, area_extent=area_extent), DynamicAreaDefinition)

    def test_create_area_def_dynamic_omerc(self):
        """Test 'omerc' projections work in 'create_area_def'."""
        from pyresample import create_area_def as cad
        from pyresample.geometry import DynamicAreaDefinition
        area_def = cad('omerc_bb', {'ellps': 'WGS84', 'proj': 'omerc'})
        assert isinstance(area_def, DynamicAreaDefinition)


def _get_fake_antimeridian_lonlats(use_dask: bool) -> tuple:
    lon_min = 165
    lon_max = 195
    lons = np.arange(lon_min, lon_max, dtype=np.float64)
    lons[lons >= 180] -= 360.0
    lats = np.linspace(25.0, 35.0, lons.size, dtype=np.float64)
    if use_dask:
        lons = da.from_array(lons, chunks=lons.size // 3)
        lats = da.from_array(lats, chunks=lons.size // 3)
    return lons, lats
