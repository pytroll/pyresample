#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021-2022 Pyresample developers
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
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test the Area and Swath Slicers."""

import unittest

import numpy as np
import pytest
import xarray as xr

from pyresample import AreaDefinition, SwathDefinition
from pyresample.area_config import create_area_def
from pyresample.geometry import IncompatibleAreas
from pyresample.slicer import create_slicer


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
        slicer = create_slicer(src_area, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start > 0 and x_slice.stop <= 100
        assert y_slice.start > 0 and y_slice.stop <= 100

    def test_source_area_does_not_cover_dest_area_entirely(self):
        """Test source area does not cover dest area entirely."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  100, 100,
                                  (5550000.0, 4440000.0, -5550000.0, -6660000.0))

        slicer = create_slicer(src_area, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start > 0 and x_slice.stop < 100
        assert y_slice.start > 0 and y_slice.stop >= 100

    def test_source_area_does_not_cover_dest_area_at_all(self):
        """Test source area does not cover dest area at all."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                  80, 100,
                                  (5550000.0, 3330000.0, -5550000.0, -5550000.0))

        slicer = create_slicer(src_area, self.dst_area)
        with pytest.raises(IncompatibleAreas):
            slicer.get_slices()

    def test_source_area_does_not_cover_dest_area_at_all_2(self):
        """Test source area does not cover dest area at all."""
        src_area = AreaDefinition('src', 'src area', None,
                                  {'proj': 'merc', 'lon_0': -60, 'lat_0': 0, "ellps": "bessel"},
                                  100, 100,
                                  (-100000, -100000, 100000, 100000))
        dst_area = AreaDefinition('merc', 'merc', None,
                                  {'proj': 'merc', 'lon_0': 120.0,
                                   'lat_0': 0,
                                   'ellps': 'bessel'},
                                  102, 102,
                                  (-100000, -100000,
                                   100000, 100000))
        slicer = create_slicer(src_area, dst_area)
        with pytest.raises(IncompatibleAreas):
            slicer.get_slices()

    def test_dest_area_is_outside_source_area_domain(self):
        """Test dest area is outside the source area domain (nan coordinates)."""
        area_to_crop = AreaDefinition('dst', 'dst area', None,
                                      {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                      100, 100,
                                      (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        area_to_contain = AreaDefinition('merc', 'Kasimbar, Indonesia', None,
                                         {'proj': 'merc', 'lon_0': 120.0,
                                          'lat_0': 0,
                                          'ellps': 'WGS84'},
                                         102, 102,
                                         (-100000, -100000,
                                          100000, 100000))
        slicer = create_slicer(area_to_crop, area_to_contain)
        with pytest.raises(IncompatibleAreas):
            slicer.get_slices()

    def test_dest_area_is_partly_outside_source_area_domain(self):
        """Test dest area is outside the source area domain (nan coordinates)."""
        area_to_crop = AreaDefinition('dst', 'dst area', None,
                                      {'ellps': 'WGS84', 'h': '35785831', 'proj': 'geos'},
                                      100, 100,
                                      (5550000.0, 5550000.0, -5550000.0, -5550000.0))
        area_to_contain = AreaDefinition('afghanistan', 'Afghanistan', None,
                                         {'proj': 'merc', 'lon_0': 67.5,
                                          'lat_0': 35.0,
                                          'lat_ts': 35.0,
                                          'ellps': 'WGS84'},
                                         102, 102,
                                         (-1600000.0, 1600000.0, 1600000.0, 4800000.0))
        slicer = create_slicer(area_to_crop, area_to_contain)
        assert slicer.get_slices() == (slice(1, 24, None), slice(63, 89, None))

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
                                  (-1040095.6961, 4369712.0686,
                                   1040095.6961, 9020047.8481))
        slicer = create_slicer(src_area, dst_area)
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

        slicer = create_slicer(src_area, dst_area)
        with pytest.raises(IncompatibleAreas):
            slicer.get_slices()

    def test_slicing_works_with_extents_of_different_units(self):
        """Test a problematic case."""
        src_area = create_area_def("epsg4326", "EPSG:4326", 200, 200,
                                   (20., 60., 30., 70.))

        area_id = 'Suomi_3067'
        description = 'Suomi_kansallinen, EPSG 3067'
        proj_id = 'Suomi_3067'
        projection = 'EPSG:3067'
        width = 116
        height = 182
        from pyproj import Proj
        pp = Proj(proj='utm', zone=35, ellps='GRS80')
        xx1, yy1 = pp(15.82308183, 55.93417040)  # LL_lon, LL_lat
        xx2, yy2 = pp(43.12029189, 72.19756918)  # UR_lon, UR_lat
        area_extent = (xx1, yy1, xx2, yy2)
        dst_area = AreaDefinition(area_id, description, proj_id,
                                  projection, width, height,
                                  area_extent)

        slicer = create_slicer(src_area, dst_area[:50, :50])
        slice_x, slice_y = slicer.get_slices()
        assert 60 <= slice_x.stop < 65
        assert 50 <= slice_y.stop < 55

    def test_dest_area_is_a_swath(self):
        """Test dest area is a swath."""
        src_area = AreaDefinition('dst', 'dst area', None,
                                  {'ellps': 'WGS84', 'proj': 'latlong'},
                                  100, 100,
                                  (-20, 0, 20, 90))
        dst_swath = swath_from_area(self.dst_area, 35)

        slicer = create_slicer(src_area, dst_swath)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start >= 0 and x_slice.stop <= 200
        assert y_slice.start >= 0 and y_slice.stop <= 200


_SEVIRI_CRS = {"a": 6378169.0, "b": 6356583.8, "h": 35785831.0, "lon_0": 9.5, "proj": "geos", "units": "m"}
_SEVIRI_HALF_EXTENT = 5568748.2758
_GEOS_SOURCES = {
    # name: (crs, (height, width), area_extent)
    "full_disk": (_SEVIRI_CRS, (3712, 3712),
                  (-_SEVIRI_HALF_EXTENT, -_SEVIRI_HALF_EXTENT, _SEVIRI_HALF_EXTENT, _SEVIRI_HALF_EXTENT)),
    # Northern 1392 lines of a rapid scan, flipped like the native data (pyresample#728)
    "rapid_scan_strip": (_SEVIRI_CRS, (1392, 3712),
                         (_SEVIRI_HALF_EXTENT, _SEVIRI_HALF_EXTENT, -_SEVIRI_HALF_EXTENT, 1392187.0689)),
    # Area-of-interest crop over Europe, the top corners are off the disk
    "europe_crop": (_SEVIRI_CRS, (700, 1500), (-2500000.0, 3300000.0, 2000000.0, 5400000.0)),
}
_LONLAT_CRS = {"proj": "longlat", "datum": "WGS84"}
_EURO4_CRS = {"proj": "stere", "ellps": "bessel", "lat_0": 90.0, "lon_0": 14.0, "lat_ts": 60.0}
_EURO4_EXTENT = (-2717181.7304994687, -5571048.14031214, 1378818.2695005313, -1475048.1403121399)


def _slice_with_get_area_slices(src_area, dst_area):
    return src_area.get_area_slices(dst_area)


def _slice_with_area_slicer(src_area, dst_area):
    return create_slicer(src_area, dst_area).get_slices()


def _get_needed_source_slices(src_area, dst_area):
    """Get the source slices containing every target pixel center that is inside the source area."""
    dst_lons, dst_lats = dst_area.get_lonlats()
    cols, rows = src_area.get_array_indices_from_lonlat(dst_lons, dst_lats)
    valid = ~np.ma.getmaskarray(cols) & ~np.ma.getmaskarray(rows)
    cols = np.ma.getdata(cols)[valid]
    rows = np.ma.getdata(rows)[valid]
    return slice(int(cols.min()), int(cols.max()) + 1), slice(int(rows.min()), int(rows.max()) + 1)


def _assert_slices_cover(slices, needed_slices, max_missing, max_extra=5):
    for axis, got, needed in zip("xy", slices, needed_slices, strict=True):
        missing = max(got.start - needed.start, needed.stop - got.stop)
        assert missing <= max_missing, f"{axis} slice {got} misses {missing} pixels of the needed {needed}"
        extra = max(needed.start - got.start, got.stop - needed.stop)
        assert extra <= max_extra, f"{axis} slice {got} is {extra} pixels bigger than the needed {needed}"


@pytest.mark.parametrize(
    "get_slices", [_slice_with_get_area_slices, _slice_with_area_slicer], ids=["get_area_slices", "AreaSlicer"])
@pytest.mark.parametrize(
    ("src_name", "dst_crs", "dst_shape", "dst_extent", "max_missing"),
    [
        pytest.param("full_disk", _EURO4_CRS, (512, 512), _EURO4_EXTENT, 1,
                     id="target_inside_full_disk"),
        pytest.param("rapid_scan_strip", _EURO4_CRS, (512, 512), _EURO4_EXTENT, 1,
                     id="target_inside_partial_disk"),
        pytest.param("rapid_scan_strip", _LONLAT_CRS, (400, 400), (-10.0, 0.0, 30.0, 40.0), 0,
                     id="target_crosses_clipped_edge"),
        pytest.param("rapid_scan_strip", _LONLAT_CRS, (400, 400), (-85.0, 0.0, -45.0, 35.0), 2,
                     id="target_crosses_clipped_edge_and_limb"),
        pytest.param("europe_crop", _LONLAT_CRS, (400, 400), (-5.0, 25.0, 45.0, 45.0), 0,
                     id="target_crosses_two_clipped_edges"),
        pytest.param("europe_crop", _LONLAT_CRS, (400, 200), (0.0, 25.0, 15.0, 85.0), 0,
                     id="target_crosses_clipped_edge_and_opposite_limb"),
        pytest.param("europe_crop", _LONLAT_CRS, (400, 400), (-90.0, 20.0, 110.0, 89.0), 0,
                     id="target_contains_source"),
        pytest.param("full_disk", _LONLAT_CRS, (400, 400), (40.0, -10.0, 120.0, 50.0), 2,
                     id="target_crosses_limb"),
        pytest.param("full_disk", _LONLAT_CRS, (400, 800), (-180.0, -90.0, 180.0, 90.0), 5,
                     id="target_contains_full_disk"),
        pytest.param("rapid_scan_strip", dict(_SEVIRI_CRS, lon_0=0.0), (400, 1000),
                     _GEOS_SOURCES["rapid_scan_strip"][2], 16,
                     id="target_is_other_partial_disk"),
    ])
def test_geos_slices_cover_target_area(request, create_test_area, get_slices, src_name, dst_crs, dst_shape,
                                       dst_extent, max_missing):
    """Test that both ways of slicing a geos area keep every source pixel needed to cover the target area.

    ``dst_shape`` is ``(height, width)``. ``max_missing`` is how many needed
    rows or columns either method may cut off on any side. Non-zero values are
    known inaccuracies of the boundaries used for the intersection: the
    geostationary outline is drawn slightly inside the limb with a limited
    number of vertices, and the target outline only has a few vertices per
    side. Lower them as those boundaries get more accurate.
    """
    if get_slices is _slice_with_area_slicer and dst_crs["proj"] == "geos":
        # Part of the source disk isn't visible from the target satellite, so its outline
        # becomes inf when AreaSlicer reprojects it to the target's projection.
        request.applymarker(pytest.mark.xfail(raises=IncompatibleAreas, strict=True,
                                              reason="AreaSlicer can't crop a geos area to another geos projection"))
    crs, (height, width), area_extent = _GEOS_SOURCES[src_name]
    src_area = create_test_area(crs, width, height, area_extent)
    dst_area = create_test_area(dst_crs, dst_shape[1], dst_shape[0], dst_extent)

    slices = get_slices(src_area, dst_area)

    _assert_slices_cover(slices, _get_needed_source_slices(src_area, dst_area), max_missing)


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

        self.src_swath = swath_from_area(self.src_area, chunks)

    def test_slicer_init(self):
        """Test slicer initialization."""
        slicer = create_slicer(self.src_swath, self.dst_area)
        assert slicer.area_to_crop == self.src_swath
        assert slicer.area_to_contain == self.dst_area

    def test_source_swath_slicing_does_not_return_full_dataset(self):
        """Test source area covers dest area."""
        slicer = create_slicer(self.src_swath, self.dst_area)
        y_max, x_max = self.src_swath.shape
        y_max -= 1
        x_max -= 1
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start > 0 or x_slice.stop < x_max
        assert y_slice.start > 0 or y_slice.stop < y_max

    def test_source_area_slicing_does_not_return_full_dataset(self):
        """Test source area covers dest area."""
        slicer = create_slicer(self.src_swath, self.dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start == 0
        assert x_slice.stop == 41
        assert y_slice.start == 9
        assert y_slice.stop == 91

    def test_source_area_slicing_over_date_line(self):
        src_area = AreaDefinition(
            'omerc_otf',
            'On-the-fly omerc area',
            None,
            {'alpha': '8.99811271718795',
             'ellps': 'sphere',
             'gamma': '0',
             'k': '1',
             'lat_0': '0',
             'lonc': '179.8096029486222',
             'proj': 'omerc',
             'units': 'm'},
            50, 100,
            (-1461111.3603, 3440088.0459, 1534864.0322, 9598335.0457)
        )
        chunks = 10
        src_swath = swath_from_area(src_area, chunks)

        dst_area = AreaDefinition('somewhere in the pacific', 'somewhere', None,
                                  {'proj': 'stere', 'lon_0': 180.0,
                                   'lat_0': 90.0, 'lat_ts': 60.0,
                                   'ellps': 'bessel'},
                                  102, 102,
                                  (-2717181.7304994687, -5571048.14031214,
                                   1378818.2695005313, -1475048.1403121399))

        slicer = create_slicer(src_swath, dst_area)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start == 0
        assert x_slice.stop == 41
        assert y_slice.start == 9
        assert y_slice.stop == 91

    def test_source_area_slicing_with_custom_work_crs(self):
        """Test source area covers dest area."""
        from pyresample.slicer import SwathSlicer
        slicer = SwathSlicer(self.src_swath, self.dst_area, work_crs=self.src_area.crs)
        x_slice, y_slice = slicer.get_slices()
        assert x_slice.start == 0
        assert x_slice.stop == 41
        assert y_slice.start == 9
        assert y_slice.stop == 91

    def test_area_get_polygon_returns_a_polygon(self):
        """Test getting a polygon returns a polygon."""
        from shapely.geometry import Polygon
        slicer = create_slicer(self.src_swath, self.dst_area)
        poly = slicer.get_polygon_to_contain()
        assert isinstance(poly, Polygon)

    def test_swath_get_polygon_returns_a_polygon(self):
        """Test getting a polygon returns a polygon."""
        from shapely.geometry import Polygon
        slicer = create_slicer(self.src_swath, self.dst_area)
        poly = slicer.get_polygon_to_contain()
        assert isinstance(poly, Polygon)

    def test_cannot_slice_a_string(self):
        """Test that we cannot slice a string."""
        with pytest.raises(NotImplementedError):
            create_slicer("my_funky_area", self.dst_area)


def swath_from_area(src_area, chunks):
    """Create a SwathDefinition from an AreaDefinition."""
    lons, lats = src_area.get_lonlats(chunks=chunks)
    xrlons = xr.DataArray(lons.persist())
    xrlats = xr.DataArray(lats.persist())
    return SwathDefinition(xrlons, xrlats)
