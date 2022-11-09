#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013-2022 Pyresample Developers
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
"""Test cases for SExtent class."""
import unittest

import numpy as np
import pytest

from pyresample.future.spherical import SExtent


class TestSExtent(unittest.TestCase):
    """Test SExtent."""

    def test_sextent_with_bad_value(self):
        """Test it raise error when providing bad values to SExtent."""
        # lon_min > lon_max
        extent = [10, 0, -40, 40]
        with pytest.raises(ValueError):
            SExtent(extent)

        # lat_min > lat_max
        extent = [-10, 0, 41, 40]
        with pytest.raises(ValueError):
            SExtent(extent)

        # lon_min < -180
        extent = [-181, 0, -40, 40]
        with pytest.raises(ValueError):
            SExtent(extent)

        # lon_max > 180
        extent = [-180, 181, -40, 40]
        with pytest.raises(ValueError):
            SExtent(extent)

        # lat_min < -90
        extent = [50, 60, -91, 40]
        with pytest.raises(ValueError):
            SExtent(extent)

        # lat_max > 90
        extent = [50, 60, -40, 91]
        with pytest.raises(ValueError):
            SExtent(extent)

        # Passing a list of extent with bad values
        list_extent = [[-180, 181, -40, 91], [-180, 181, -40, 91]]
        with pytest.raises(ValueError):
            SExtent(list_extent)

    def test_sextent_with_bad_input(self):
        """Test it raise error when providing bad inputs to SExtent."""
        # Length 5
        extent = [-180, 181, -40, 91, 50]
        with pytest.raises(ValueError):
            SExtent(extent)
        # Length 1
        extent = [-180]
        with pytest.raises(ValueError):
            SExtent(extent)
        # 4 single digits
        with pytest.raises(TypeError):
            SExtent(1, 2, 3, 4)
        # Length 0
        extent = []
        with pytest.raises(ValueError):
            SExtent(extent)
        # No argument
        with pytest.raises(ValueError):
            SExtent()
        # None
        with pytest.raises(TypeError):
            SExtent(None)
        # Passing a list of extent with bad length
        list_extent = [[-180, 180, -40, 90, 50], [-180, 180, -40, 90]]
        with pytest.raises(ValueError):
            SExtent(list_extent)

    def test_sextent_with_correct_format(self):
        """Test SExtent when providing correct extent(s) format and values."""
        # Accept list
        extent = [-180, -175, -40, 40]
        assert list(SExtent(extent))[0] == extent
        # Accept tuple
        extent = (-180, -175, -40, 40)
        assert list(SExtent(extent))[0] == list(extent)
        # Accept numpy array
        extent = np.array([-180, -175, -40, 40])
        assert list(SExtent(extent))[0] == extent.tolist()

        # Accept list of extents (list)
        extent1 = [50, 60, -40, 40]
        extent2 = [175, 180, -40, 40]
        list_extent = [extent1, extent2]
        assert np.allclose(list(SExtent(list_extent)), list_extent)
        # Accept list of extents (tuple)
        extent1 = (50, 60, -40, 40)
        extent2 = (175, 180, -40, 40)
        list_extent = [extent1, extent2]
        assert np.allclose(list(SExtent(list_extent)), list_extent)
        # Accept list of extents (np.array)
        extent1 = np.array([0, 60, -40, 40])
        extent2 = np.array([175, 180, -40, 40])
        list_extent = [extent1, extent2]
        assert np.allclose(list(SExtent(list_extent)), list_extent)
        # Accept multiple extents (list)
        extent1 = [50, 60, -40, 40]
        extent2 = [175, 180, -40, 40]
        list_extent = [extent1, extent2]
        assert np.allclose(list(SExtent(extent1, extent2)), list_extent)
        # Accept multiple extents (tuple)
        extent1 = (50, 60, -40, 40)
        extent2 = (175, 180, -40, 40)
        list_extent = [extent1, extent2]
        assert np.allclose(list(SExtent(extent1, extent2)), list_extent)
        # Accept multiple extents (np.array)
        extent1 = np.array([0, 60, -40, 40])
        extent2 = np.array([175, 180, -40, 40])
        list_extent = [extent1, extent2]
        assert np.allclose(list(SExtent(extent1, extent2)), list_extent)

    def test_single_sextent_bad_topology(self):
        """Test that raise error when the extents is a point or a line."""
        # - Point extent
        extent = [0, 0, 40, 40]
        with pytest.raises(ValueError):
            SExtent(extent)
        # - Line extent
        extent = [0, 10, 40, 40]
        with pytest.raises(ValueError):
            SExtent(extent)
        extent = [0, 0, -40, 50]
        with pytest.raises(ValueError):
            SExtent(extent)

    def test_multple_touching_extents(self):
        """Test that touching extents composing SExtent do not raise error."""
        extent1 = [0, 40, 0, 40]
        extent2 = [0, 40, -40, 0]
        _ = SExtent(extent1, extent2)

    def test_multple_overlapping_extents(self):
        """Test that raise error when the extents composing SExtent overlaps."""
        # Intersecting raise error
        extent1 = [0, 40, 0, 40]
        extent2 = [20, 60, 20, 60]
        with pytest.raises(ValueError):
            SExtent(extent1, extent2)

        # Duplicate extent raise errors
        extent1 = [0, 40, 0, 40]
        with pytest.raises(ValueError):
            SExtent(extent1, extent1)

        # Within extent raise errors
        extent1 = [0, 40, 0, 40]
        extent2 = [10, 20, 10, 20]
        with pytest.raises(ValueError):
            SExtent(extent1, extent2)

    def test_to_shapely(self):
        """Test shapely conversion."""
        from shapely.geometry import MultiPolygon, Polygon
        extent = [0, 20, 10, 30]
        bounds = [0, 10, 20, 30]
        shapely_sext = SExtent(extent).to_shapely()
        shapely_polygon = MultiPolygon([Polygon.from_bounds(*bounds)])
        assert shapely_sext.equals(shapely_polygon)

    def test_str(self):
        """Check the string representation."""
        extent = [0, 20, 10, 30]
        sext = SExtent(extent)
        self.assertEqual(str(sext), '[[0, 20, 10, 30]]')

    def test_repr(self):
        """Check the representation."""
        extent = [0, 20, 10, 30]
        sext = SExtent(extent)
        self.assertEqual(repr(sext), '[[0, 20, 10, 30]]')

    def test_is_global(self):
        """Check is_global property."""
        # Is global
        extent = [-180, 180, -90, 90]
        sext = SExtent(extent)
        assert sext.is_global

        # Is clearly not global
        extent = [-175, 180, -90, 90]
        sext = SExtent(extent)
        assert not sext.is_global

        # Is global, but with multiple extents
        extent1 = [-180, 0, -90, 90]
        extent2 = [0, 180, -90, 90]
        sext = SExtent(extent1, extent2)
        assert sext.is_global

        # Is not global, but polgon bounds ...
        extent1 = [-180, 0, -90, 90]
        extent2 = [0, 180, 0, 90]
        sext = SExtent(extent1, extent2)
        assert not sext.is_global

    def test_SExtent_single_not_intersect(self):
        """Check disjoint extents."""
        # - Not intersecting across longitude
        extent1 = [50, 60, -40, 40]
        extent2 = [175, 180, -40, 40]
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.disjoint(sext2)
        assert sext2.disjoint(sext1)

        assert not sext1.intersects(sext2)
        assert not sext2.intersects(sext1)

        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

        # - Not intersecting across longitude (at the antimeridian)
        extent1 = [175, 180, -40, 40]
        extent2 = [-180, -175, -40, 40]
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.disjoint(sext2)
        assert sext2.disjoint(sext1)

        assert not sext1.intersects(sext2)
        assert not sext2.intersects(sext1)

        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

        # - Not intersecting across latitude
        extent1 = [50, 60, -50, -40]
        extent2 = [50, 60, 0, 40]
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.disjoint(sext2)
        assert sext2.disjoint(sext1)

        assert not sext1.intersects(sext2)
        assert not sext2.intersects(sext1)

        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

    def test_SExtent_single_touches(self):
        """Check touching extents."""
        extent1 = [0, 10, 40, 60]
        extent2 = [0, 10, 30, 40]
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.touches(sext2)
        assert sext2.touches(sext1)

        # Touching extents are not disjoint !
        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        # Touching extents do no intersect !
        assert not sext1.intersects(sext2)
        assert not sext2.intersects(sext1)

        # Touching extents does not contain or are within
        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

    def test_SExtent_single_intersect(self):
        """Check intersecting extents."""
        extent1 = [0, 10, 40, 60]
        extent2 = [0, 20, 30, 50]
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.intersects(sext2)
        assert sext2.intersects(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

    def test_SExtent_single_untouching_within_contains(self):
        """Check untouching extents which contains/are within each other."""
        extent1 = [0, 40, 0, 40]   # contains extent2
        extent2 = [10, 20, 10, 20]  # is within extent1
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.intersects(sext2)
        assert sext2.intersects(sext1)

        assert sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert sext2.within(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

    def test_SExtent_single_touching_within_contains(self):
        """Check touching extents which contains/are within each other."""
        extent1 = [0, 40, 0, 40]   # contains extent2
        extent2 = [10, 40, 10, 40]  # is within extent1
        sext1 = SExtent(extent1)
        sext2 = SExtent(extent2)

        assert sext1.intersects(sext2)
        assert sext2.intersects(sext1)

        assert sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert sext2.within(sext1)

        # Although they touch interiorly, touches is False !
        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

    def test_SExtent_multiple_not_intersect(self):
        """Check non-intersecting extents."""
        extent11 = [50, 60, -40, 40]
        extent21 = [-180, -170, -40, 40]
        extent22 = [90, 100, -40, 40]
        sext1 = SExtent(extent11)
        sext2 = SExtent(extent21, extent22)

        assert sext1.disjoint(sext2)
        assert sext2.disjoint(sext1)

        assert not sext1.intersects(sext2)
        assert not sext2.intersects(sext1)

        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

    def test_SExtent_multiple_touches(self):
        """Check touching extents."""
        # - Touches (not crossing the interior)
        extent11 = [50, 60, -40, 40]
        extent21 = [60, 70, -40, 40]
        extent22 = [-175, -170, -40, 40]
        sext1 = SExtent(extent11)
        sext2 = SExtent(extent21, extent22)

        assert sext1.touches(sext2)
        assert sext2.touches(sext1)

        # Touching extents do no intersect !
        assert not sext1.intersects(sext2)
        assert not sext2.intersects(sext1)

        # Touching extents are not disjoint !
        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        # Touching extents does not contain or are within
        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

    def test_SExtent_multiple_intersect(self):
        """Check intersecting extents."""
        # - Intersect 1
        extent11 = [50, 60, -40, 40]
        extent21 = [55, 70, -40, 40]
        extent22 = [-175, -170, -40, 40]
        sext1 = SExtent(extent11)
        sext2 = SExtent(extent21, extent22)

        assert sext1.intersects(sext2)
        assert sext2.intersects(sext1)

        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

        assert not sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert not sext2.within(sext1)

    def test_SExtent_multiple_within(self):
        """Check touching extents which contains/are within each other."""
        extent11 = [-180, 180, -40, 40]  # containts extent 2
        extent21 = [55, 70, -40, 40]    # within extent 1
        extent22 = [-180, -170, -40, 40]  # within extent 1
        sext1 = SExtent(extent11)
        sext2 = SExtent(extent21, extent22)

        assert sext2.intersects(sext1)
        assert sext1.intersects(sext2)

        assert sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert sext2.within(sext1)

        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        # Touches is False because touches in the interior !
        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

    def test_SExtent_multiple_within_at_antimeridian(self):
        """Check touching extents which contains/are within each other."""
        # 2 within 1
        extent11 = [160, 180, -40, 40]   # containts extent 2
        extent12 = [-180, -160, -40, 40]  # containts extent 2
        extent21 = [170, 180, -20, 20]   # within extent 1
        extent22 = [-180, -170, -20, 20]  # within extent 1
        sext1 = SExtent(extent11, extent12)
        sext2 = SExtent(extent21, extent22)

        assert sext1.intersects(sext2)
        assert sext2.intersects(sext1)

        assert sext1.contains(sext2)
        assert not sext2.contains(sext1)

        assert not sext1.within(sext2)
        assert sext2.within(sext1)

        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        # Touches is False because touches in the interior !
        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

    def test_SExtent_multiple_equals(self):
        """Check extent equality."""
        extent11 = [160, 180, -40, 40]
        extent12 = [-180, -160, -40, 40]
        extent21 = [160, 170, -40, 40]
        extent22 = [170, 180, -40, 40]
        extent23 = [-180, -160, -40, 40]
        sext1 = SExtent(extent11, extent12)
        sext2 = SExtent(extent21, extent22, extent23)

        assert sext1.equals(sext2)
        assert sext2.equals(sext1)

        # Equal extents intersects, are within/contains each other
        assert sext1.intersects(sext2)
        assert sext2.intersects(sext1)

        assert sext1.within(sext2)
        assert sext2.within(sext1)

        assert sext1.contains(sext2)
        assert sext2.contains(sext1)

        # Equal extents are not disjoint !
        assert not sext1.disjoint(sext2)
        assert not sext2.disjoint(sext1)

        # Equal extents do not touches !
        assert not sext1.touches(sext2)
        assert not sext2.touches(sext1)

    def test_SExtent_binary_predicates_bad_input(self):
        extent = [160, 180, -40, 40]
        sext = SExtent(extent)
        # Disjoint
        with pytest.raises(TypeError):
            sext.disjoint("bad_dtype")
        # Intersects
        with pytest.raises(TypeError):
            sext.intersects("bad_dtype")
        # Within
        with pytest.raises(TypeError):
            sext.within("bad_dtype")
        # Contains
        with pytest.raises(TypeError):
            sext.contains("bad_dtype")
        # Touches
        with pytest.raises(TypeError):
            sext.touches("bad_dtype")
        # Equals
        with pytest.raises(TypeError):
            sext.equals("bad_dtype")
