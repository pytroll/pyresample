#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
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
"""Test various builtin cache classes."""

from pyresample.geometry import AreaDefinition
from pyresample import create_area_def
from pyresample.resampler import BaseResampler

class TestResampler(BaseResampler):
    """Fake resampler class to test out different caching needs."""

    def __init__(self, src_geom, dst_geom, cache=None):
        # TODO: This is a Pyresample 2.0 interface, but BaseResampler is currently 1.x
        super().__init__(src_geom, dst_geom)
        self.cache = cache

    def precompute(self, **kwargs):
        self.cache.store("scalar_int", 5)
        self.cache.store("scalar_float", 10.0)
        self.cache.store("a_list", ["a", "b"])
        self.cache.store("a_dict", {"a": 1, "b": 2})

    def compute(self, data, **kwargs):
        assert self.cache.load("scalar_int") == 5
        assert self.cache.load("scalar_float") == 10.0
        assert self.cache.load("a_list") == ["a", "b"]
        assert self.cache.load("a_dict") == {"a": 1, "b": 2}


def _create_resampler_with_cache(cache_inst, **kwargs):
    src_geom = create_area_def(
        "src_geometry",
        "EPSG:4326",
        shape=(200, 100),
        area_extent=(-10000, -10000, 10000, 10000),
    )
    dst_geom = create_area_def(
        "dst_geometry",
        "EPSG:9000",
        shape=(250, 150),
        area_extent=(-10000, -10000, 10000, 10000),
    )
    resampler = TestResampler(src_geom, dst_geom,
                              cache=cache_inst, **kwargs)
    return resampler


class TestInMemoryCache:
    """Tests for the InMemoryCache class."""

    def test_create(self):
        from pyresample._cache import InMemoryCache
        cache_inst = InMemoryCache()
        resampler = _create_resampler_with_cache(cache_inst)
        resampler.precompute()
        resampler.compute(None)
