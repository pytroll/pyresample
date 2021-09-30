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
"""Test base resampler class functionality."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pyresample.future.resamplers.resampler import Resampler


class FakeResampler(Resampler):
    """Helper fake resampler for easier testing."""

    def __init__(self, *args, cache_something=False, **kwargs):
        self.cache_something = cache_something
        self.precompute = mock.Mock(wraps=self.precompute)
        self.resample = mock.Mock(wraps=self.resample)
        super().__init__(*args, **kwargs)

    @property
    def should_cache(self):
        return self.cache_something and self.cache is not None

    def precompute(self, **kwargs):
        some_result = np.empty((2, 2))
        if self.should_cache:
            self.cache.store("my_result", some_result)
        # FIXME: Legacy resamplers returned a cache_id...is that what we want?
        return "my_result"

    def resample(self, data, **kwargs):
        # TODO: Replace with some hashing function
        cache_id = self.precompute(**kwargs)
        if self.should_cache:
            assert self.cache.load(cache_id)
        return np.empty(self.target_geo_def.shape)


@pytest.mark.parametrize(
    "src",
    [
        lazy_fixture("swath_def_numpy"),
        lazy_fixture("swath_def_dask"),
        lazy_fixture("swath_def_xarray_numpy"),
        lazy_fixture("swath_def_xarray_dask"),
    ]
)
@pytest.mark.parametrize(
    "dst",
    [
        lazy_fixture("area_def_lcc_conus_1km"),
    ]
)
@pytest.mark.parametrize(
    ("cache", "cache_something"),
    [
        (None, False),
    ]
)
def test_resampler(src, dst, cache, cache_something):
    """Test basic operations of the base resampler with and without a caching."""
    rs = FakeResampler(src, dst, cache, cache_something=cache_something)
    some_data = np.zeros(src.shape, dtype=np.float64)
    resample_results = rs.resample(some_data)
    rs.precompute.assert_called_once()
    assert resample_results.shape == dst.shape
