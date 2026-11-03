# Copyright (C) 2026 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the on-disk Zarr cache."""

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest

import pyresample
from pyresample._caching import ZarrCache, hash_cache_key
from pyresample.test.utils import assert_maximum_dask_computes


@pytest.fixture
def cache(tmp_path):
    """Create a cache in a temporary directory."""
    return ZarrCache("test_luts", version=3, cache_dir=tmp_path)


def test_is_enabled_follows_config():
    assert not ZarrCache.is_enabled()
    with pyresample.config.set(cache_resampling_luts=True):
        assert ZarrCache.is_enabled()


def test_directory_uses_cache_dir_from_config(tmp_path):
    cache = ZarrCache("test_luts", version=2)
    with pyresample.config.set(cache_dir=tmp_path):
        assert cache.directory == tmp_path / "resampling_luts" / "test_luts_v2"
        assert cache.get_path("abc") == tmp_path / "resampling_luts" / "test_luts_v2" / "abc.zarr"


def test_directory_without_cache_dir_raises():
    with pyresample.config.set(cache_dir=None), pytest.raises(RuntimeError, match="cache_dir"):
        _ = ZarrCache("test_luts").directory


def test_load_missing_entry_returns_none(cache):
    assert cache.load("missing") is None


def test_save_and_load_roundtrip(cache):
    indices = np.arange(12, dtype=np.int64).reshape(3, 4)
    distances = da.linspace(0, 1, 30, chunks=7).reshape(2, 15)

    cache.save("key", {"indices": indices, "distances": distances})
    with assert_maximum_dask_computes(0):
        loaded = cache.load("key", chunks={"distances": (2, 5)})

    assert set(loaded) == {"indices", "distances"}
    assert all(isinstance(arr, da.Array) for arr in loaded.values())
    assert loaded["distances"].chunks == ((2,), (5, 5, 5))
    assert loaded["indices"].dtype == indices.dtype
    np.testing.assert_array_equal(loaded["indices"].compute(), indices)
    np.testing.assert_array_equal(loaded["distances"].compute(), distances.compute())


def test_save_does_not_overwrite_existing_entry(cache):
    cache.save("key", {"arr": np.zeros(3)})
    cache.save("key", {"arr": np.ones(3)})

    np.testing.assert_array_equal(cache.load("key")["arr"].compute(), np.zeros(3))


def test_save_leaves_no_temporary_files(cache):
    cache.save("key", {"arr": np.zeros(3)})

    assert os.listdir(cache.directory) == ["key.zarr"]


def test_failed_save_leaves_no_entry(cache):
    def _fail(*args, **kwargs):
        raise ValueError("Boom")

    failing = da.zeros(3, chunks=1).map_blocks(_fail, dtype=float)
    with pytest.raises(ValueError, match="Boom"):
        cache.save("key", {"arr": failing})

    assert cache.load("key") is None
    assert os.listdir(cache.directory) == []


def test_concurrently_written_entry_is_kept(cache):
    """Test that losing the race to write an entry keeps the winner's entry."""
    def _other_process_wins(src, dst):
        os.mkdir(dst)
        raise OSError("Directory not empty")

    with mock.patch("pyresample._caching.os.rename", side_effect=_other_process_wins):
        cache.save("key", {"arr": np.zeros(3)})

    assert os.listdir(cache.directory) == ["key.zarr"]


def test_clear(cache):
    cache.save("key", {"arr": np.zeros(3)})
    cache.clear()

    assert not cache.directory.exists()


def test_hash_cache_key_depends_on_values_not_names(create_test_area):
    proj_dict = {"proj": "stere", "lat_0": 90, "lon_0": 0, "ellps": "WGS84"}
    extent = (-1000, -1000, 1000, 1000)
    area1 = create_test_area(proj_dict, 10, 10, extent, attrs={"name": "one"})
    area2 = create_test_area(proj_dict, 10, 10, extent, attrs={"name": "two"})
    area3 = create_test_area(proj_dict, 10, 11, extent)

    assert hash_cache_key(area1, "wkt") == hash_cache_key(area2, "wkt")
    assert hash_cache_key(area1, "wkt") != hash_cache_key(area3, "wkt")
    assert hash_cache_key(area1, "wkt") != hash_cache_key(area1, "other wkt")
    assert hash_cache_key("a", "b") != hash_cache_key("ab")
    assert hash_cache_key(np.arange(3), 1.5) == hash_cache_key(np.arange(3), 1.5)
