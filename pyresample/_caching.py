"""Various tools for caching.

These tools are rarely needed by users and are used where they make sense
throughout pyresample.

"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import warnings
from functools import update_wrapper
from glob import glob
from pathlib import Path
from typing import Any, Callable, Mapping

import pyresample

logger = logging.getLogger(__name__)


class JSONCacheHelper:
    """Decorator class to cache results to a JSON file on-disk."""

    def __init__(
            self,
            func: Callable,
            cache_config_key: str,
            cache_version: int = 1,
    ):
        self._callable = func
        self._cache_config_key = cache_config_key
        self._cache_version = cache_version
        self._uncacheable_arg_type_names = ("",)

    @staticmethod
    def cache_clear(cache_dir: str | None = None):
        """Remove all on-disk files associated with this function.

        Intended to mimic the :func:`functools.cache` behavior.
        """
        cache_path = _get_cache_dir_from_config(cache_dir=cache_dir, cache_version="*")
        for json_file in glob(str(cache_path / "*.json")):
            os.remove(json_file)

    def __call__(self, *args):
        """Call decorated function and cache the result to JSON."""
        should_cache = pyresample.config.get(self._cache_config_key, False)
        if not should_cache:
            return self._callable(*args)

        try:
            arg_hash = _hash_args(args)
        except TypeError as err:
            warnings.warn("Cannot cache function due to unhashable argument: " + str(err),
                          stacklevel=2)
            return self._callable(*args)

        return self._run_and_cache(arg_hash, args)

    def _run_and_cache(self, arg_hash: str, args: tuple[Any]) -> Any:
        base_cache_dir = _get_cache_dir_from_config(cache_version=self._cache_version)
        json_path = base_cache_dir / f"{arg_hash}.json"
        if not json_path.is_file():
            res = self._callable(*args)
            json_path.parent.mkdir(exist_ok=True)
            with open(json_path, "w") as json_cache:
                json.dump(res, json_cache, cls=_JSONEncoderWithSlice)

        # for consistency, always load the cached result
        with open(json_path, "r") as json_cache:
            res = json.load(json_cache, object_hook=_object_hook)
        return res


def _get_cache_dir_from_config(cache_dir: str | None = None, cache_version: int | str = 1) -> Path:
    cache_dir = cache_dir or pyresample.config.get("cache_dir")
    if cache_dir is None:
        raise RuntimeError("Can't use JSON caching. No 'cache_dir' configured.")
    subdir = f"geometry_slices_v{cache_version}"
    return Path(cache_dir) / subdir


def _hash_args(args: tuple[Any]) -> str:
    from pyresample.future.geometry import AreaDefinition, SwathDefinition
    from pyresample.geometry import AreaDefinition as LegacyAreaDefinition
    from pyresample.geometry import SwathDefinition as LegacySwathDefinition

    hashable_args = []
    for arg in args:
        if isinstance(arg, (SwathDefinition, LegacySwathDefinition)):
            raise TypeError(f"Unhashable type ({type(arg)})")
        if isinstance(arg, (AreaDefinition, LegacyAreaDefinition)):
            arg = hash(arg)
        hashable_args.append(arg)
    arg_hash = hashlib.sha1()  # nosec
    arg_hash.update(json.dumps(tuple(hashable_args)).encode("utf8"))
    return arg_hash.hexdigest()


class _JSONEncoderWithSlice(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, slice):
            return {"__slice__": True, "start": obj.start, "stop": obj.stop, "step": obj.step}
        return super().default(obj)


def _object_hook(obj: object) -> Any:
    if isinstance(obj, dict) and obj.get("__slice__", False):
        return slice(obj["start"], obj["stop"], obj["step"])
    return obj


def cache_to_json_if(cache_config_key: str) -> Callable:
    """Decorate a function and cache the results to a JSON file on disk.

    This caching only happens if the ``pyresample.config`` boolean value for
    the provided key is ``True`` as well as some other conditions. See
    :class:`JSONCacheHelper` for more information. Most importantly this
    decorator does not limit how many items can be cached and does not clear
    out old entries. It is up to the user to manage the size of the cache.

    """
    def _decorator(func: Callable) -> Callable:
        zarr_cacher = JSONCacheHelper(func, cache_config_key)
        wrapper = update_wrapper(zarr_cacher, func)
        return wrapper

    return _decorator


class ZarrCache:
    """On-disk cache of named, geometry-only arrays stored as Zarr groups.

    This is meant for the expensive, reusable look-up tables (LUTs) that
    resamplers compute from their source and target geometries alone, for
    example target coordinates transformed to the source projection or
    neighbour indices. Each cache entry is a Zarr group holding one or more
    named arrays and is identified by a hexadecimal key, typically created with
    :func:`hash_cache_key` from everything the arrays depend on. Entries are
    stored in ``<cache_dir>/resampling_luts/<namespace>_v<version>/<key>.zarr``
    where ``cache_dir`` is ``pyresample.config["cache_dir"]`` unless given
    explicitly. Each user of the cache should have its own ``namespace``, and
    increase its ``version`` whenever the content or layout of what it stores
    changes, so that stale entries are never read.

    Entries are written to a temporary directory first and then renamed into
    place, so a partially written entry is never visible to readers and
    concurrent writers of the same entry do not corrupt each other.

    Whether a resampler should use this cache at all is controlled by the
    ``cache_resampling_luts`` configuration key, see :meth:`is_enabled`.

    Like the other caching in Pyresample, this cache does not limit its size nor
    expire old entries. It is up to the user to manage the cache directory.
    """

    config_key = "cache_resampling_luts"

    def __init__(self, namespace: str, version: int = 1, cache_dir: str | os.PathLike | None = None):
        self._namespace = namespace
        self._version = version
        self._cache_dir = cache_dir

    @classmethod
    def is_enabled(cls) -> bool:
        """Check whether caching resampling LUTs is enabled in ``pyresample.config``."""
        return bool(pyresample.config.get(cls.config_key, False))

    @property
    def directory(self) -> Path:
        """Get the directory holding the entries of this namespace and version."""
        cache_dir = self._cache_dir or pyresample.config.get("cache_dir")
        if cache_dir is None:
            raise RuntimeError("Can't use Zarr caching. No 'cache_dir' configured.")
        return Path(cache_dir) / "resampling_luts" / f"{self._namespace}_v{self._version}"

    def get_path(self, key: str) -> Path:
        """Get the path of the Zarr group for the cache entry *key*."""
        return self.directory / f"{key}.zarr"

    def load(self, key: str, chunks: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """Load the arrays of a cache entry lazily as dask arrays.

        Args:
            key: Key of the cache entry.
            chunks: Mapping of array name to the chunks the loaded dask array
                should have. Arrays not in the mapping use the chunking they
                were stored with.

        Returns:
            Mapping of array names to dask arrays, or ``None`` if the entry
            does not exist.
        """
        import dask.array as da
        import zarr

        path = self.get_path(key)
        if not path.is_dir():
            return None
        chunks = chunks or {}
        group = zarr.open_group(str(path), mode="r")
        logger.debug("Loading cached resampling LUTs from %s", path)
        return {name: da.from_zarr(group[name], chunks=chunks.get(name, None), inline_array=True)
                for name in sorted(group.array_keys())}

    def save(self, key: str, arrays: Mapping[str, Any]) -> None:
        """Compute and store arrays as a cache entry.

        Dask arrays are computed, in parallel, while being written. If the
        entry already exists it is left untouched.

        Args:
            key: Key of the cache entry.
            arrays: Mapping of array names to numpy or dask arrays.
        """
        import dask
        import dask.array as da

        path = self.get_path(key)
        if path.is_dir():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mkdtemp(prefix=f".{key}-", suffix=".tmp", dir=path.parent))
        try:
            writes = [da.to_zarr(da.asarray(arr), str(tmp_path), component=name, compute=False)
                      for name, arr in arrays.items()]
            dask.compute(*writes)
            logger.debug("Saving resampling LUTs to %s", path)
            os.rename(tmp_path, path)
        except OSError:
            if not path.is_dir():
                raise
            # Another process wrote the same entry first, keep theirs.
        finally:
            if tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)

    def clear(self) -> None:
        """Remove all entries of this namespace and version."""
        shutil.rmtree(self.directory, ignore_errors=True)


def hash_cache_key(*parts: Any) -> str:
    """Create a cache key from geometries, strings, numbers and numpy arrays.

    Geometries are hashed with their ``update_hash`` method, so the key only
    depends on the geometry's values (CRS, shape and extent for areas) and not
    on metadata such as its name.
    """
    import numpy as np

    the_hash = hashlib.sha1()  # nosec: B324
    for part in parts:
        if hasattr(part, "update_hash"):
            part.update_hash(the_hash)
        elif isinstance(part, np.ndarray):
            the_hash.update(np.ascontiguousarray(part))
        else:
            the_hash.update(json.dumps(part).encode("utf8"))
    return the_hash.hexdigest()
