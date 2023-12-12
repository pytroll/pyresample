"""Various tools for caching.

These tools are rarely needed by users and are used where they make sense
throughout pyresample.

"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from functools import update_wrapper
from glob import glob
from pathlib import Path
from typing import Any, Callable

import pyresample


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
