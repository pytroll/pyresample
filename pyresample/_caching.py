"""Various tools for caching.

These tools are rarely needed by users and are used where they make sense
throughout pyresample.

"""
import hashlib
import json
import shutil
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

    def cache_clear(self, cache_dir: str | None = None):
        """Remove all on-disk files associated with this function.

        Intended to mimic the :func:`functools.cache` behavior.
        """
        cache_dir = self._get_cache_dir_from_config(cache_dir=cache_dir, cache_version="*")
        for zarr_dir in glob(str(cache_dir / "*.json")):
            shutil.rmtree(zarr_dir, ignore_errors=True)

    def __call__(self, *args, **kwargs):
        """Call decorated function and cache the result to JSON."""
        if not pyresample.config.get(self._cache_config_key, False):
            return self._callable(*args, **kwargs)

        existing_hash = hashlib.sha1()
        # TODO: exclude SwathDefinition for hashing reasons
        hashable_args = [hash(arg) if arg.__class__.__name__ in ("AreaDefinition",) else arg for arg in args]
        hashable_args += sorted(kwargs.items())
        existing_hash.update(json.dumps(tuple(hashable_args)).encode("utf8"))
        arg_hash = existing_hash.hexdigest()
        base_cache_dir = self._get_cache_dir_from_config(cache_version=self._cache_version)
        json_path = base_cache_dir / f"{arg_hash}.json"
        if not json_path.is_file():
            res = self._callable(*args, **kwargs)
            json_path.parent.mkdir(exist_ok=True)
            with open(json_path, "w") as json_cache:
                json.dump(res, json_cache, cls=_ExtraJSONEncoder)
        else:
            with open(json_path, "r") as json_cache:
                res = json.load(json_cache, object_hook=_object_hook)
        return res

    @staticmethod
    def _get_cache_dir_from_config(cache_dir: str | None = None, cache_version: int | str = 1) -> Path:
        cache_dir = cache_dir or pyresample.config.get("cache_dir")
        if cache_dir is None:
            raise RuntimeError("Can't use JSON caching. No 'cache_dir' configured.")
        subdir = f"geometry_slices_v{cache_version}"
        return Path(cache_dir) / subdir


class _ExtraJSONEncoder(json.JSONEncoder):
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
