"""Various tools for caching.

These tools are rarely needed by users and are used where they make sense
throughout pyresample.

"""

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pyresample


class JSONCache:
    """Decorator class to cache results to a JSON file on-disk."""

    def __init__(self, *args, **kwargs):
        self._callable = None
        if len(args) == 1 and not kwargs:
            self._callable = args[0]

    def __call__(self, *args, **kwargs):
        """Call decorated function and cache the result to JSON."""
        is_decorated = len(args) == 1 and isinstance(args[0], Callable)
        if is_decorated:
            self._callable = args[0]

        @functools.wraps(self._callable)
        def _func(*args, **kwargs):
            if not pyresample.config.get("cache_geom_slices", False):
                return self._callable(*args, **kwargs)

            # TODO: kwargs
            existing_hash = hashlib.sha1()
            # hashable_args = [hash(arg) if isinstance(arg, AreaDefinition) else arg for arg in args]
            hashable_args = [hash(arg) if arg.__class__.__name__ == "AreaDefinition" else arg for arg in args]
            existing_hash.update(json.dumps(tuple(hashable_args)).encode("utf8"))
            arg_hash = existing_hash.hexdigest()
            print(arg_hash)
            base_cache_dir = Path(pyresample.config.get("cache_dir")) / "geometry_slices"
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

        if is_decorated:
            return _func
        return _func(*args, **kwargs)


class _ExtraJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, slice):
            return {"__slice__": True, "start": obj.start, "stop": obj.stop, "step": obj.step}
        return super().default(obj)


def _object_hook(obj: object) -> Any:
    if isinstance(obj, dict) and obj.get("__slice__", False):
        return slice(obj["start"], obj["stop"], obj["step"])
    return obj
