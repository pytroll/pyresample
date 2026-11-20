# Move nearest/bilinear LUT caching onto ZarrCache, with a converter for old Satpy caches

## Context
Satpy's `KDTreeResampler` and `BilinearResampler` (copied for reference to `satpy_resample/kdtree.py`) cache
their LUTs as `<cache_dir>/nn_lut-<sha1>.zarr` and `<cache_dir>/bil_lut-<sha1>.zarr`. The hash is
`BaseResampler.get_hash()`: source `update_hash`, then target `update_hash`, then `hash_dict(kwargs)`. It can't
be reversed, so old files can only be matched by recomputing the hash from a candidate area pair plus kwargs.
pyresample now has a self-describing, versioned `ZarrCache` (`pyresample/_caching.py`). The goal is to make the
nearest and bilinear resamplers use it, and to give users helpers that move existing Satpy caches over so they
don't have to recompute them.

Decisions taken (user): change the resamplers too, not only add a converter; offer a per-pair converter plus a
directory-scan batch helper.

Constraint (AGENTS.md): legacy interfaces are feature-frozen. So nearest caching goes on the modern
`future.resamplers.KDTreeNearestXarrayResampler`, **not** on `kd_tree.XArrayResamplerNN`, which is what Satpy's
`KDTreeResampler` uses today. Satpy only benefits once it switches to the future resampler; this is a Satpy-side
follow-up. Bilinear caching goes on `bilinear.XArrayBilinearResampler`, which is the supported interface and which
Satpy already uses.

## Cache location rule (same as gradient search)
A `cache_dir` passed to `precompute`/`resample` (as Satpy does with `scn.resample(..., cache_dir=...)`) enables
caching under that directory and overrides both `cache_resampling_luts` and `pyresample.config["cache_dir"]`.
Without it, the config/env settings decide. Reuse the gradient helper pattern (`_get_indices_cache(cache_dir)` in
`pyresample/gradient/__init__.py`). For bilinear, `get_bil_info()` has no `cache_dir` today, so thread it in from
`resample(..., cache_dir=None)`. The converter helpers below take the destination the same way: an optional
`dest_cache_dir`, falling back to the config.

## 1. Cache formats, defined next to their consumers
Follow the gradient pattern (`pyresample/gradient/__init__.py`: `_INDICES_CACHE`, `_describe_area`,
`_get_cached_indices`). Only area-to-area pairs are cached (reuse `is_area_to_area`), and only when
`ZarrCache.is_enabled()`.

- `pyresample/future/resamplers/nearest.py`
  - `_NEAREST_CACHE = ZarrCache("kdtree_nearest_indices", version=1)`
  - `_nearest_cache_key(src, dst, radius_of_influence, epsilon)` → `hash_cache_key(src, dst, {"neighbours": 1,
    "radius_of_influence": float(roi), "epsilon": epsilon})`. The radius is always the resolved number.
  - In `precompute()`: when `mask is None`, the pair is area-to-area and the cache is enabled, load
    `{"valid_input_index", "index_array"}` from the cache, or compute with `_get_neighbor_info`, save and reload.
    The loaded chunks must match the uncached ones. The result goes into `_internal_cache` as it does now. Masked
    calls are never cached (same rule as today).
- `pyresample/bilinear/xarr.py`
  - `_BILINEAR_CACHE = ZarrCache("bilinear_luts", version=1)` storing the 7 `BIL_COORDINATES` arrays.
  - `_bilinear_cache_key(src, dst, radius_of_influence, neighbours, epsilon, reduce_data)`.
  - Override `get_bil_info()` in `XArrayBilinearResampler`: if the cache is enabled, the pair is area-to-area and
    nothing has been computed yet, set the attributes from the cache (as `load_resampling_info` does). Otherwise
    call `super().get_bil_info()`, save, and reload. Leave the existing `save_resampling_info`/
    `load_resampling_info` untouched.
- Move `_describe_area` from `gradient/__init__.py` to `_caching.py` (as `describe_area`) so all three users share
  it. Store the description, plus the kwargs, with every entry.

## 2. Converter: new module `pyresample/utils/legacy_cache.py`
- `convert_legacy_cache(cache_dir, source_geo_def, target_geo_def, resampler, *, radius_of_influence=<Satpy
  default>, epsilon=0, remove=False) -> Path | None`
  - `resampler` is `"nearest"` or `"bilinear"`. The defaults match Satpy's precompute defaults: nearest
    `radius_of_influence=None`, `neighbours=1`, `mask=None`; bilinear `radius_of_influence=50000`, `neighbours=32`.
  - Old filename: reuse `future.resamplers.resampler.hash_resampler_geometries(src, dst, **kw)`, which matches
    `BaseResampler.get_hash`. Nearest kw = `{mask: None, radius_of_influence, neighbours: 1, epsilon}`; bilinear
    kw = `{radius_of_influence, neighbours: 32, epsilon}`. Prefix `nn_lut-` / `bil_lut-`.
  - Read the file with `xr.open_zarr` → dask arrays. Nearest keeps `valid_input_index` (cast to bool, as Satpy
    does) and `index_array`, and drops `valid_output_index`, which the future resampler doesn't use. Bilinear
    keeps all `BIL_COORDINATES`.
  - New key: for nearest, resolve `radius_of_influence=None` with the same `_compute_radius_of_influence` the
    future resampler uses. Bilinear uses `reduce_data=True`, because Satpy never passed it on.
  - Save with `description` = area descriptions + kwargs + `"converted_from": <old file name>`. Delete the old
    file only when `remove=True`, and only after a successful save. Return the new path, or `None` (and log it)
    if no file matched. Non-area pairs raise `ValueError`, since they can't be cached.
- `convert_legacy_caches(cache_dir, area_pairs, *, nearest_kwargs=({},), bilinear_kwargs=({},), remove=False)
  -> LegacyCacheConversion`
  - Builds the expected old filename for every pair × resampler × kwargs combination, matches them against the
    `nn_lut-*.zarr`/`bil_lut-*.zarr` files actually present, and converts the matches.
  - Returns a small dataclass: `converted: list[(old_path, new_path, resampler, kwargs)]` and
    `unmatched: list[Path]`, the leftover legacy files (e.g. from swaths, other kwargs, or areas not listed).
  - `area_pairs` is an iterable of `(source, target)` geometries; users get them from `load_area` etc.
- Both functions take an optional `dest_cache_dir` (default: `pyresample.config["cache_dir"]`), the same rule as
  the resamplers. `cache_dir` is where the legacy files are read from; it is often the same directory for Satpy
  users, which is fine since the new entries live under `resampling_luts/`.

## 3. Tests
- `pyresample/test/test_resamplers/test_nearest.py`, `pyresample/test/test_bilinear.py`: cache disabled by
  default; cached == uncached; warm cache is lazy (`assert_maximum_dask_computes(0)`) and skips the search
  (mock); masked / swath input not cached; a version-pinning test per resampler, modelled on
  `test_indices_are_pinned_to_cache_version` (`pyresample/test/test_gradient.py`).
- New `pyresample/test/test_utils/test_legacy_cache.py`, using the `create_test_area` fixture:
  - Bilinear legacy files are written with the real `XArrayBilinearResampler.save_resampling_info` to
    `BaseResampler._create_cache_filename(prefix="bil_lut-", ...)`. Nearest legacy files are written by a small
    helper that copies Satpy's `save_neighbour_info` layout (an xr.Dataset with `NN_COORDINATES` dims, from
    `XArrayResamplerNN.get_neighbour_info`).
  - After conversion, resampling with the cache enabled doesn't call the search and equals the uncached result.
    This also checks that the legacy and future default radius agree and that the index layouts are identical.
  - Unmatched file reported; `remove=True` deletes only converted files; missing file → `None`; batch over
    several pairs/kwargs.
- Docstrings are required in `test_utils/` (D102/D103 aren't exempt there).

## 4. Docs
- `docs/source/howtos/configuration.rst` "Cache Resampling LUTs": list nearest (future resampler) and bilinear.
- New short how-to `docs/source/howtos/convert_legacy_caches.rst` (linked from the howtos index) showing both
  helpers with `load_area`.
- AGENTS.md: update the `ZarrCache` users sentence.

## Files
`pyresample/_caching.py`, `pyresample/gradient/__init__.py` (helper move only),
`pyresample/future/resamplers/nearest.py`, `pyresample/bilinear/xarr.py`, new `pyresample/utils/legacy_cache.py`,
tests and docs above. `satpy_resample/` stays untracked reference material and isn't committed.

## Verification
- `pytest pyresample/test/test_utils/test_legacy_cache.py pyresample/test/test_bilinear.py
  pyresample/test/test_resamplers pyresample/test/test_gradient.py pyresample/test/test_caching.py`, then the
  full suite (only the 2 known `test_to_odc_geobox` failures allowed).
- `pre-commit run --files <changed files>`; `sphinx-build -b doctest docs/source <tmp>` for the new how-to.
- Manual end-to-end: create a legacy bilinear cache in a scratch dir, run `convert_legacy_caches`, and inspect
  `ZarrCache.get_attrs` of the new entry (`converted_from` present).
