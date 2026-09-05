# AGENTS.md

Orientation for AI agents working on Pyresample.

## What Pyresample is

Pyresample resamples geolocated ("georeferenced") image data — typically satellite imagery
— from one geographic geometry to another. It is the resampling engine behind
[Satpy](https://github.com/pytroll/satpy) as well as a standalone resampling library.
It works with numpy arrays, and in many places
with dask arrays and xarray `DataArray` objects. Three Cython extensions do the heavy
lifting for EWA and gradient search. LGPL-3.0-or-later, Python >= 3.12.

## Repo orientation

| Path | Notes                                                                                    |
| --- |------------------------------------------------------------------------------------------|
| `pyresample/geometry.py` | ~3150 lines. All the legacy geometry classes. The single most important file.            |
| `pyresample/area_config.py` | `create_area_def`, `load_area`, `parse_area_file`, YAML + legacy `.cfg` parsing, `dump`. |
| `pyresample/kd_tree.py` | Nearest-neighbour / gauss / custom resampling on pykdtree.                               |
| `pyresample/bilinear/` | `XArrayBilinearResampler` (dask) and `NumpyBilinearResampler`.                           |
| `pyresample/ewa/` | Elliptical Weighted Averaging + the `_ll2cr.pyx` / `_fornav.pyx` extensions.             |
| `pyresample/bucket/` | `BucketResampler` — per-target-pixel statistics.                                         |
| `pyresample/gradient/` | Gradient search + `_gradient_search.pyx`.                                                |
| `pyresample/resampler.py` | Legacy `BaseResampler` and `resample_blocks` (dask blockwise machinery).                 |
| `pyresample/future/` | The 2.0 staging area. See below — parts of it are already load-bearing.                  |
| `pyresample/slicer.py` | `create_slicer` / `AreaSlicer` / `SwathSlicer` — crop one geometry to another.           |
| `pyresample/utils/` | `proj4.py`, `cf.py` (CF/netCDF areas), `cartopy.py`, `rasterio.py`, `errors.py`.         |
| `pyresample/spherical.py` | Current spherical geometry primitives (`SCoordinate`, `CCordinate`, `Arc`, `SphPolygon`). |
| `pyresample/_config.py` | The donfig `config` object.                                                              |
| `pyresample/_caching.py` | On-disk JSON caching for geometry slices.                                                |
| `pyresample/test/` | ~1180 tests. Subpackages `test_geometry/`, `test_utils/`, `test_resamplers/`.            |

Legacy or effectively unmaintained, listed so you don't build on them: `image.py`,
`grid.py`, `spherical_geometry.py`, `geo_filter.py`, `_spatial_mp.py`, `_multi_proc.py`,
`data_reduce.py`, `boundary/legacy_boundary.py`, and the Basemap half of `plot.py`.

A working checkout often carries untracked scratch files (large `.tif`s, `profile_*.html`,
generated `.c`/`.cpp`/`.html` from Cython, etc). Ignore them, never commit them, and remember
they can pollute repo-wide greps.

## Geometry concepts

Full prose lives in `docs/source/concepts/{geolocated_data,projections,geometries,resampling}.rst`.
The short version:

- **`AreaDefinition`** — uniformly spaced pixels on a projected CRS. The only geometry that
  really understands projections. `area_extent` is `(lower_left_x, lower_left_y,
  upper_right_x, upper_right_y)` in *projection units* (metres for most CRSes, degrees for
  lon/lat CRSes) and describes the **outer edges of the corner pixels**, not their centers.
- **`SwathDefinition`** — irregularly spaced pixels given by 2D `lons`/`lats` arrays in
  degrees at pixel centers. A `crs` kwarg exists, but most operations still assume generic
  WGS84 lon/lat. Bounding polygons assume the outer rows/columns are the swath edge, which
  isn't always true.
- **`DynamicAreaDefinition`** — an area with some parameters left unspecified; `.freeze(lonlats)`
  fills them in from data. Not a `BaseDefinition` subclass (it derives from `object`), so
  `isinstance` checks against the geometry hierarchy will miss it. Its docstring holds the
  antimeridian caveat: freezing lon/lat data that crosses ±180 yields the smallest
  containing area, whose right-hand X extent can exceed 180.
- **`GridDefinition`** — lon/lat arrays that happen to be evenly spaced. It is a special
  case of `AreaDefinition`, is far more expensive in memory and CPU, and is slated for
  deprecation. Don't build anything new on it.
- **`StackedAreaDefinition`** — vertically stacked areas that share a CRS.
- **`CoordinateDefinition`** — base class for lon/lat-only geometries; shows up in resampler
  type hints as the non-area geometry type.

Since 1.8.0 geometry objects do **not** validate lon/lat. Longitudes must be in
`[-180, 180[` and latitudes in `[-90, 90]`; call `pyresample.utils.check_and_wrap()`
yourself if the input might not be.

## Ordering and naming traps

These cause silent wrong answers rather than exceptions, so check them every time:

- `.shape` is `(height, width)`, but the legacy `AreaDefinition` constructor takes
  `width, height`, and `.resolution`, `radius`, `center` and `upper_left_extent` are all
  `(x, y)`. The `from_*` classmethods take `shape=(height, width)`.
- `AreaDefinition.get_area_slices()` returns `(x_slice, y_slice)` while `__getitem__` takes
  `(y_slice, x_slice)`. `crop_around()` is the one place that bridges them correctly.
- `.area_extent` — "ll" means *lower left*. `.area_extent_ll` — "ll" means *lon/lat*.
- `.crs` is a property that re-parses `.crs_wkt` on **every access** (a `pyproj.CRS` is not
  thread-safe, so it is deliberately not cached). `area.crs is area.crs` is `False`, and
  touching `.crs` in a loop is expensive. The stored form is `.crs_wkt`.
- `get_lonlats` has different signatures per class: `BaseDefinition.get_lonlats(data_slice,
  chunks)` versus `AreaDefinition.get_lonlats(nprocs, data_slice, cache, dtype, chunks)`.
  Never call it positionally.
- `create_area_def` and every `AreaDefinition.from_*` classmethod can return a
  `DynamicAreaDefinition` when the inputs are incomplete, despite being classmethods on
  `AreaDefinition`.

## Resamplers: which interface to use

| Algorithm | Use | Accepts |
| --- | --- | --- |
| Nearest neighbour | `kd_tree.resample_nearest` / `kd_tree.XArrayResamplerNN` | numpy+masked / xarray-over-dask |
| Nearest (2.0 style) | `future.resamplers.KDTreeNearestXarrayResampler` | DataArray, dask, numpy (warns) |
| Bilinear | `bilinear.XArrayBilinearResampler` / `NumpyBilinearResampler` | xarray-over-dask / numpy |
| EWA | `ewa.DaskEWAResampler` | numpy, dask, or DataArray; source must be a swath |
| Bucket statistics | `bucket.BucketResampler` | dask arrays |
| Gradient search | `gradient.create_gradient_search_resampler` | DataArray-over-dask |
| Blockwise dask plumbing | `resampler.resample_blocks` | dask only; `chunk_size` and `dtype` required |

**Every resampler has a different calling convention.** Read the class before assuming
anything. Known divergences:

- `ewa.ll2cr`/`ewa.fornav` take `fill=`; everything else takes `fill_value=`.
- `fill_value` defaults differ: `0` (numpy kd_tree, `NumpyBilinearResampler.resample`),
  `np.nan` (`XArrayResamplerNN`, future nearest, bucket), `None` meaning dtype-dependent
  (EWA, gradient, `resample_blocks`, `XArrayBilinearResampler`).
- In `BucketResampler`, `fill_value` describes the **input** invalid-data sentinel, not the
  output fill; the empty-bin value is a separate `empty_bucket_value` on `get_sum` only.
  `BucketResampler` has no `resample()` at all — only `get_sum`, `get_count`, `get_average`,
  `get_min`, `get_max`, `get_abs_max`, `get_fractions`.
- The precompute/compute split is named three different ways: `precompute`/`compute`
  (`BaseResampler`, EWA, gradient), `get_neighbour_info`/`get_sample_from_neighbour_info`
  (kd_tree), `get_bil_info`/`get_sample_from_bil_info` (bilinear).
- There are three unrelated caching mechanisms sharing similar names: `cache_dir=` on
  `BaseResampler` (only bilinear really uses it; `DaskEWAResampler.precompute` accepts it
  and warns that it is ignored), bilinear's own zarr `save_resampling_info`/
  `load_resampling_info`, and `pyresample.config["cache_dir"]`, which is used only by
  `_caching.py` for geometry slices.
- Only `nearest` is registered in the `pyresample.resamplers` entry point group, so
  `future.list_resamplers()` does not see bilinear, EWA, gradient or bucket.

## Deprecated — don't write new code against these

`pyresample.image` (`ImageContainer*`), `pyresample.grid`, `pyresample.spherical_geometry`
(warns on import, but is still load-bearing for `BaseDefinition.overlaps`/`intersection`/
`__contains__`), `GridDefinition`, `geo_filter`, `nprocs`/`segments` and the `_spatial_mp`
multiprocessing path (superseded by dask).

Renamed members: `get_lonlats_dask`/`get_proj_vectors_dask`/`get_proj_coords_dask` →
pass `chunks=`; `proj4_string` → `proj_str`; `x_size`/`y_size` → `width`/`height`;
`AreaDefinition.name` → `.description`; `frequency=` → `vertices_per_side=`;
`lonlat2colrow`/`get_xy_from_lonlat`/`get_xy_from_proj_coords` →
`get_array_indices_from_*`; `create_areas_def` → `dump`; `resample_bilinear`/`get_bil_info`/
`get_sample_from_bil_info` → the bilinear resampler classes; `XArrayResamplerBilinear`/
`NumpyResamplerBilinear` → `XArrayBilinearResampler`/`NumpyBilinearResampler`;
`GradientSearchResampler` → `create_gradient_search_resampler`; `area_def2basemap` and the
Basemap code in `plot.py` → `AreaDefinition.to_cartopy_crs()`; `AreaDefBoundary` →
`.boundary()`; `get_geostationary_bounding_box` → `..._in_lonlats`.

**These shims are intentional.** Do not "clean up" a deprecation, a legacy code path, or an
odd-looking backwards-compatibility branch unless the task explicitly asks for it.

**The legacy interfaces are feature-frozen.** They are kept working for existing users, not
grown. New functionality does not get added to `ImageContainer` and friends, to
`pyresample.grid`, to `resample_bilinear`/`get_bil_info`, to the `nprocs`/`segments`
multiprocessing path, or to any other interface in the list above — even when doing so
looks like the smallest possible diff. The only changes those interfaces should receive are
bug fixes, deprecation warnings, and helpers that make it easier for a user to move to the
supported interface. If a task seems to call for a new feature on a legacy class, implement
it on the modern equivalent (or in `pyresample/future/`) and, if it helps, add a transition
path from the old one; if that isn't possible, say so rather than extending the legacy API.

## Configuration

`pyresample.config` is a [donfig](https://donfig.readthedocs.io/) object defined in
`pyresample/_config.py`. Keys and defaults:

- `cache_dir` — platformdirs user cache dir; used only for geometry-slice caching.
- `cache_geometry_slices` — `False`.
- `features.future_geometries` — `False`.

Environment variables are `PYRESAMPLE_` prefixed with `__` for nesting
(`PYRESAMPLE_FEATURES__FUTURE_GEOMETRIES`); in Python the key is `"features.future_geometries"`.
`pyresample.config.set(...)` works as a context manager. YAML/env settings must be in place
before `pyresample` is imported. Full details in `docs/source/howtos/configuration.rst`.

Two things worth knowing: `features.future_geometries` currently only changes what class
`create_area_def` and `get_area_def_from_raster` return — it does **not** swap the
top-level `pyresample.AreaDefinition`. And `CHUNK_SIZE` is not in donfig at all; it is a
module-level constant read from the `PYTROLL_CHUNK_SIZE` env var at import time and cannot
be changed at runtime.

## Pyresample 2.0

Read `docs/source/roadmap.rst` before touching anything in `pyresample/future/`. Progress
is tracked on the [v2.0 milestone](https://github.com/pytroll/pyresample/milestone/3).

The direction, in one paragraph: separate the "numbers" from the metadata so geometries no
longer require a name or description; build geometries through classmethods rather than a
long positional constructor; collapse today's five different resampler calling conventions
into a single `Resampler` interface with `precompute()` and `resample()`; add a resampler
registry plus a `create_resampler` factory that third parties can plug into; move caching
into pluggable Cache objects; and expose a separate "Index" interface for the many users
who consume `get_neighbour_info` output directly. Explicitly *not* in scope for 2.0: new
resampling algorithms, and vertical/higher-dimensional resampling.

What exists in-tree today: `pyresample/future/` holds the abstract `Resampler` base class,
the registry (`register_resampler`, `create_resampler`, `list_resamplers`, entry point group
`pyresample.resamplers`), `KDTreeNearestXarrayResampler`, thin `AreaDefinition`/
`SwathDefinition` subclasses carrying an `attrs` dict, and new spherical `SPoint`/`SMultiPoint`
classes. Note that `future/` is **not** optional experimental code you can ignore — parts
of it are already load-bearing for the legacy API (`future/geometry/_subset.py` implements
`AreaDefinition.get_area_slices`, `future/resamplers/nearest.py` backs `kd_tree`).

Much of the 2.0 design was prototyped in Satpy and is meant to migrate here — see
`satpy/resample/base.py`, which still carries literal `# TODO: move this to pyresample`
markers on `prepare_resampler` and `resample` (pyresample issue #194).

The roadmap is aspirational and dated in places. For example its
`AreaDefinition.from_extent_shape` example does not exist; the real classmethods are
`from_extent`, `from_circle`, `from_area_of_interest`, `from_ul_corner`, `from_epsg` and
`from_cf`.

## Development workflow

**Build.** hatchling + hatch-vcs + hatch-cython, all configured in `pyproject.toml`. There
is no `setup.py`, `setup.cfg` or `meson.build`. The version comes from git tags, so a
checkout without tags reports a bogus version. Install with `pip install -e . --no-deps`
(what CI does). There are three Cython extensions — `ewa/_ll2cr.pyx`, `ewa/_fornav.pyx`
(C++) and `gradient/_gradient_search.pyx`. **Editing a `.pyx` requires reinstalling or
rebuilding**; the existing `.so` will not pick up the change.

**Tests.** `pytest pyresample/test` — about 1180 tests in roughly two minutes. There is no
pytest config section, so no registered markers or default options. Two tests already fail
on a clean `main` because of odc-geo version drift:
`test_geometry/test_area.py::TestAreaDefinition::test_to_odc_geobox[AreaDefinition]` and
`[LegacyAreaDefinition]`. Don't chase them.

**Test conventions.** `pyresample/test/conftest.py` provides `create_test_area` and
`create_test_swath` factory fixtures that transparently run a test against **both** the
legacy and the future geometry classes (hence the `[AreaDefinition]` / `[LegacyAreaDefinition]`
suffixes on test ids). Use them for new geometry tests unless the test genuinely needs one
specific class. `reset_pyresample_config` is autouse and pins the config defaults.
`pyresample/test/utils.py` has `assert_maximum_dask_computes()` for asserting laziness,
`create_test_longitude`/`create_test_latitude`, `friendly_crs_equal`, and
`assert_future_geometry`.

**Lint.** pre-commit runs ruff (linter only — there is **no** formatter), isort
(`profile=black`), mypy, bandit, and whitespace fixers. Line length is **120**. Ruff selects
`E,W,B,D,T10,C90,NPY`, so **docstrings are required** on modules, classes and public
functions (`D107` is exempt; `pyresample/test/*.py` exempts `D102`/`D103`, but that glob
does *not* cover the `test_geometry/`, `test_utils/` etc. subdirectories). The configured
docstring convention is google, while Sphinx is set up for numpydoc — existing code is
genuinely mixed, so **match the surrounding file**. Do not reorder the imports inside the
`# isort: off` / `# isort: on` block in `pyresample/__init__.py`; the order avoids a
circular import.

**Docs.** Sphinx, organized by [Diátaxis](https://diataxis.fr/): `concepts/`
(no code examples), `tutorials/` (end-to-end with provided data), `howtos/` (short, exact,
single-feature) and `reference/` (API). CI runs `sphinx-build -b doctest`, and Read the
Docs has `fail_on_warning: true`, so changing a public signature can break the docs build.
Pages under `docs/source/api/` are auto-generated — don't hand-write them.

**CI.** `.github/workflows/ci.yaml` builds a conda env from
`continuous_integration/environment.yaml` and tests Python 3.12/3.13/3.14 on Linux, macOS
and Windows. `deploy.yaml` builds wheels with cibuildwheel.

## Conventions for agents

- Prefer `create_area_def` or the `from_*` classmethods over the seven-positional-argument
  `AreaDefinition(...)` constructor.
- Prefer `.crs` / `.crs_wkt` over `.proj_dict` / `.proj_str` / `.proj4_string`.
- New features go on the modern interfaces, never on the legacy ones. When extending a
  resampler, check whether the change belongs in `pyresample/future/` under the new
  `Resampler` base class rather than being bolted onto a legacy class.
- Keep new public API dask- and xarray-friendly and lazy; guard laziness in tests with
  `assert_maximum_dask_computes`.
- Don't commit the untracked scratch files that accumulate in a working checkout.
