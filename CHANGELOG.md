## Version 1.30.0 (2024/08/28)

### Pull Requests Merged

#### Bugs fixed

* [PR 616](https://github.com/pytroll/pyresample/pull/616) - Update cibuildwheel to fix setuptools 74.0.0 compatibility
* [PR 614](https://github.com/pytroll/pyresample/pull/614) - Fix geocentric_resolution compatibility with numpy 2.1.0
* [PR 613](https://github.com/pytroll/pyresample/pull/613) - Switch on pytest-lazy-fixtures

#### Features added

* [PR 616](https://github.com/pytroll/pyresample/pull/616) - Update cibuildwheel to fix setuptools 74.0.0 compatibility
* [PR 615](https://github.com/pytroll/pyresample/pull/615) - Allow overriding area repr map section HTML

In this release 5 pull requests were closed.


## Version 1.29.0 (2024/07/31)

### Issues Closed

* [Issue 609](https://github.com/pytroll/pyresample/issues/609) - Error in SwathDefinition html representation if lon/lat arrays are dask arrays ([PR 610](https://github.com/pytroll/pyresample/pull/610) by [@BENR0](https://github.com/BENR0))
* [Issue 354](https://github.com/pytroll/pyresample/issues/354) - `get_sum` not matched with `bucket_sum`

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 610](https://github.com/pytroll/pyresample/pull/610) - Fix SwathDefinition html representation error when lons/lats 1D ([609](https://github.com/pytroll/pyresample/issues/609))
* [PR 601](https://github.com/pytroll/pyresample/pull/601) - Fix spill over of ocean/and features in cartopy plots in case of geostationary full disc plot.
* [PR 596](https://github.com/pytroll/pyresample/pull/596) - Fix AreaDefinition array index methods mishandling outer edge values ([691](https://github.com/ssec/polar2grid/issues/691))

#### Features added

* [PR 602](https://github.com/pytroll/pyresample/pull/602) - Add support for `fill_value` and `set_empty_bucket_to` in BucketResampler `get_sum`

In this release 4 pull requests were closed.


## Version 1.28.4 (2024/07/01)

### Pull Requests Merged

#### Bugs fixed

* [PR 603](https://github.com/pytroll/pyresample/pull/603) - Add Python 3.12 wheels and bump pypa/gh-action-pypi-publish from 1.8.14 to 1.9.0

#### Features added

* [PR 598](https://github.com/pytroll/pyresample/pull/598) - Add NPY to ruff rules

In this release 2 pull requests were closed.


## Version 1.28.3 (2024/04/15)

### Issues Closed

* [Issue 587](https://github.com/pytroll/pyresample/issues/587) - Resampling GOES mesoscale data to my area gives blank data

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 594](https://github.com/pytroll/pyresample/pull/594) - Build against numpy 2.0rc1

In this release 1 pull request was closed.


## Version 1.28.2 (2024/02/29)

### Issues Closed

* [Issue 539](https://github.com/pytroll/pyresample/issues/539) - Compatibility with libproj v9.3

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 586](https://github.com/pytroll/pyresample/pull/586) - Include package data

#### Features added

* [PR 578](https://github.com/pytroll/pyresample/pull/578) - Add attrs to future swath definition

In this release 2 pull requests were closed.


## Version 1.28.1 (2024/02/15)

### Pull Requests Merged

#### Bugs fixed

* [PR 585](https://github.com/pytroll/pyresample/pull/585) - Fix optimize_projection handling in YAML parsing
* [PR 584](https://github.com/pytroll/pyresample/pull/584) - Fix other numpy 2 incompatibilities

In this release 2 pull requests were closed.


## Version 1.28.0 (2024/02/14)

### Issues Closed

* [Issue 570](https://github.com/pytroll/pyresample/issues/570) - errors in area definition should not be silently ignored ([PR 577](https://github.com/pytroll/pyresample/pull/577) by [@djhoese](https://github.com/djhoese))
* [Issue 547](https://github.com/pytroll/pyresample/issues/547) - How should this warning be addressed? ([PR 548](https://github.com/pytroll/pyresample/pull/548) by [@djhoese](https://github.com/djhoese))
* [Issue 537](https://github.com/pytroll/pyresample/issues/537) - Upgrade to Cython 3.0 and check annotations ([PR 582](https://github.com/pytroll/pyresample/pull/582) by [@djhoese](https://github.com/djhoese))
* [Issue 527](https://github.com/pytroll/pyresample/issues/527) - area definition for a rotated pole coordinate system ([PR 532](https://github.com/pytroll/pyresample/pull/532) by [@djhoese](https://github.com/djhoese))

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 577](https://github.com/pytroll/pyresample/pull/577) - Fix area definition YAML not warning on typos ([570](https://github.com/pytroll/pyresample/issues/570))
* [PR 573](https://github.com/pytroll/pyresample/pull/573) - Switch test area fixtures to session-scoped
* [PR 556](https://github.com/pytroll/pyresample/pull/556) - Add missing meta keyword arguments on all dask map_blocks calls
* [PR 551](https://github.com/pytroll/pyresample/pull/551) - Fix shortcut for prime meridian checks
* [PR 536](https://github.com/pytroll/pyresample/pull/536) - Use pyproj TransformDirection enum for better performance
* [PR 532](https://github.com/pytroll/pyresample/pull/532) - Migrate usages of pyproj Proj to pyproj Transformer ([527](https://github.com/pytroll/pyresample/issues/527))
* [PR 526](https://github.com/pytroll/pyresample/pull/526) - Replace and deprecate frequency arg for bbox methods

#### Features added

* [PR 583](https://github.com/pytroll/pyresample/pull/583) - Build wheels against numpy 2
* [PR 582](https://github.com/pytroll/pyresample/pull/582) - Fix Cython 3 and Numpy 2 compatibility ([537](https://github.com/pytroll/pyresample/issues/537))
* [PR 572](https://github.com/pytroll/pyresample/pull/572) - Suppress PROJ4 warning about losing projection by using `to_cf()` instead of `to_dict()`
* [PR 557](https://github.com/pytroll/pyresample/pull/557) - Add more pre-commit checks (mccabe, bandit, mypy, etc)
* [PR 553](https://github.com/pytroll/pyresample/pull/553) - Add optional caching to AreaDefinition.get_area_slices
* [PR 548](https://github.com/pytroll/pyresample/pull/548) - Remove unnecessary proj4 argument parsing from get_area_def ([547](https://github.com/pytroll/pyresample/issues/547))
* [PR 545](https://github.com/pytroll/pyresample/pull/545) - Convert AreaDefinitions to odc geoboxes
* [PR 542](https://github.com/pytroll/pyresample/pull/542) - Switch to cibuildwheel for wheel building
* [PR 533](https://github.com/pytroll/pyresample/pull/533) - Replace width/height in future AreaDefinition with "shape" argument
* [PR 522](https://github.com/pytroll/pyresample/pull/522) - Handle value-less parameters in `proj4_dict_to_str`
* [PR 519](https://github.com/pytroll/pyresample/pull/519) - Add builtin 'config' object and 'features.future_geometry' toggle
* [PR 516](https://github.com/pytroll/pyresample/pull/516) - Allow cropping non-geos areas
* [PR 450](https://github.com/pytroll/pyresample/pull/450) - Area definition html representation for Jupyter notebooks

#### Documentation changes

* [PR 519](https://github.com/pytroll/pyresample/pull/519) - Add builtin 'config' object and 'features.future_geometry' toggle
* [PR 450](https://github.com/pytroll/pyresample/pull/450) - Area definition html representation for Jupyter notebooks

#### Backward incompatible changes

* [PR 522](https://github.com/pytroll/pyresample/pull/522) - Handle value-less parameters in `proj4_dict_to_str`

#### Refactoring

* [PR 566](https://github.com/pytroll/pyresample/pull/566) - Refactor area boundary sides retrieval with `_geographic_sides` and `_projection_sides` methods
* [PR 565](https://github.com/pytroll/pyresample/pull/565) - Move legacy boundary to boundary directory
* [PR 564](https://github.com/pytroll/pyresample/pull/564) - Refactor ``test_area`` and move boundary related tests to ``test_area_boundary``
* [PR 563](https://github.com/pytroll/pyresample/pull/563) - Remove `__file__` usage in test units for `test_files` path

In this release 27 pull requests were closed.


## Version 1.27.1 (2023/06/21)

### Issues Closed

* [Issue 517](https://github.com/pytroll/pyresample/issues/517) - EWA resampling in 1.27 slows down four times than 1.26.1 ([PR 520](https://github.com/pytroll/pyresample/pull/520) by [@djhoese](https://github.com/djhoese))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 524](https://github.com/pytroll/pyresample/pull/524) - Preserve get_area_slices behavior when area to cover has an invalid boundary
* [PR 523](https://github.com/pytroll/pyresample/pull/523) - Fix DynamicAreaDefinition not preserving user's requested resolution ([517](https://github.com/pytroll/pyresample/issues/517))
* [PR 520](https://github.com/pytroll/pyresample/pull/520) - Fix performance regression in base resampler class when comparing geometries ([517](https://github.com/pytroll/pyresample/issues/517))

#### Documentation changes

* [PR 518](https://github.com/pytroll/pyresample/pull/518) - Add configuration for readthedocs to fail on warnings

In this release 4 pull requests were closed.


## Version 1.27.0 (2023/05/17)

### Issues Closed

* [Issue 507](https://github.com/pytroll/pyresample/issues/507) - `gradient_search` fails when resampling Himawari data ([PR 508](https://github.com/pytroll/pyresample/pull/508) by [@mraspaud](https://github.com/mraspaud))
* [Issue 504](https://github.com/pytroll/pyresample/issues/504) - `get_neighbour_info` slows down significantly when working with large target rasters using many segments ([PR 505](https://github.com/pytroll/pyresample/pull/505) by [@SwamyDev](https://github.com/SwamyDev))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 513](https://github.com/pytroll/pyresample/pull/513) - Remove more warnings encountered during tests
* [PR 512](https://github.com/pytroll/pyresample/pull/512) - Ignore pyproj to_proj4 warning when converting an AreaDefinition to a string
* [PR 508](https://github.com/pytroll/pyresample/pull/508) - Fix gradient resampling to same area not working ([507](https://github.com/pytroll/pyresample/issues/507))
* [PR 502](https://github.com/pytroll/pyresample/pull/502) - Add stacklevel to all pyresample warnings
* [PR 498](https://github.com/pytroll/pyresample/pull/498) - Fix data type handling (complex) in nearest neighbor resampling

#### Features added

* [PR 505](https://github.com/pytroll/pyresample/pull/505) - Add pre-allocation option to `get_neighbour_info` to improve performance on large raster data ([504](https://github.com/pytroll/pyresample/issues/504))
* [PR 499](https://github.com/pytroll/pyresample/pull/499) - Drop Python 3.8 support
* [PR 496](https://github.com/pytroll/pyresample/pull/496) - Deprecate AreaDefinition 'rotation' argument
* [PR 464](https://github.com/pytroll/pyresample/pull/464) - Add optional metadata to Pyresample 2.0 AreaDefinition

#### Documentation changes

* [PR 515](https://github.com/pytroll/pyresample/pull/515) - Remove python versions tested note in documentation
* [PR 501](https://github.com/pytroll/pyresample/pull/501) - Couple of small typos found in documentation.
* [PR 434](https://github.com/pytroll/pyresample/pull/434) - Add initial restructuring of sphinx docs

#### Backward incompatible changes

* [PR 499](https://github.com/pytroll/pyresample/pull/499) - Drop Python 3.8 support
* [PR 496](https://github.com/pytroll/pyresample/pull/496) - Deprecate AreaDefinition 'rotation' argument

In this release 14 pull requests were closed.


## Version 1.26.1 (2023/02/07)

### Issues Closed

* [Issue 497](https://github.com/pytroll/pyresample/issues/497) - resampling fails with `ValueError` if not padding data
* [Issue 492](https://github.com/pytroll/pyresample/issues/492) - Infinite values in geostationary bounding box crash intersection function ([PR 493](https://github.com/pytroll/pyresample/pull/493) by [@mraspaud](https://github.com/mraspaud))
* [Issue 486](https://github.com/pytroll/pyresample/issues/486) - BUG: Import fails due to misconfigured setup.py ([PR 487](https://github.com/pytroll/pyresample/pull/487) by [@bzah](https://github.com/bzah))
* [Issue 484](https://github.com/pytroll/pyresample/issues/484) - Fails to build with Shapely 2.0 ([PR 485](https://github.com/pytroll/pyresample/pull/485) by [@sebastic](https://github.com/sebastic))
* [Issue 481](https://github.com/pytroll/pyresample/issues/481) - Intermittent failures on 32bit architectures
* [Issue 448](https://github.com/pytroll/pyresample/issues/448) - Release 1.25.1 missing usual GPG signature

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 493](https://github.com/pytroll/pyresample/pull/493) - Fix geostationary bbox having inf values ([2368](https://github.com/pytroll/satpy/issues/2368), [492](https://github.com/pytroll/pyresample/issues/492))
* [PR 487](https://github.com/pytroll/pyresample/pull/487) - FIX: Update pyproj min version to 3.0.0 ([486](https://github.com/pytroll/pyresample/issues/486))
* [PR 485](https://github.com/pytroll/pyresample/pull/485) - Tune tolerance to fix test failures with PROJ 9.1.1. ([484](https://github.com/pytroll/pyresample/issues/484))
* [PR 482](https://github.com/pytroll/pyresample/pull/482) - Fix intermittent EWA test failures

In this release 4 pull requests were closed.


## Version 1.26.0.post0 (2022/11/24)

### Pull Requests Merged

#### Features added

* [PR 480](https://github.com/pytroll/pyresample/pull/480) - Add Python 3.11 to wheel building

In this release 1 pull request was closed.


## Version 1.26.0 (2022/11/24)

### Issues Closed

* [Issue 474](https://github.com/pytroll/pyresample/issues/474) - get_geostationary_bounding_box* contains duplicated vertices at the equator  ([PR 475](https://github.com/pytroll/pyresample/pull/475) by [@ghiggi](https://github.com/ghiggi))
* [Issue 457](https://github.com/pytroll/pyresample/issues/457) - Pyresample 1.25.1 create_area_def return wrong lons with the .get_lonlats()
* [Issue 453](https://github.com/pytroll/pyresample/issues/453) - Import Error using XArrayBilinearResampler missing failed import of dask ([PR 454](https://github.com/pytroll/pyresample/pull/454) by [@benjaminesse](https://github.com/benjaminesse))
* [Issue 445](https://github.com/pytroll/pyresample/issues/445) - Release GIL in gradient search resampling ([PR 455](https://github.com/pytroll/pyresample/pull/455) by [@mraspaud](https://github.com/mraspaud))
* [Issue 439](https://github.com/pytroll/pyresample/issues/439) - SwathDefinition.update_hash() raise error after slicing the swath object ([PR 462](https://github.com/pytroll/pyresample/pull/462) by [@mraspaud](https://github.com/mraspaud))

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 479](https://github.com/pytroll/pyresample/pull/479) - Fix bbox creation for SwathDefinitions with NaNs
* [PR 475](https://github.com/pytroll/pyresample/pull/475) - Fix for duplicate coordinates in bbox_lonlats for geostationary area  ([474](https://github.com/pytroll/pyresample/issues/474), [474](https://github.com/pytroll/pyresample/issues/474))
* [PR 463](https://github.com/pytroll/pyresample/pull/463) - Fix EWA default for 'weight_delta_max' to match docstring
* [PR 462](https://github.com/pytroll/pyresample/pull/462) - Fix hashing of definitions for non contiguous arrays ([439](https://github.com/pytroll/pyresample/issues/439))
* [PR 438](https://github.com/pytroll/pyresample/pull/438) - Fix using cached LUTs in bilinear resampler

#### Features added

* [PR 473](https://github.com/pytroll/pyresample/pull/473) - Add boundary method to AreaDefinition and SwathDefinition
* [PR 465](https://github.com/pytroll/pyresample/pull/465) - [Future Spherical Class] Add SPoint and SMultiPoint
* [PR 455](https://github.com/pytroll/pyresample/pull/455) - Use memoryviews and allow nogil in gradient search ([445](https://github.com/pytroll/pyresample/issues/445))
* [PR 451](https://github.com/pytroll/pyresample/pull/451) - Refactor the area loading internal function

#### Documentation changes

* [PR 454](https://github.com/pytroll/pyresample/pull/454) - Fix import warning in bilinear resampler to mention dask ([453](https://github.com/pytroll/pyresample/issues/453))

In this release 10 pull requests were closed.


## Version 1.25.1 (2022/08/02)

### Pull Requests Merged

#### Bugs fixed

* [PR 447](https://github.com/pytroll/pyresample/pull/447) - Fix handling of lon/lat coordinates on CRS with prime meridian != 0

In this release 1 pull request was closed.


## Version 1.25.0 (2022/07/29)

### Issues Closed

* [Issue 428](https://github.com/pytroll/pyresample/issues/428) - Add more flexible antimeridian handling to DynamicAreaDefinition ([PR 431](https://github.com/pytroll/pyresample/pull/431) by [@djhoese](https://github.com/djhoese))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 446](https://github.com/pytroll/pyresample/pull/446) - Fix incorrect extents for DynamicAreaDefinition with 'modify_crs' antimeridian mode

#### Features added

* [PR 431](https://github.com/pytroll/pyresample/pull/431) - Add 'antimeridian_mode' to DynamicAreaDefinition ([428](https://github.com/pytroll/pyresample/issues/428))

In this release 2 pull requests were closed.


## Version 1.24.1 (2022/07/06)

### Pull Requests Merged

#### Bugs fixed

* [PR 441](https://github.com/pytroll/pyresample/pull/441) - Fix infinite coordinates failing to be equal

#### Features added

* [PR 443](https://github.com/pytroll/pyresample/pull/443) - Remove Python 3.7 support

#### Backward incompatible changes

* [PR 443](https://github.com/pytroll/pyresample/pull/443) - Remove Python 3.7 support

In this release 3 pull requests were closed.


## Version 1.24.0 (2022/07/06)

### Issues Closed

* [Issue 417](https://github.com/pytroll/pyresample/issues/417) - Add get_abs_max (and get_abs_min) to BucketResampler ([PR 418](https://github.com/pytroll/pyresample/pull/418) by [@gerritholl](https://github.com/gerritholl))
* [Issue 316](https://github.com/pytroll/pyresample/issues/316) - Upgrade to pyresample 1.17.0 causes IndexError with one-dimensional data ([PR 324](https://github.com/pytroll/pyresample/pull/324) by [@pnuu](https://github.com/pnuu))
* [Issue 171](https://github.com/pytroll/pyresample/issues/171) - Update AreaDefinition to accept pyproj CRS objects and WKT

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 324](https://github.com/pytroll/pyresample/pull/324) - Fix bilinear resampler for 1D data ([316](https://github.com/pytroll/pyresample/issues/316))

#### Features added

* [PR 435](https://github.com/pytroll/pyresample/pull/435) - Fix SwathDefinition causing unnecessary dask computes when used as a dict key
* [PR 418](https://github.com/pytroll/pyresample/pull/418) - Implement get_abs_max on BucketResampler ([417](https://github.com/pytroll/pyresample/issues/417))
* [PR 368](https://github.com/pytroll/pyresample/pull/368) - Speed up Bucket `get_min` and `get_max`
* [PR 341](https://github.com/pytroll/pyresample/pull/341) - Dask resampler and gradient search overhaul

#### Documentation changes

* [PR 429](https://github.com/pytroll/pyresample/pull/429) - Improve docs for dump and load_area_from_string
* [PR 427](https://github.com/pytroll/pyresample/pull/427) - Add Cython classifier to package metadata

In this release 7 pull requests were closed.


## Version 1.23.0 (2022/03/21)

### Issues Closed

* [Issue 425](https://github.com/pytroll/pyresample/issues/425) - Pyresample/geometry.py resampling error related to dask.
* [Issue 422](https://github.com/pytroll/pyresample/issues/422) - Cannot resample with `bilinear` from lat/lon grid onto MSG full disk ([PR 423](https://github.com/pytroll/pyresample/pull/423) by [@pnuu](https://github.com/pnuu))
* [Issue 416](https://github.com/pytroll/pyresample/issues/416) -  Unexpected results resampling Lambert Conformal to PlateCarree: pyresample or cartopy problem?

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 426](https://github.com/pytroll/pyresample/pull/426) - Fix EWA resampling not ignoring fill values with maximum_weight_mode
* [PR 424](https://github.com/pytroll/pyresample/pull/424) - Fix DynamicAreaDefinition resolution handling for incomplete projection definitions
* [PR 423](https://github.com/pytroll/pyresample/pull/423) - Fix bilinear resampling to areas with invalid coordinates ([422](https://github.com/pytroll/pyresample/issues/422))
* [PR 421](https://github.com/pytroll/pyresample/pull/421) - Fix inplace modification occuring in Arc.intersections
* [PR 414](https://github.com/pytroll/pyresample/pull/414) - Fix gradient search for single band data

#### Features added

* [PR 415](https://github.com/pytroll/pyresample/pull/415) - Update AreaDefinition equality to use pyproj CRS
* [PR 406](https://github.com/pytroll/pyresample/pull/406) - Change tested Python versions to 3.8, 3.9 and 3.10

#### Backward incompatible changes

* [PR 415](https://github.com/pytroll/pyresample/pull/415) - Update AreaDefinition equality to use pyproj CRS

In this release 8 pull requests were closed.


## Version 1.22.3 (2021/12/07)

### Issues Closed

* [Issue 375](https://github.com/pytroll/pyresample/issues/375) - Importing pyresample without having Xarray and/or zarray raises UserWarning ([PR 400](https://github.com/pytroll/pyresample/pull/400) by [@yunjunz](https://github.com/yunjunz))
* [Issue 318](https://github.com/pytroll/pyresample/issues/318) - Add fill_value keyword argument for AreaDefinition.get_lonlats
* [Issue 231](https://github.com/pytroll/pyresample/issues/231) - Copyright notice out of date ([PR 403](https://github.com/pytroll/pyresample/pull/403) by [@gerritholl](https://github.com/gerritholl))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 404](https://github.com/pytroll/pyresample/pull/404) - Fix dask EWA code not creating unique dask task names for different target areas
* [PR 400](https://github.com/pytroll/pyresample/pull/400) - Move bilinear import to avoid unnecessary warning ([375](https://github.com/pytroll/pyresample/issues/375))
* [PR 399](https://github.com/pytroll/pyresample/pull/399) - Fix deprecated numpy data type usage in bilinear resampling

#### Documentation changes

* [PR 403](https://github.com/pytroll/pyresample/pull/403) - Update copyright note in documentation ([231](https://github.com/pytroll/pyresample/issues/231))

In this release 4 pull requests were closed.


## Version 1.22.2 (2021/12/03)

### Pull Requests Merged

#### Features added

* [PR 401](https://github.com/pytroll/pyresample/pull/401) - Optimize AreaDefinition.get_proj_coords when requesting dask arrays ([1902](https://github.com/pytroll/satpy/issues/1902))

In this release 1 pull request was closed.


## Version 1.22.1 (2021/11/18)

### Issues Closed

* [Issue 390](https://github.com/pytroll/pyresample/issues/390) - What units does SphPolygon.area return?

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 398](https://github.com/pytroll/pyresample/pull/398) - Fix EWA resampling when input data is larger than the output area
* [PR 389](https://github.com/pytroll/pyresample/pull/389) - Fix SwathDefinition get_bbox_lonlats returning counter-clockwise coordinates

#### Features added

* [PR 396](https://github.com/pytroll/pyresample/pull/396) - Add Python 3.9 to CI runs and use it for the experimental run
* [PR 395](https://github.com/pytroll/pyresample/pull/395) - Replace depracated Numpy dtypes

#### Documentation changes

* [PR 388](https://github.com/pytroll/pyresample/pull/388) - Fix indentation on geometry utils page

In this release 5 pull requests were closed.


## Version 1.22.0 (2021/10/25)

### Issues Closed

* [Issue 384](https://github.com/pytroll/pyresample/issues/384) - Inconsistent SphPolygon intersection behavior ([PR 385](https://github.com/pytroll/pyresample/pull/385) by [@djhoese](https://github.com/djhoese))
* [Issue 353](https://github.com/pytroll/pyresample/issues/353) - cut'n'paste error for `codecov.yml`?

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 386](https://github.com/pytroll/pyresample/pull/386) - Fix geocentric_resolution method not working for lat/lon projections
* [PR 385](https://github.com/pytroll/pyresample/pull/385) - Fix SphPolygon producing unexpected results for 32-bit float coordinates ([384](https://github.com/pytroll/pyresample/issues/384))
* [PR 383](https://github.com/pytroll/pyresample/pull/383) - Fix AreaDefinition dumping when extents have Numpy values
* [PR 378](https://github.com/pytroll/pyresample/pull/378) - Fix compatibility with cartopy 0.20.0+

#### Features added

* [PR 379](https://github.com/pytroll/pyresample/pull/379) - Define new Resampler base class, nearest neighbor class, and resampler registry

#### Documentation changes

* [PR 380](https://github.com/pytroll/pyresample/pull/380) - Add pre-commit running to PRs and add isort

In this release 6 pull requests were closed.


## Version 1.21.1 (2021/09/17)

### Issues Closed

* [Issue 374](https://github.com/pytroll/pyresample/issues/374) - Geographic EWA projection for swaths crossing the anti-meridian omits values for 90 ≤ longitude (degrees east) ≤ 180. ([PR 376](https://github.com/pytroll/pyresample/pull/376))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 377](https://github.com/pytroll/pyresample/pull/377) - Remove unnecessary dask computation in 'nearest' resampler
* [PR 376](https://github.com/pytroll/pyresample/pull/376) - Update EWA ll2cr_static to handle swaths crossing the anti-meridian. ([374](https://github.com/pytroll/pyresample/issues/374))

In this release 2 pull requests were closed.


## Version 1.21.0 (2021/08/19)

### Pull Requests Merged

#### Bugs fixed

* [PR 370](https://github.com/pytroll/pyresample/pull/370) - Fix dask ewa issues with newer versions of dask

#### Features added

* [PR 347](https://github.com/pytroll/pyresample/pull/347) - Add spherical geometry support for deriving total/common area coverage of several satellite overpasses

#### Documentation changes

* [PR 373](https://github.com/pytroll/pyresample/pull/373) - Add initial draft of a Roadmap page

In this release 3 pull requests were closed.


## Version 1.20.0 (2021/06/04)

### Issues Closed

* [Issue 365](https://github.com/pytroll/pyresample/issues/365) - `get_proj_coords` result of Satpy is different from that of `load_cf_area`
* [Issue 361](https://github.com/pytroll/pyresample/issues/361) - __version__ in pip wheel  is mangled ([PR 363](https://github.com/pytroll/pyresample/pull/363))
* [Issue 355](https://github.com/pytroll/pyresample/issues/355) - Handle X/Y in meters for CF conversion ([PR 362](https://github.com/pytroll/pyresample/pull/362))
* [Issue 350](https://github.com/pytroll/pyresample/issues/350) - Breaking change for fill value in area.lonlat2colrow ([PR 351](https://github.com/pytroll/pyresample/pull/351))
* [Issue 296](https://github.com/pytroll/pyresample/issues/296) - Add CITATION information

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 367](https://github.com/pytroll/pyresample/pull/367) - Fix AreaDefinition.get_area_slices returning non-integer slices
* [PR 363](https://github.com/pytroll/pyresample/pull/363) - Fix version number not being preserved in wheels ([361](https://github.com/pytroll/pyresample/issues/361))
* [PR 362](https://github.com/pytroll/pyresample/pull/362) - Fix handling of geostationary x/y units in CF conversion ([355](https://github.com/pytroll/pyresample/issues/355))

#### Features added

* [PR 356](https://github.com/pytroll/pyresample/pull/356) - Add `get_min` and `get_max` to Bucket resampler (experimental)

#### Documentation changes

* [PR 351](https://github.com/pytroll/pyresample/pull/351) - Fix the documentation of get_array_indices_from_lonlat and add a test ([350](https://github.com/pytroll/pyresample/issues/350))

In this release 5 pull requests were closed.


## Version 1.19.0 (2021/04/14)

### Issues Closed

* [Issue 344](https://github.com/pytroll/pyresample/issues/344) - Improve handling of dask arrays in DynamicAreaDefinition ([PR 346](https://github.com/pytroll/pyresample/pull/346))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 348](https://github.com/pytroll/pyresample/pull/348) - Allow rows_per_scan=0 convenience in EWA resampler
* [PR 340](https://github.com/pytroll/pyresample/pull/340) - Fix get_area_slices for flipped areas

#### Features added

* [PR 346](https://github.com/pytroll/pyresample/pull/346) - Add better dask handling to DynamicAreaDefinitions ([344](https://github.com/pytroll/pyresample/issues/344))

In this release 3 pull requests were closed.


## Version 1.18.1 (2021/03/22)

### Issues Closed

* [Issue 345](https://github.com/pytroll/pyresample/issues/345) - Deprecated numpy data types (numpy >=1.20.0)

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 343](https://github.com/pytroll/pyresample/pull/343) - Fix EWA resampling when the result is all fill values
* [PR 342](https://github.com/pytroll/pyresample/pull/342) - Fix DynamicAreaDefinition not handling lons over antimeridian

In this release 2 pull requests were closed.


## Version 1.18.0 (2021/03/12)

### Issues Closed

* [Issue 317](https://github.com/pytroll/pyresample/issues/317) - Fix mask_all_nan kwarg in Average Bucket Resampler ([PR 319](https://github.com/pytroll/pyresample/pull/319))
* [Issue 315](https://github.com/pytroll/pyresample/issues/315) - Comparison of AreaDefinition to other types fails
* [Issue 295](https://github.com/pytroll/pyresample/issues/295) - Undetermined values in bilinear resampling result when resampling swath to grid data with geographic output coordinates. ([PR 330](https://github.com/pytroll/pyresample/pull/330))
* [Issue 293](https://github.com/pytroll/pyresample/issues/293) - Add Elliptical Weighted Nearest Neighbor option for swath resampling
* [Issue 281](https://github.com/pytroll/pyresample/issues/281) - Dask-ify Elliptical Weighted Averaging (EWA) resampling ([PR 284](https://github.com/pytroll/pyresample/pull/284))
* [Issue 152](https://github.com/pytroll/pyresample/issues/152) - Add dump/dumps methods to AreaDefinition ([PR 308](https://github.com/pytroll/pyresample/pull/308))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 330](https://github.com/pytroll/pyresample/pull/330) - Fix a division-by-zero failure causing missing data in bilinear interpolation ([295](https://github.com/pytroll/pyresample/issues/295))

#### Features added

* [PR 336](https://github.com/pytroll/pyresample/pull/336) - Switch to building cython extensions at installation time
* [PR 332](https://github.com/pytroll/pyresample/pull/332) - Require pyproj 2.2+ and remove fallbacks when CRS objects can be used
* [PR 330](https://github.com/pytroll/pyresample/pull/330) - Fix a division-by-zero failure causing missing data in bilinear interpolation ([295](https://github.com/pytroll/pyresample/issues/295))
* [PR 308](https://github.com/pytroll/pyresample/pull/308) - Change create_areas_def to dump and set default yaml style ([152](https://github.com/pytroll/pyresample/issues/152))
* [PR 306](https://github.com/pytroll/pyresample/pull/306) - Add a function to join/enclose areas. ([306](https://github.com/pytroll/pyresample/issues/306))
* [PR 284](https://github.com/pytroll/pyresample/pull/284) - Add dask-friendly EWA resampler class (DaskEWAResampler) ([281](https://github.com/pytroll/pyresample/issues/281))

#### Documentation changes

* [PR 338](https://github.com/pytroll/pyresample/pull/338) - Add CRS option to create_area_def docs ([338](https://github.com/pytroll/pyresample/issues/338))
* [PR 337](https://github.com/pytroll/pyresample/pull/337) - Switch docstrings to Google style
* [PR 334](https://github.com/pytroll/pyresample/pull/334) - Fix wrong varname in bucket doc
* [PR 314](https://github.com/pytroll/pyresample/pull/314) - Add citation information

#### Backward incompatible changes

* [PR 332](https://github.com/pytroll/pyresample/pull/332) - Require pyproj 2.2+ and remove fallbacks when CRS objects can be used

In this release 12 pull requests were closed.


## Version 1.17.0 (2020/11/12)

### Issues Closed

* [Issue 299](https://github.com/pytroll/pyresample/issues/299) - Refactor bilinear interpolation ([PR 300](https://github.com/pytroll/pyresample/pull/300))
* [Issue 297](https://github.com/pytroll/pyresample/issues/297) - Using CRS from `to_cartopy_crs()` triggers AttributeError in shapely
* [Issue 291](https://github.com/pytroll/pyresample/issues/291) - Help text for the bucket resampler is incorrect
* [Issue 289](https://github.com/pytroll/pyresample/issues/289) - AreaDefinition.area_extent mutability leads to hash violations ([PR 290](https://github.com/pytroll/pyresample/pull/290))
* [Issue 287](https://github.com/pytroll/pyresample/issues/287) - AttributeError when comparing `AreaDefinition` against other type ([PR 288](https://github.com/pytroll/pyresample/pull/288))
* [Issue 237](https://github.com/pytroll/pyresample/issues/237) - Test failure in test_kd_tree.Test.test_custom

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 305](https://github.com/pytroll/pyresample/pull/305) - Add missing test data to package
* [PR 304](https://github.com/pytroll/pyresample/pull/304) - Improve test compatibility with new pyproj
* [PR 298](https://github.com/pytroll/pyresample/pull/298) - Fix fill value for integer datasets, fix band assignment
* [PR 294](https://github.com/pytroll/pyresample/pull/294) - Fix colrow2lonlat working only for square areadefs

#### Features added

* [PR 310](https://github.com/pytroll/pyresample/pull/310) - Remove appveyor CI in favor of travis Windows and add Python 3.9 wheels
* [PR 303](https://github.com/pytroll/pyresample/pull/303) - Add caching of bilinear information
* [PR 300](https://github.com/pytroll/pyresample/pull/300) - Refactor bilinear ([299](https://github.com/pytroll/pyresample/issues/299))
* [PR 290](https://github.com/pytroll/pyresample/pull/290) - Make AreaDefinition.area_extent read only ([289](https://github.com/pytroll/pyresample/issues/289))

In this release 8 pull requests were closed.


## Version 1.16.0 (2020/06/10)

### Issues Closed

* [Issue 274](https://github.com/pytroll/pyresample/issues/274) - segmentation fault or AssertionError when resampling ([PR 277](https://github.com/pytroll/pyresample/pull/277))
* [Issue 272](https://github.com/pytroll/pyresample/issues/272) - `kd_tree.get_sample_from_neighbour_info` can't handle `fill_value` with `numpy` data types. ([PR 273](https://github.com/pytroll/pyresample/pull/273))
* [Issue 269](https://github.com/pytroll/pyresample/issues/269) - add a from_cf() mechanism for AreaDefinition ([PR 271](https://github.com/pytroll/pyresample/pull/271))
* [Issue 261](https://github.com/pytroll/pyresample/issues/261) - AreaDefinition docstring does not agree with its constructors definition ([PR 263](https://github.com/pytroll/pyresample/pull/263))
* [Issue 232](https://github.com/pytroll/pyresample/issues/232) - Possibly an issue with get_lonlats() for robinson projection

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 285](https://github.com/pytroll/pyresample/pull/285) - Check the source area orientation before defining slices ([274](https://github.com/pytroll/pyresample/issues/274))
* [PR 277](https://github.com/pytroll/pyresample/pull/277) - Fix calculating area slices for flipped projections ([274](https://github.com/pytroll/pyresample/issues/274))
* [PR 275](https://github.com/pytroll/pyresample/pull/275) - Check area slices for correct step
* [PR 273](https://github.com/pytroll/pyresample/pull/273) - Remove type check for nearest neighbour fill value. ([272](https://github.com/pytroll/pyresample/issues/272))
* [PR 265](https://github.com/pytroll/pyresample/pull/265) - Fix geocentric resolution favoring one area dimension over the other
* [PR 264](https://github.com/pytroll/pyresample/pull/264) - Switch to storing CRS WKT in AreaDefinitions instead of the CRS object
* [PR 251](https://github.com/pytroll/pyresample/pull/251) - Fix bugs and use real data in the plotting docs
* [PR 249](https://github.com/pytroll/pyresample/pull/249) - Fix bucket assignment

#### Features added

* [PR 282](https://github.com/pytroll/pyresample/pull/282) - Remove chunks that do not intersect target area in gradient search resampling
* [PR 279](https://github.com/pytroll/pyresample/pull/279) - Refactor API docs to document things more easily
* [PR 276](https://github.com/pytroll/pyresample/pull/276) - Create AreaDefinition from epsg codes
* [PR 271](https://github.com/pytroll/pyresample/pull/271) - Create AreaDefinition from a netCDF/CF file ([269](https://github.com/pytroll/pyresample/issues/269))

#### Documentation changes

* [PR 280](https://github.com/pytroll/pyresample/pull/280) - Remove unnecessary -P flag from the docs readme
* [PR 279](https://github.com/pytroll/pyresample/pull/279) - Refactor API docs to document things more easily
* [PR 263](https://github.com/pytroll/pyresample/pull/263) - Fix parameter order in AreaDefinition docstring ([261](https://github.com/pytroll/pyresample/issues/261), [261](https://github.com/pytroll/pyresample/issues/261))
* [PR 251](https://github.com/pytroll/pyresample/pull/251) - Fix bugs and use real data in the plotting docs

In this release 16 pull requests were closed.


## Version 1.15.0 (2020/03/20)

### Issues Closed

* [Issue 250](https://github.com/pytroll/pyresample/issues/250) - Misleading error when area file doesn't exist ([PR 259](https://github.com/pytroll/pyresample/pull/259))
* [Issue 244](https://github.com/pytroll/pyresample/issues/244) - Release wheels for pyresample ([PR 257](https://github.com/pytroll/pyresample/pull/257))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 258](https://github.com/pytroll/pyresample/pull/258) - Compatibility with PyProj v2.6.0
* [PR 255](https://github.com/pytroll/pyresample/pull/255) - Fix rpm building for python 3
* [PR 253](https://github.com/pytroll/pyresample/pull/253) - Fix SwathDefinition geocentric_resolution when resolution is None
* [PR 252](https://github.com/pytroll/pyresample/pull/252) - Make omerc bouding box use sphere as ellps ([1002](https://github.com/pytroll/satpy/issues/1002))

#### Features added

* [PR 259](https://github.com/pytroll/pyresample/pull/259) - Improve load_area functionality and error report ([250](https://github.com/pytroll/pyresample/issues/250), [250](https://github.com/pytroll/pyresample/issues/250))
* [PR 257](https://github.com/pytroll/pyresample/pull/257) -  Add Azure configuration to build wheels ([244](https://github.com/pytroll/pyresample/issues/244))
* [PR 254](https://github.com/pytroll/pyresample/pull/254) - Switch to pytest for CI and remove Python <3.4 support

#### Documentation changes

* [PR 253](https://github.com/pytroll/pyresample/pull/253) - Fix SwathDefinition geocentric_resolution when resolution is None

In this release 8 pull requests were closed.


## Version 1.14.0 (2019/12/22)

### Issues Closed

* [Issue 242](https://github.com/pytroll/pyresample/issues/242) - AreaDefinition.get_lonlats ignores dtype option ([PR 243](https://github.com/pytroll/pyresample/pull/243))
* [Issue 233](https://github.com/pytroll/pyresample/issues/233) - get_neighbour_info can not handle SwathDefinitions with lat lon of type integers ([PR 235](https://github.com/pytroll/pyresample/pull/235))
* [Issue 229](https://github.com/pytroll/pyresample/issues/229) - Update old documentation on easy quicklook display (stop using rainbow color map!) ([PR 230](https://github.com/pytroll/pyresample/pull/230))
* [Issue 228](https://github.com/pytroll/pyresample/issues/228) - Area definition boundaries where space pixels are excluded

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 245](https://github.com/pytroll/pyresample/pull/245) - Remove pyximports from gradient search
* [PR 243](https://github.com/pytroll/pyresample/pull/243) - Respect dtype when get_lonlats provide dask array ([242](https://github.com/pytroll/pyresample/issues/242))
* [PR 241](https://github.com/pytroll/pyresample/pull/241) - Fix typo in ImageContainerQuick and ImageContainerNearest docs
* [PR 236](https://github.com/pytroll/pyresample/pull/236) - Fix compatibility with pyproj 2.4.2
* [PR 227](https://github.com/pytroll/pyresample/pull/227) - Fix EWA resampling hanging when geolocation had a lot of NaNs
* [PR 224](https://github.com/pytroll/pyresample/pull/224) - Fix deprecation warning for abc classes

#### Features added

* [PR 230](https://github.com/pytroll/pyresample/pull/230) - No rainbow update documentation ([229](https://github.com/pytroll/pyresample/issues/229))
* [PR 225](https://github.com/pytroll/pyresample/pull/225) - Add smarter default radius_of_influence to XArrayResamplerNN resampling
* [PR 222](https://github.com/pytroll/pyresample/pull/222) - Make the uniform shape computation more effective for dask arrays
* [PR 191](https://github.com/pytroll/pyresample/pull/191) - Implement gradient search resampling method

#### Documentation changes

* [PR 241](https://github.com/pytroll/pyresample/pull/241) - Fix typo in ImageContainerQuick and ImageContainerNearest docs
* [PR 238](https://github.com/pytroll/pyresample/pull/238) - Update load_area docstring to mention that multiple files are allowed
* [PR 230](https://github.com/pytroll/pyresample/pull/230) - No rainbow update documentation ([229](https://github.com/pytroll/pyresample/issues/229))

In this release 13 pull requests were closed.


## Version 1.13.2 (2019/10/08)

### Issues Closed

* [Issue 220](https://github.com/pytroll/pyresample/issues/220) - Problem with dynamic areas on numpy arrays with newest pyresample ([PR 221](https://github.com/pytroll/pyresample/pull/221))
* [Issue 148](https://github.com/pytroll/pyresample/issues/148) - Complete dask conversion of XArrayResamplerBilinear
* [Issue 10](https://github.com/pytroll/pyresample/issues/10) - Computing density_of_x (alternatively "counting number of x)" while re-gridding

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 221](https://github.com/pytroll/pyresample/pull/221) - Make optimal bb computation work with numpy arrays ([220](https://github.com/pytroll/pyresample/issues/220))

In this release 1 pull request was closed.


## Version 1.13.1 (2019/09/26)

### Pull Requests Merged

#### Bugs fixed

* [PR 218](https://github.com/pytroll/pyresample/pull/218) - Fix proj_str returning invalid PROJ strings when towgs84 was included
* [PR 217](https://github.com/pytroll/pyresample/pull/217) - Fix get_geostationary_angle_extent assuming a/b definitions
* [PR 216](https://github.com/pytroll/pyresample/pull/216) - Fix proj4 radius parameters for spherical cases

In this release 3 pull requests were closed.

## Version 1.13.0 (2019/09/13)

### Issues Closed

* [Issue 210](https://github.com/pytroll/pyresample/issues/210) - Incompatibility with new proj/pyproj versions

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 213](https://github.com/pytroll/pyresample/pull/213) - Remove extra conversion to dask array
* [PR 208](https://github.com/pytroll/pyresample/pull/208) - Bugfix bilinear resampler masking ([735](https://github.com/pytroll/satpy/issues/735))
* [PR 207](https://github.com/pytroll/pyresample/pull/207) - Make output index tiling in bilinear interpolation work with dask
* [PR 205](https://github.com/pytroll/pyresample/pull/205) - Exclude NaNs from Bucket Average
* [PR 197](https://github.com/pytroll/pyresample/pull/197) - Fix to_cartopy_crs for latlong projections
* [PR 196](https://github.com/pytroll/pyresample/pull/196) - Improve handling of EPSG codes with pyproj 2.0+

#### Features added

* [PR 212](https://github.com/pytroll/pyresample/pull/212) - Use slices in bilinear resampler
* [PR 203](https://github.com/pytroll/pyresample/pull/203) - Add Numpy version limitation for Python 2
* [PR 198](https://github.com/pytroll/pyresample/pull/198) - Clarify warning if no overlap data and projection
* [PR 196](https://github.com/pytroll/pyresample/pull/196) - Improve handling of EPSG codes with pyproj 2.0+
* [PR 192](https://github.com/pytroll/pyresample/pull/192) - Add bucket resampling

#### Documentation changes

* [PR 204](https://github.com/pytroll/pyresample/pull/204) - Add Example for Regular Lat-Lon Grid
* [PR 201](https://github.com/pytroll/pyresample/pull/201) - fix bug in plot example code
* [PR 198](https://github.com/pytroll/pyresample/pull/198) - Clarify warning if no overlap data and projection
* [PR 195](https://github.com/pytroll/pyresample/pull/195) - Update docs for create_area_def and improve AreaDefinition property consistency

In this release 15 pull requests were closed.


## Version 1.12.3 (2019/05/17)

### Pull Requests Merged

#### Bugs fixed

* [PR 193](https://github.com/pytroll/pyresample/pull/193) - Fix striding slicing in AreaDefinition

In this release 1 pull request was closed.


## Version 1.12.2 (2019/05/10)

### Issues Closed

* [Issue 187](https://github.com/pytroll/pyresample/issues/187) - Numerous `RuntimeWarning`s when resampling

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 190](https://github.com/pytroll/pyresample/pull/190) - Fix aggregate method using non-serializable internal function
* [PR 189](https://github.com/pytroll/pyresample/pull/189) - Fix dask race condition in KDTree resampling

#### Features added

* [PR 183](https://github.com/pytroll/pyresample/pull/183) - Fix bb computation to generate areas with equal h and v resolutions

In this release 3 pull requests were closed.


## Version 1.12.1 (2019/04/24)

### Pull Requests Merged

#### Bugs fixed

* [PR 186](https://github.com/pytroll/pyresample/pull/186) - Fix support for pyproj-2 EPSG syntax

#### Documentation changes

* [PR 185](https://github.com/pytroll/pyresample/pull/185) - Fix argument order in get_area_def doc

In this release 2 pull requests were closed.


## Version 1.12.0 (2019/04/06)

### Issues Closed

* [Issue 178](https://github.com/pytroll/pyresample/issues/178) - Can't install pyresample on OSX Mojave

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 177](https://github.com/pytroll/pyresample/pull/177) - Fix dynamic omerc parameter computation to survive nans

#### Features added

* [PR 182](https://github.com/pytroll/pyresample/pull/182) - Implement striding and aggregation for Swath- and AreaDefinitions
* [PR 180](https://github.com/pytroll/pyresample/pull/180) - Remove radians from create_area_def and allow compatibility with pyproj-2.0+

In this release 3 pull requests were closed.


## Version 1.11.2 (2019/03/18)

### Pull Requests Merged

#### Documentation changes

* [PR 176](https://github.com/pytroll/pyresample/pull/176) - Fix typos in README and index page

In this release 1 pull request was closed.

## Version 1.11.1 (2019/03/18)

### Issues Closed

* [Issue 165](https://github.com/pytroll/pyresample/issues/165) - Update use of dask `atop` to `blockwise`
* [Issue 172](https://github.com/pytroll/pyresample/issues/172) - Missing metadata on PyPI ([PR 173](https://github.com/pytroll/pyresample/pull/173))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 175](https://github.com/pytroll/pyresample/pull/175) - Fix dask.blockwise backwards compatibility with dask.atop

In this release 1 pull request was closed.


## Version 1.11.0 (2019/03/15)

### Issues Closed

* [Issue 160](https://github.com/pytroll/pyresample/issues/160) - No coastlines in cartopy if area is flipped ([PR 161](https://github.com/pytroll/pyresample/pull/161))
* [Issue 136](https://github.com/pytroll/pyresample/issues/136) - Update documentation to not reference scipy kdtree ([PR 155](https://github.com/pytroll/pyresample/pull/155))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 170](https://github.com/pytroll/pyresample/pull/170) - Allow create_area_def to work with incomplete proj dicts to create DynamicAreas
* [PR 167](https://github.com/pytroll/pyresample/pull/167) - Fix yaml area loading when projection is latlong (units degrees)

#### Features added

* [PR 164](https://github.com/pytroll/pyresample/pull/164) - Delete unused requirements.txt file
* [PR 156](https://github.com/pytroll/pyresample/pull/156) - Refactor pyresample.utils in to separate modules
* [PR 150](https://github.com/pytroll/pyresample/pull/150) - Switch to versioneer
* [PR 145](https://github.com/pytroll/pyresample/pull/145) - Refactor and deprecate geometry "*_dask" methods
* [PR 138](https://github.com/pytroll/pyresample/pull/138) - Add `create_area_def` utility method and refactor AreaDefinition arguments

#### Documentation changes

* [PR 155](https://github.com/pytroll/pyresample/pull/155) - Update installation instructions to match current best practices ([136](https://github.com/pytroll/pyresample/issues/136))

In this release 8 pull requests were closed.


## Version 1.10.3 (2018/11/23)

### Issues Closed

* [Issue 92](https://github.com/pytroll/pyresample/issues/92) - Add utility function for converting geotiffs to area definitions ([PR 143](https://github.com/pytroll/pyresample/pull/143))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 147](https://github.com/pytroll/pyresample/pull/147) - Fix dtype preservation for kdtree resampling
* [PR 144](https://github.com/pytroll/pyresample/pull/144) - Non-contiguous area definitions are now not concatenable ([491](https://github.com/pytroll/satpy/issues/491))

#### Features added

* [PR 143](https://github.com/pytroll/pyresample/pull/143) - get_area_def_from_raster ([92](https://github.com/pytroll/pyresample/issues/92))
* [PR 142](https://github.com/pytroll/pyresample/pull/142) - Add converter from def to yaml

In this release 4 pull requests were closed.


## Version 1.10.2 (2018/10/01)

### Issues Closed

* [Issue 133](https://github.com/pytroll/pyresample/issues/133) - Build issue with Python 3.7 ([PR 135](https://github.com/pytroll/pyresample/pull/135))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 139](https://github.com/pytroll/pyresample/pull/139) - Fix area parsing code to convert PROJ.4 parameters to float if possible
* [PR 135](https://github.com/pytroll/pyresample/pull/135) - Fix Python 3.7 compatibility ([133](https://github.com/pytroll/pyresample/issues/133))

In this release 2 pull requests were closed.


## Version 1.10.1 (2018/07/03)

### Pull Requests Merged

#### Bugs fixed

* [PR 130](https://github.com/pytroll/pyresample/pull/130) - Fix log message not to rely on the proj key
* [PR 129](https://github.com/pytroll/pyresample/pull/129) - Ignore proj dicts with no  key for slicing

In this release 2 pull requests were closed.


## Version 1.10.0 (2018/06/25)

### Pull Requests Merged

#### Features added

* [PR 128](https://github.com/pytroll/pyresample/pull/128) - Add option to provide KDTree's 'mask' argument when querying

In this release 1 pull request was closed.


## Version 1.9.3 (2018/06/08)

### Issues Closed

* [Issue 113](https://github.com/pytroll/pyresample/issues/113) - Not all the close neighbours are found until search radius is increased ([PR 112](https://github.com/pytroll/pyresample/pull/112))
* [Issue 111](https://github.com/pytroll/pyresample/issues/111) - Bilinear interpolation leaves holes in fields with constant value ([PR 112](https://github.com/pytroll/pyresample/pull/112))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 125](https://github.com/pytroll/pyresample/pull/125) - Fix area slices not working for non-geos projections
* [PR 119](https://github.com/pytroll/pyresample/pull/119) - Add hashing to StackedAreaDefinitions

In this release 1 pull request was closed.


## Version 1.9.2 (2018/05/13)

### Pull Requests Merged

#### Bugs fixed

* [PR 117](https://github.com/pytroll/pyresample/pull/117) - Fix get_area_slices ([218](https://github.com/pytroll/satpy/issues/218))

#### Features added

* [PR 116](https://github.com/pytroll/pyresample/pull/116) - Simplify get_sample_from_neighbour_info method

In this release 2 pull requests were closed.


## Version 1.9.1 (2018/05/03)

### Pull Requests Merged

#### Features added

* [PR 115](https://github.com/pytroll/pyresample/pull/115) - Geos area reduction

In this release 1 pull request was closed.
