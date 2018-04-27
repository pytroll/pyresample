Changelog
=========


v1.9.0 (2018-04-27)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.8.3 → 1.9.0. [Martin Raspaud]
- Merge pull request #114 from pytroll/feature-hash-update. [Martin
  Raspaud]

  Implement Deterministic hashing
- Fix __hash__ inheritance block in python 3. [Martin Raspaud]
- Add the hash attribute to all definitions. [Martin Raspaud]
- Add the update_hash method, centralize __hash__ [Martin Raspaud]
- Merge pull request #102 from pytroll/feature-cartopy-convert. [David
  Hoese]

  Add cartopy conversion method to AreaDefinition
- Add license and copyright to plot and test_plot. [davidh-ssec]
- Update appveyor badge to point to new project. [davidh-ssec]
- Fix line too long. [davidh-ssec]
- Merge branch 'develop' into feature-cartopy-convert. [davidh-ssec]

  # Conflicts:
  #	appveyor.yml
  #	pyresample/__init__.py
  #	pyresample/geometry.py
  #	pyresample/kd_tree.py

- Merge pull request #109 from pytroll/feature-better-omerc-azimuth.
  [Martin Raspaud]

  Make azimuth angle for omerc dynamic areas more robust
- Fix tests for new omerc parameters computation. [Martin Raspaud]
- Take care of azimuth flipping in omerc parameter computation. [Martin
  Raspaud]
- Take care of small omerc azimuth angles. [Martin Raspaud]
- Use no_rot for better 2-point omerc fitting. [Martin Raspaud]
- Make azimuth angle for omerc dynamic areas more robust. [Martin
  Raspaud]
- Add basemap to travis for doctests. [davidh-ssec]
- Remove appveyor unused scripts. [davidh-ssec]
- Fix conda dependencies on travis and switch to ci-helpers for
  appveyor. [davidh-ssec]
- Add missing coverage dependency to travis CI. [davidh-ssec]
- Use conda for travis tests. [davidh-ssec]
- Update github templates. [davidh-ssec]
- Fix flake8 issues. [davidh-ssec]
- Add basemap quicklook generation back in as a fallback. [davidh-ssec]
- Install proj libraries binaries for cartopy. [davidh-ssec]
- Remove python 3.4 and 3.5 from CI tests. [davidh-ssec]
- Add simple cartopy conversion test. [davidh-ssec]
- Skip basemap tests if basemap isn't available. [davidh-ssec]
- Switch quicklook to use cartopy instead of basemap. [davidh-ssec]
- Replace quicklook functionality with cartopy. [davidh-ssec]
- Update documentation to include cartopy example. [davidh-ssec]
- Add cartopy conversion method to AreaDefinition. [davidh-ssec]


v1.8.3 (2018-03-19)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.8.2 → 1.8.3. [Martin Raspaud]
- Merge branch 'develop' into new_release. [Martin Raspaud]
- Merge pull request #107 from pytroll/bugfix-memory-leak. [Martin
  Raspaud]

  [WIP] Remove closures to allow memory to be freed
- Prevend dynamic areas to choke on NaNs. [Martin Raspaud]
- Make CHUNK_SIZE int if taken from environment. [Martin Raspaud]
- Reorganize indices assignments. [Martin Raspaud]
- Remove closures to allow memory to be freed. [Martin Raspaud]
- Merge pull request #106 from pytroll/bugfix-area-equality. [David
  Hoese]

  Fix area equality to support np.nan, xarray and dask
- Add dask and xarray to appveyor. [Martin Raspaud]
- Use numpy's allclose for swathdef equality. [Martin Raspaud]
- Require a newer numpy for nan equality. [Martin Raspaud]
- Style cleanup. [Martin Raspaud]
- Add tests for swath equality. [Martin Raspaud]
- Style cleanup. [Martin Raspaud]
- Fix area equality to support xarray and dask. [Martin Raspaud]
- Merge pull request #108 from pytroll/add-stickler-config. [Martin
  Raspaud]

  Adding .stickler.yml configuration file
- Adding .stickler.yml. [stickler-ci]


v1.8.2 (2018-03-01)
-------------------
- update changelog. [davidh-ssec]
- Bump version: 1.8.1 → 1.8.2. [davidh-ssec]
- Merge pull request #104 from pytroll/bugfix-chunk-size. [David Hoese]

  Allow chunk size in dask methods to be 2D
- Fix line too long. [davidh-ssec]
- Fix chunk size 'get_proj_vectors_dask' so it can be 2D. [davidh-ssec]


v1.8.1 (2018-02-22)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.8.0 → 1.8.1. [Martin Raspaud]
- Merge pull request #101 from floriankrb/master. [Martin Raspaud]

  Update README to include correct versions of python tested
- Update README.md. [Florian]
- Update README.md. [Florian]
- Merge pull request #99 from pytroll/feature-dynamic-projs. [Martin
  Raspaud]

  Add support for dynamic resampling for most projections
- Do not overwrite provided lon_0 and lat_0. [Martin Raspaud]
- Add support for dynamic resampling for most projections. [Martin
  Raspaud]
- Merge pull request #98 from pytroll/bugfix-data-reduce. [Martin
  Raspaud]

  Revert "Fix data reduction when poles are within area"
- Add test for data reduction over the poles. [Martin Raspaud]
- Make pep8 happy. [Martin Raspaud]
- Revert "Fix data reduction when poles are within area" [Martin
  Raspaud]

  This reverts commit 1c9ac493aea549a354f384059e9aa6ad41558fd8.

- Merge pull request #96 from pytroll/bugfix-partially-invalid-source-
  data. [David Hoese]

  Fix xarray resampling for partially invalid source datasets
- Fix xarray resampling for partially invalid source datasets. [Martin
  Raspaud]


v1.8.0 (2018-02-02)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.7.1 → 1.8.0. [Martin Raspaud]
- Merge branch 'develop' into new_release. [Martin Raspaud]
- Merge pull request #95 from pytroll/bugfix-pyproj-version. [Martin
  Raspaud]

  Provide the minimum version of pyproj needed
- Provide the minimum version of pyproj needed. [Martin Raspaud]
- Merge pull request #94 from pytroll/optimize-xarray. [Martin Raspaud]

  Optimize xarray
- Add test for new wrap_and_check function. [davidh-ssec]
- Rename chunk size environment variable to PYTROLL_CHUNK_SIZE. [davidh-
  ssec]
- Fix circular import between geometry and init's CHUNK_SIZE. [davidh-
  ssec]
- Revert import removal in init and add easy access imports. [davidh-
  ssec]

  Includes attempt to remove circular dependency between utils and
  geometry module.

- Use central CHUNK_SIZE constant for dask based operations. [davidh-
  ssec]
- Add `check_and_wrap` utility function and fix various docstring
  issues. [davidh-ssec]
- Remove tests for removed features. [davidh-ssec]
- Remove longitude/latitude validity checks in BaseDefinition. [davidh-
  ssec]

  This was causing issues with dask based inputs and was a performance
  penalty for all use cases even when the arrays were valid. Removing
  this check should not affect 99% of users.

- Combine dask operations to improve resampling performance. [davidh-
  ssec]

  Still a lot that could be done probably.

- Fix dask minimum version number for meshgrid support. [davidh-ssec]
- Add dask extra to setup.py to specify minimum dask version. [davidh-
  ssec]

  pyresample uses dask meshgrid which came in version 1.9

- Merge pull request #86 from pytroll/feature-multiple-dims. [Martin
  Raspaud]

  [WIP] Feature multiple dims
- Remove explicit chunksize. [Martin Raspaud]
- Clean up with pep8. [Martin Raspaud]
- Take care of coordinates when resampling. [Martin Raspaud]
- Define default blocksizes for dask arrays. [Martin Raspaud]
- Merge branch 'feature-optimize-dask' into feature-multiple-dims.
  [Martin Raspaud]
- Style cleanup. [Martin Raspaud]
- Fix get_hashable_array for variations of np arrays. [Martin Raspaud]
- Print warning when wrapping is needed independently of type. [Martin
  Raspaud]
- Change default blocksize to 5000. [Martin Raspaud]
- Make use of dask's map_blocks. [Martin Raspaud]

  Instead of writing our own array definitions
- Revert "Make resampling lazy" [Martin Raspaud]

  This reverts commit 5a4f9c342f9c8262c06c28986163fc682242ce75.

- Make resampling lazy. [Martin Raspaud]
- Revert yapf change. [Martin Raspaud]
- Clean up code (pycodestyle, pydocstyle) [Martin Raspaud]
- Make XR resampling work with more dimensions. [Martin Raspaud]
- Merge pull request #91 from avalentino/issues/gh-090. [David Hoese]

  Fix test_get_array_hashable on big-endian machines (closes #90)
- Fix test_get_array_hashable on big-endian machines. [Antonio
  Valentino]


v1.7.1 (2017-12-21)
-------------------
- update changelog. [davidh-ssec]
- Bump version: 1.7.0 → 1.7.1. [davidh-ssec]
- Merge pull request #88 from pytroll/bugfix-masked-target. [David
  Hoese]

  Fix kdtree when target lons/lats are masked arrays
- Add test for masked valid_output_index fix. [davidh-ssec]
- Move bilinear test setup to a special method. [davidh-ssec]
- Fix kdtree when target lons/lats are masked arrays. [davidh-ssec]
- Merge pull request #89 from Funkensieper/fix-masks-in-get-resampled-
  image. [David Hoese]

  Fix masks in grid.get_resampled_image
- Add test for mask preservation. [Stephan Finkensieper]
- Distinguish between ndarrays and masked arrays. [Stephan Finkensieper]
- Fix masks in grid.get_resampled_image. [Stephan Finkensieper]

  Use numpy.ma version of row_stack to prevent loosing the mask of
  large images (rows > cut_off)

- Add github templates. [Martin Raspaud]
- Merge pull request #84 from pytroll/feature-add-hash. [Martin Raspaud]

  Add hash method to AreaDefinition and SwathDefinition
- Fix dask array not being hashable in py3.x. [Martin Raspaud]
- Use identity checking instead of equality. [Martin Raspaud]
- Do not has the mask if it's empty. [Martin Raspaud]
- Bugfix geometry test. [Martin Raspaud]
- Replace hash value checks with type checks. [Martin Raspaud]

  The value can be different depending on the python version apparently.
- Add dask and xarray for testing on travis. [Martin Raspaud]
- Fix case of missing xarray dependency in the tests. [Martin Raspaud]
- Add __hash__ for SwathDefinitions, along with some unittests. [Martin
  Raspaud]
- Add hash method to AreaDefinition. [davidh-ssec]

  Removes annoying log message when xarray/dask is missing

- Merge branch 'feature-xarray-improvements' into develop. [Martin
  Raspaud]

  Conflicts:
  	pyresample/geometry.py

- Type coords to np.float. [Martin Raspaud]
- Add support for fill_value in nn search. [Martin Raspaud]
- Change the get_lonlats_dask interface to return a tuple. [Martin
  Raspaud]
- Fix masking bad latitude values. [davidh-ssec]
- Fix consistency with numpy arrays. [davidh-ssec]
- Allow xarrays internally in geometry objects. [davidh-ssec]
- Merge remote-tracking branch 'origin/develop' into develop. [davidh-
  ssec]

  # Conflicts:
  #	.travis.yml

- Fix proj4 dict to string against recent changes to str to dict funcs.
  [davidh-ssec]
- Change appveyor python 3.5 environments to python 3.6. [davidh-ssec]

  Also removes slack notification webhook which is no longer the
  recommended way to post to slack from appveyor.

- Exclude buggy version of matplotlib in travis tests. [davidh-ssec]
- Fix proj4 dict conversion test. [davidh-ssec]
- Use more descriptive variable names. [davidh-ssec]
- Add proj4_dict_to_str utility function. [davidh-ssec]

  Includes fixes for dynamic area definitions proj_id and
  small performance improvement for projection coordinate generation

- Merge pull request #83 from loreclem/master. [Martin Raspaud]

  Added ROTATION in an area definition
- Bugfix in get_area_def. [lorenzo clementi]
- Unit test for rotation. [lorenzo clementi]
- Removed unused parameter. [lorenzo clementi]
- Now working also with yaml. [lorenzo clementi]
- Code improvements. [lorenzo clementi]
- Added ROTATION in an area definition. [lorenzo clementi]


v1.7.0 (2017-10-13)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.6.1 → 1.7.0. [Martin Raspaud]
- Merge pull request #82 from pytroll/fix-resample-bilinear. [David
  Hoese]

  Fix output shape of resample_bilinear()
- Reshape output to have correct shape for the output area and num of
  chans. [Panu Lahtinen]
- Update tests to check proper output shape for resample_bilinear()
  [Panu Lahtinen]
- Merge pull request #79 from pytroll/fix-bil-documentation. [David
  Hoese]

  Fix example data for BIL, clarify text and add missing output_shape p…
- Merge branch 'fix-bil-documentation' of
  https://github.com/mraspaud/pyresample into fix-bil-documentation.
  [Panu Lahtinen]
- Fix example data for BIL, clarify text and add missing output_shape
  param. [Panu Lahtinen]
- Fix example data for BIL, clarify text and add missing output_shape
  param. [Panu Lahtinen]
- Merge pull request #75 from pytroll/fix-bil-mask-deprecation. [David
  Hoese]

  Fix bil mask deprecation
- Merge branch 'develop' into fix-bil-mask-deprecation. [David Hoese]
- Merge pull request #81 from pytroll/fix-reduce-bil-memory-use. [David
  Hoese]

  Reduce the memory use for ImageContainerBilinear tests
- Reduce area size for BIL, reduce neighbours and adjust expected
  results. [Panu Lahtinen]
- Add proj4_dict_to_str utility function (#78) [David Hoese]

  * Add proj4_dict_to_str utility function

  Includes fixes for dynamic area definitions proj_id and
  small performance improvement for projection coordinate generation

  * Use more descriptive variable names

  * Fix proj4 dict conversion test

  * Exclude buggy version of matplotlib in travis tests

  * Change appveyor python 3.5 environments to python 3.6

  Also removes slack notification webhook which is no longer the
  recommended way to post to slack from appveyor.

  * Fix proj4 dict to string against recent changes to str to dict funcs

- Utils edits for retreiving projection semi-major / semi-minor axes
  (#77) [goodsonr]

  proj4 strings converted to dictionary now consistent with other code (no longer has leading '+')
  new logic for reporting projection semi-major / semi-minor axes ('a', 'b') based on information in proj4

- Merge pull request #71 from pytroll/feature-bilinear-image. [David
  Hoese]

  Add image container for bilinear interpolation
- Fix test result assertation. [Panu Lahtinen]
- Add tests for ImageContainerBilinear, rewrap long lines. [Panu
  Lahtinen]
- Fix docstrings. [Panu Lahtinen]
- Mention also ImageContainerBilinear. [Panu Lahtinen]
- Handle 3D input data with bilinear interpolation. [Panu Lahtinen]
- Add ImageContainerBilinear, autopep8. [Panu Lahtinen]
- Merge pull request #74 from pytroll/fix-close-area-file. [David Hoese]

  Use context manager to open area definition files
- Use context manager to open files, PEP8. [Panu Lahtinen]
- Merge pull request #76 from pytroll/feature-xarray. [Martin Raspaud]

  Support resampling of xarray.DataArrays
- Move docstring to init for consistency. [Martin Raspaud]
- Merge develop into feature_xarray. [Martin Raspaud]
- Support get_lonlats_dask in StackedAreaDefinitions. [Martin Raspaud]
- Add get_lonlats_dask for SwathDefinitions. [Martin Raspaud]
- Fix resampling of multidimensional xarrays. [Martin Raspaud]
- Support xarray and use dask for simple cases. [Martin Raspaud]
- WIP: Resampler for xarrays using dask. [Martin Raspaud]
- Fix formatting. [Martin Raspaud]
- Optimize memory consumption. [Martin Raspaud]
- Clean up doc formatting. [Martin Raspaud]
- Add dask.Array returning get_lonlats and get_proj_coords. [Martin
  Raspaud]
- Remove Python 3.3 from travis tests, it's not supported anymore. [Panu
  Lahtinen]
- Supress UserWarning about possible extra neighbours within search
  radius. [Panu Lahtinen]
- Handle masked arrays properly for new Numpy versions. [Panu Lahtinen]


v1.6.1 (2017-09-18)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.6.0 → 1.6.1. [Martin Raspaud]
- Merge pull request #60 from pytroll/feature-dynamic-area. [David
  Hoese]

  Add support for dynamic areas
- Merge branch 'develop' into feature-dynamic-area. [Martin Raspaud]
- Apply assert_allclose to proj dicts for tests. [Martin Raspaud]
- Fix some style issues. [Martin Raspaud]
- Set DynamicArea proj to `omerc` by default. [Martin Raspaud]
- Implement proposed changes in PR review. [Martin Raspaud]
- Use numpy's assert almost equal for area_extent comparisons. [Martin
  Raspaud]
- Document the DynamicArea class. [Martin Raspaud]
- Fix optimal projection computation tests. [Martin Raspaud]
- Pep8 cleanup. [Martin Raspaud]
- Valid index computation optimization. [Martin Raspaud]
- Change bb computation api to use the whole proj_dict. [Martin Raspaud]
- Fix unittests for updated omerc computations. [Martin Raspaud]
- Use other azimuth direction for omerc. [Martin Raspaud]
- Flip x and y size in omerc projection. [Martin Raspaud]
- Bugfix typo. [Martin Raspaud]
- Allow lons and lats to be any array in bb computation. [Martin
  Raspaud]
- Add SwathDefinition tests to the test suite. [Martin Raspaud]
- Support bounding box area computation from SwathDefintion. [Martin
  Raspaud]

  This add support for computing a bounding box area from a swath definition that would fit optimally. The default projection is oblique mercator, with is optimal for locally received imager passes.
- Add support for dynamic areas. [Martin Raspaud]
- Merge pull request #70 from pytroll/feature-radius-parameters. [David
  Hoese]

  Add 'proj4_radius_parameters' to calculate 'a' and 'b' from ellps
- Add tests for proj4_radius_parameters. [davidh-ssec]
- Fix typo in function call in radius parameters. [davidh-ssec]
- Add 'proj4_radius_parameters' to calculate 'a' and 'b' from ellps.
  [davidh-ssec]
- Merge pull request #68 from pytroll/feature-56. [Martin Raspaud]

  Fix GridDefinition as permitted definition in preprocessing utils
- Add more preprocessing tests. [davidh-ssec]
- Fix preprocessing functions to use duck type on provided areas.
  [davidh-ssec]
- Fix GridDefinition as permitted definition in preprocessing utils.
  [davidh-ssec]


v1.6.0 (2017-09-12)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.5.0 → 1.6.0. [Martin Raspaud]
- Make sure x_size and y_size are ints. [Martin Raspaud]
- Merge pull request #69 from pytroll/bugfix-66. [Martin Raspaud]

  Fix write to mask affecting original mask in future versions of numpy

  Fixes #66
- Add python 3.6 to travis tests. [davidh-ssec]
- Fix write to mask affecting original mask in future versions of numpy.
  [davidh-ssec]

  Fix #66

- Merge pull request #67 from pytroll/bugfix-13. [Martin Raspaud]

  Rename `proj_x/y_coords` to `projection_x/y_coords`
- Rename `proj_x/y_coords` to `projection_x/y_coords` [davidh-ssec]

  Fix #13

- Merge pull request #63 from pytroll/feature-multiple-area-files.
  [David Hoese]

  Parse multiple area files
- Fix tests_require in setup.py. [davidh-ssec]
- Use libgeos-dev to depend on the C++ libgeos-X.X.X and libgeos-c1.
  [davidh-ssec]
- Add simple tests for parsing multiple yaml area strings. [davidh-ssec]
- Fix indentation in area file parsing functions. [davidh-ssec]
- Add ability to parse multiple area files at once. [davidh-ssec]
- Merge pull request #65 from pytroll/fix-numpy-1.13. [Martin Raspaud]

  Fix numpy 1.13 compatibility
- Fix boolean mask array usage in gaussian resampling. [davidh-ssec]

  In numpy 1.13 it is illegal to index an array with a boolean
  array of a different size.

- Add mock to test dependencies for python <3.3. [davidh-ssec]
- Use prepackaged numexpr in bdist_rpm. [Martin Raspaud]


v1.5.0 (2017-05-02)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.4.1 → 1.5.0. [Martin Raspaud]
- Merge pull request #58 from pytroll/feature-yaml-areas. [David Hoese]

  Add support for areas in yaml format
- Remove support for python 2.6. [Martin Raspaud]
- Explain that x/y can be lon/lat. [Martin Raspaud]
- Fix __str__ and dump of area defs to be more explicit. [Martin
  Raspaud]
- Add missing doctest file. [Martin Raspaud]
- Add yaml as a requirement. [Martin Raspaud]
- Add support for areas in yaml format. [Martin Raspaud]
- Fix travis script not going back to base directory for coveralls to
  work. [davidh-ssec]

  Sphinx was used for testing and included a `cd` command but that made coveralls unable to find the .coverage output.

  (cherry picked from commit 33e692a)

- Replace dict comprehension for 2.6 compatibility. [davidh-ssec]
- Add basic ll2cr and fornav wrapper tests. [davidh-ssec]


v1.4.1 (2017-04-07)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.4.0 → 1.4.1. [Martin Raspaud]
- Fix non-contiguous arrays passed to EWA resampling. [davidh-ssec]

  Includes fixes for tuple `out` and proper passing of keyword arguments

- Ensure pyproj gets ndarrays with np.nans instead of masked arrays.
  [Panu Lahtinen]
- Handle older numpy versions without "copy" kwrd in .astype() [Panu
  Lahtinen]


v1.4.0 (2017-04-02)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.3.1 → 1.4.0. [Martin Raspaud]
- Add mock to appveyor. [Martin Raspaud]
- Fix 2.6 compatibility. [Martin Raspaud]
- Add StackedAreaDefinition class and helper functions. [Martin Raspaud]


v1.3.1 (2017-03-22)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.3.0 → 1.3.1. [Martin Raspaud]
- Handle TypeError raised by case where all values are masked. [Panu
  Lahtinen]
- Remove trailing spaces in data_reduce.py. [Martin Raspaud]
- Fix data reduction when poles are within area. [Martin Raspaud]
- Make rtd happy with a new requirements file. [Martin Raspaud]
- add pytroll's pykdtree to requirements.txt. [Martin Raspaud]


v1.3.0 (2017-02-07)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.9 → 1.3.0. [Martin Raspaud]
- Merge pull request #55 from pytroll/feature-bilinear. [Martin Raspaud]

  Feature bilinear
- Add Python2 miniconda version number. [Panu Lahtinen]
- Rename *area_in* to *source_geo_def* and *area_out* to
  *target_area_def* [Panu Lahtinen]
- Fix search radius from 50e5 meters to 50e3 meters. [Panu Lahtinen]
- Add access to kd_tree parameters reduce_data, segments and epsilon.
  [Panu Lahtinen]
- Add missing return value to docstring. [Panu Lahtinen]
- Remove possibility to use tuple of coordinates as "in_area" [Panu
  Lahtinen]
- Try if older version of Pillow is installable with Python 2.6. [Panu
  Lahtinen]
- Remove obsolete tests + minor adjustments + comments. [Panu Lahtinen]

  Remove tests for functions that were removed.  Add test for getting
  coefficients for quadratic equations.  Add test for _get_ts().  Test
  that small variations doesn't cause failures when solving the quadratic
  equation.  Check all pixels of the output in test_get_bil_info().

- Adjust order so that most common case is first. [Panu Lahtinen]
- Remove parallelity checks. [Panu Lahtinen]

  Don't bother checking if lines area parallel, just run the most common
  (irregular rectangle) for all data, and run the two algorigthms
  consecutively for those where no valid data is yet present (ie. have
  np.nan).

- Test failure of _get_ts_irregular when verticals are parallel. [Panu
  Lahtinen]
- Refactor numpyfying. [Panu Lahtinen]
- Clarify function name. [Panu Lahtinen]
- Refactor. [Panu Lahtinen]

  Move common parts of _get_ts_irregular() and _get_ts_uprights_parallel()
  to two functions: one to get the parameters for quadratic equation and
  one to solve the other fractional distance not solved from the quadratic
  equation.

- Fix example code. [Panu Lahtinen]
- Enable doctest for resampling from bilinear coefficients. [Panu
  Lahtinen]
- Fix unittest which had wrong "correct" value. [Panu Lahtinen]
- Replace np.ma.masked_where() with np.ma.masked_invalid() [Panu
  Lahtinen]
- Move input checks to a function. [Panu Lahtinen]
- Add more unit tests. [Panu Lahtinen]
- Move check of source area to get_bil_info() [Panu Lahtinen]
- Ensure data is not a masked array. [Panu Lahtinen]
- Remove indexing which isn't used. [Panu Lahtinen]
- Unpack result one step further to get a float instead of ndarray.
  [Panu Lahtinen]
- Mask out warnings about invalid values in less and greater. [Panu
  Lahtinen]
- Documentation for pyresample.bilinear. [Panu Lahtinen]
- Add few tests for bilinear interpolation. [Panu Lahtinen]
- Fix typos, fix _get_ts_parallellogram() [Panu Lahtinen]
- Adjust comment. [Panu Lahtinen]
- Ignore messages about invalid values due to np.nan. [Panu Lahtinen]
- Handle cases with parallel sides in the rectangle formed by
  neighbours. [Panu Lahtinen]
- Make it possible to give input coordinates instead of area definition.
  [Panu Lahtinen]
- Fixes: check for # datasets, output shape for multiple datasets,
  masking, make output reshaping optional. [Panu Lahtinen]
- Add convenience function resample_bilinear(), remove unused logging.
  [Panu Lahtinen]
- Rename get_corner() as _get_corner() [Panu Lahtinen]
- Add better docstrings, rename helper functions private. [Panu
  Lahtinen]
- Cleanup code. [Panu Lahtinen]
- Extend docstrings, add a keyword to return masked arrays or arrays
  with np.nan:s. [Panu Lahtinen]
- Add default value for search radius, adjust default number of
  neighbours. [Panu Lahtinen]
- Initial version of bilinear resampling. [Panu Lahtinen]

  NOTE: Only works if both source and destination are area definitions.
  Also to be added is handling for the cases where a__ equals zero (use
  linear solution of bx + c = 0), testing, logging and all the error
  handling.

- Allow areas to be flipped. [Martin Raspaud]
- Factorize get_xy_from_lonlat and get_xy_from_proj_coords. [Martin
  Raspaud]
- Remove `fill_value` documentation for get_neighbour_info. [davidh-
  ssec]

  Fix #50



v1.2.9 (2016-12-13)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.8 → 1.2.9. [Martin Raspaud]
- Merge pull request #52 from mitkin/mitkin-pr-setuptools32. [Martin
  Raspaud]

  Specify minimum version of setuptools
- Specify minimum version of setuptools. [Mikhail Itkin]

  Prior to version 3.2 setuptools would not recognize correctly the language of `*.cpp` extensions and would assume it's `*.c` no matter what. Version 3.2 of setuptools fixes that.
- Fix sphinx dependency to support python 2.6 and 3.3. [Martin Raspaud]


v1.2.8 (2016-12-06)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.7 → 1.2.8. [Martin Raspaud]
- Correct style in setup.py. [Martin Raspaud]
- Make pykdtree a requirement. [Martin Raspaud]
- Correct style in geometry.py. [Martin Raspaud]
- Allow precision errors when comparing area_extents. [Martin Raspaud]
- Allow numbers in proj dict when building proj4 string. [Martin
  Raspaud]


v1.2.7 (2016-11-15)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.6 → 1.2.7. [Martin Raspaud]
- Add bump and changelog config files. [Martin Raspaud]
- Merge pull request #49 from Funkensieper/fix-polygon-area. [Martin
  Raspaud]

  Fix polygon area
- Disable snapping of angles in get_polygon_area() [Stephan
  Finkensieper]

  - Add option to disable snapping in Arc.angle()
  - Don't snap angles when computing polygon areas in order to
    prevent negative area values
  - Adjust reference values in tests on overlap-rate

- Fix polygon area computation for R != 1. [Stephan Finkensieper]

  Parentheses were missing, see

  http://mathworld.wolfram.com/SphericalTriangle.html

  for reference. Only affects earth radius R != 1 which is not
  implemented yet.

- Install pykdtree from conda forge in pre-master. [davidh-ssec]
- Merge pull request #47 from mitkin/feature_plot-cmap. [David Hoese]

  Add option to choose colormap
- Add option to choose colormap. [Mikhail Itkin]

  Make possible to indicate which colormap to use when plotting image



v1.2.6 (2016-10-19)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.5 → 1.2.6. [Martin Raspaud]
- Pre master (#44) [Radar, Satellite and Nowcasting Division]

  * add a subset function to the geometry file

  Often subsets of the SEVIRI disk are save in
  SEVIRI products. This new function calculated the
  area extent needed for the specification of the area,
  starting from the SEVIRI full disk area object.

  * added the get_area_extent_for_subset function

  to the geometry.py file

  * new function for getting area subsets

  * new function get_xy_from_proj_coordinates

  retrieve the pixel indices x and y
  from the map projection coordinates in meter
  (very similar to get_xy_from_lonlat)

  * removed pyc file, that should not be in the git repository

- Add appveyor status badge to README. [davidh-ssec]
- Merge remote-tracking branch 'deni90/master' into pre-master-davidh.
  [davidh-ssec]
- Fix test_custom_uncert and test_gauss_uncert for mips* [Daniel
  Knezevic]
- Fix pykdtree install on appveyor by turning off OpenMP. [davidh-ssec]
- Update appveyor config to install missing headers required by
  pykdtree. [davidh-ssec]
- Change appveyor to use conda-forge instead of IOOS. [davidh-ssec]
- Add slack notifications from appveyor. [davidh-ssec]


v1.2.5 (2016-07-21)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.4 → 1.2.5. [Martin Raspaud]
- Fix EWA fornav for integer data and add test. [davidh-ssec]

  There was a bug when doing the averaging in EWA where the current pixel was being rounded based on the previous pixel's value instead of the current pixel. This only affects integer data because values above 0 are rounded up by 0.5 and values below 0 are rounded by 0.5, for floats this round value is 0.0.

- Fix certain compilers not liking integers being passed to isnan.
  [davidh-ssec]
- Replace catch_warnings in all tests with astropy version. [davidh-
  ssec]
- Use catch_warnings from astropy (or at least try to) [davidh-ssec]
- Test removing version specific warning checks in `test_swath_wrap`
  [davidh-ssec]
- Move USE_CYTHON handling to if main block in setup.py. [davidh-ssec]
- Fix isnan definition only if a macro doesn't already exist. [davidh-
  ssec]

  Numpy does some special macro stuff to define a good npy_isnan function. Some systems define a macro for it, others don't. Hopefully this works for all systems. A better solution might be to define a templated isnan that calls npy_isnan if it isn't an integer.

- fix EWA compile failure on windows python 3.5. [David Hoese]
- Make pykdtree install on appveyor optional. [davidh-ssec]
- Add pykdtree to appveyor dependencies. [davidh-ssec]
- Fix setup.py test on windows for multiprocessing tests. [davidh-ssec]

  On Windows when new processes are started the initially command is imported or re-executed. For setup.py this is a big problem since the usual boilerplate does not include `if __name__ == "__main__"` so the setup.py test command gets rerun and rerun. This results in the child processes never actually being run for newer versions of python (2.7+). There still seems to be an issue with `test_nearest_resize` on Windows.

- Merge pull request #41 from cpaulik/fix-windows-ewa. [David Hoese]

  Fix Windows CI import Error
- Install scipy in Windows CI to fix import problems. [Christoph Paulik]
- Fix copy/paste error in EWA fornav. [davidh-ssec]

  I had started rewriting EWA in cython then realized it was faster in straight C++ so copied/pasted the cython code and modified it. Seems like I missed this 'or' hanging around.

- Fix NAN constant/macro for EWA on Windows. [davidh-ssec]
- Merge branch 'add-windows-CI' into fix-windows-ewa. [davidh-ssec]
- CI: Add IOOS conda channel to get basemap for Windows and python > 2.
  [Christoph Paulik]
- Merge branch 'add-windows-CI' into fix-windows-ewa. [davidh-ssec]
- Add pyproj to conda install in Appveyor CI. [Christoph Paulik]
- Make extra_compile_args platform dependent. [Christoph Paulik]
- Add Appveyor CI configuration. [Christoph Paulik]
- Fix EWA resampling's isnan to work better with windows. [davidh-ssec]


v1.2.4 (2016-06-27)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.3 → 1.2.4. [Martin Raspaud]
- Fix setup.py extension import and use error. [davidh-ssec]
- Fix case when __builtins__ is a dict. [Martin Raspaud]


v1.2.3 (2016-06-21)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.2 → 1.2.3. [Martin Raspaud]
- Fix list of package names in setup.py. [davidh-ssec]

  'pyresample.ewa' wasn't listed before and was not importable from an installed package.



v1.2.2 (2016-06-21)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.1 → 1.2.2. [Martin Raspaud]
- Add the header files to the MANIFEST.in. [Martin Raspaud]

  Without this, the compilation of the ewa extension crashes.


v1.2.1 (2016-06-21)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.2.0 → 1.2.1. [Martin Raspaud]
- Include EWA header files as dependency for extensions. [davidh-ssec]

  The .c and .cpp files are automatically included because they are listed as sources, but the header files are not. When building a source tarball (uploading to PyPI) the _fornav_templates.h file was not included and building would fail.

- Merge branch 'pre-master' of github.com:mraspaud/pyresample into pre-
  master. [Adam.Dybbroe]
- Merge branch 'pre-master' of github.com:mraspaud/pyresample into pre-
  master. [Adam.Dybbroe]

  Conflicts:
  	docs/source/conf.py

- Run the base class init function first. [Adam.Dybbroe]


v1.2.0 (2016-06-17)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.1.6 → 1.2.0. [Martin Raspaud]
- Merge branch 'northaholic-feature-lonlat2colrow' into pre-master.
  [Adam.Dybbroe]
- Add two convenience methods lonlat2colrow and colrow2lonlat to
  AreaDefinition-class. [Sauli Joro]
- Fix bug in EWA grid origin calculation. [davidh-ssec]

  Forgot that cell height was negative so ended up subtracting a negative, going in the wrong direction for the Y origin of the grid.

- Merge pull request #37 from davidh-ssec/feature-ewa-resampling. [David
  Hoese]

  Feature ewa resampling
- Fix bug in EWA conversion from AreaDefinition to upper-left origin
  X/Y. [davidh-ssec]

  I was using the area extent for the origin x/y locations, but the extent is actually the outer edge of the pixels so half a pixel needs to be added to each coordinate.

- Add EWA C extensions to mocked modules for read the docs. [davidh-
  ssec]

  Readthedocs.org fails to import the _ll2cr and _fornav extensions because it seems to not compile them properly. Their documentation isn't necessarily needed so I'm hoping that mocking them will let the import work.

- Add pyresample.ewa to API documentation list. [davidh-ssec]
- Update EWA wrapper functions to use explicit kwargs. [davidh-ssec]
- Correct comments and documentation in EWA documentation. [davidh-ssec]
- Add ll2cr and fornav wrappers to make calling easier. [davidh-ssec]

  Updated documentation with correct usage and added information why EWA is different than kdtree

- Fix print statements in documentation so doctests are python 3
  compatible. [davidh-ssec]
- Add pillow dependency for plot tests and quicklook extra. [davidh-
  ssec]
- Add 'areas.cfg' file to repository and modify doctests to use that
  instead. [davidh-ssec]
- Run doctests after unittests on travis. [davidh-ssec]
- Fix documentation for AreaDefinition object. [davidh-ssec]
- Update documentation to be numpy style and get rid of all warnings
  when building docs. [davidh-ssec]
- Create special requirements.txt for docs. [davidh-ssec]

  Readthedocs really doesn't like an empty string for the requirements file

- Try empty string for requirements file in readthedocs yaml. [davidh-
  ssec]
- Fix readthedocs yaml config file. [davidh-ssec]

  Readthedocs was using the requirements file during package installation, but was failing to install basemap (not needed for documentation build) so I attempted to make it an empty string in the yaml file. This makes Rtd hang on the build process. This should at least stop the hanging.

- Add napoleon docs extension and intial testing with numpy style
  docstrings. [davidh-ssec]
- Add working example for EWA resampling to docs. [davidh-ssec]

  I originally had this example but removed it when I had import problems. After I figured those out I forgot to put the original example back.

- Add basemap back in to the requirements.txt so that it can be
  installed on travis. [davidh-ssec]

  Similarly removed the requirements file when readthedocs is running and mocked third-party packages to documentation can still be built

- Fix setup.py requiring numpy for extension includes. [davidh-ssec]

  The EWA extensions require the numpy headers to be built. These are normally found by importing numpy and doing `numpy.get_includes()`. Obviously if this is run on a new environment numpy is probably not installed so a simple `python setup.py install` will fail.

- Add "quicklook" extra in to travis test install. [davidh-ssec]

  These packages are needed to properly test the "plot" package. These were included in requirements.txt but have been moved for now.

- Move plot test imports in to test functions for cleaner test failures.
  [davidh-ssec]
- Add readthedocs yaml file for configuration. [davidh-ssec]
- Remove mocked modules from sphinx docs conf.py. [davidh-ssec]

  This is the first step in making pyresamples docs buildable in the current readthedocs version

- Replace relative imports with absolute imports. [davidh-ssec]

  I noticed a lot of warnings and import problems with building pyresample's documentation because of these relative imports

- Add EWA documentation to swath.rst. [davidh-ssec]
- Add tests for EWA fornav module. [davidh-ssec]
- Update documentation for ll2cr and fornav cython. [davidh-ssec]
- Merge remote-tracking branch 'davidh_fork/feature-ewa-resampling' into
  feature-ewa-resampling. [davidh-ssec]

  # Conflicts:
  #	pyresample/ewa/_fornav.pyx
  #	pyresample/ewa/_ll2cr.pyx

- Remove old and unused polar2grid ll2cr and fornav python modules.
  [davidh-ssec]
- Fix travis tests on python 2.6. [davidh-ssec]
- Add ewa ll2cr tests to main test suite. [davidh-ssec]
- Add simple tests for ewa ll2cr. [davidh-ssec]

  These tests were adapted from Polar2Grid so some of the terminology or organization might reflect P2G's design rather than satpy or pyresample.

- Revert import multiprocessing setup.py for python 2.6 compatibility.
  [davidh-ssec]
- Fix old polar2grid import in ll2cr module. [davidh-ssec]
- Add method for converting area def to areas.def string format.
  [davidh-ssec]
- Remove unused code from fornav wrapper. [davidh-ssec]
- Add initial EWA files copied from Polar2Grid. [davidh-ssec]
- Add basic documentation to fornav cython function. [davidh-ssec]
- Remove old and unused polar2grid ll2cr and fornav python modules.
  [davidh-ssec]
- Fix travis tests on python 2.6. [davidh-ssec]
- Add ewa ll2cr tests to main test suite. [davidh-ssec]
- Add simple tests for ewa ll2cr. [davidh-ssec]

  These tests were adapted from Polar2Grid so some of the terminology or organization might reflect P2G's design rather than satpy or pyresample.

- Revert import multiprocessing setup.py for python 2.6 compatibility.
  [davidh-ssec]
- Fix old polar2grid import in ll2cr module. [davidh-ssec]
- Add method for converting area def to areas.def string format.
  [davidh-ssec]
- Remove unused code from fornav wrapper. [davidh-ssec]
- Add initial EWA files copied from Polar2Grid. [davidh-ssec]
- Add .gitignore with python and C patterns. [davidh-ssec]
- Update tests so they don't fail on OSX. [davidh-ssec]

  OSX seems to calculate slightly different results from `_spatial_mp.Cartesian` regardless of numexpr being installed. Although the changes are small they seem to affect the results enough to fail this test compared to normal linux execution.

- Add 'load_tests' for easier test selection. [davidh-ssec]

  PyCharm and possibly other IDEs don't really play well with unittest TestSuites, but work as expected when `load_tests` is used.

- Make kd_tree test work on older numpy version. [Martin Raspaud]

  VisibleDeprecationWarning is not available in numpy <1.9.
- Adapt to newest pykdtree version. [Martin Raspaud]

  The kdtree object's attribute `data_pts` has been renamed to `data`.
- Run tests on python 3.5 in travis also. [Martin Raspaud]


v1.1.6 (2016-02-25)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.1.5 → 1.1.6. [Martin Raspaud]
- Fix #35 supporting scipy kdtree again. [Martin Raspaud]

  A previous commit was looking for a 'data_pts' attribute in the kdtree
  object, which is available in pykdtree, but not scipy.
- Merge pull request #32 from mitkin/master. [Martin Raspaud]

  [tests] Skip deprecation warnings in test_gauss_multi_uncert
- Merge remote-tracking branch 'gh-pytroll/pre-master' [Mikhail Itkin]
- Put quotes around pip version specifiers to make things work. [Martin
  Raspaud]
- Install the right matplotlib in travis. [Martin Raspaud]

  The latest matplotlib (1.5) doesn't support python 2.6 and 3.3. This patch
  chooses the right matplotlib version to install depending on the python
  version at hand.
- Skip deprecation warnings. [Mikhail Itkin]

  Catch the rest of the warnings. Check if there is only one, and
  whether it contains the relevant message ('possible more than 8
  neighbours found'). This patch is necessary for python 2.7.9 and newer

- Merge pull request #31 from bhawkins/fix-kdtree-dtype. [Martin
  Raspaud]

  Fix possible type mismatch with pykdtree.
- Add test to expose pykdtree TypeError exception. [Brian Hawkins]
- Fix possible type mismatch with pykdtree. [Brian Hawkins]


v1.1.5 (2015-10-12)
-------------------
- update changelog. [Martin Raspaud]
- Bump version: 1.1.4 → 1.1.5. [Martin Raspaud]
- Don't build on 3.2 anymore (because of coverage's lack of support for
  3.2). [Martin Raspaud]
- Fix build badge adress. [Martin Raspaud]
- Fix the unicode problem in python3. [Martin Raspaud]


v1.1.4 (2015-10-08)
-------------------

Fix
~~~
- Bugfix: Accept unicode proj4 strings. Fixes #24. [Martin Raspaud]

Other
~~~~~
- update changelog. [Martin Raspaud]
- Bump version: 1.1.3 → 1.1.4. [Martin Raspaud]
- Add python-configobj as a rpm requirement in setup.cfg. [Martin
  Raspaud]
- Add setup.cfg to allow rpm generation with bdist_rpm. [Martin Raspaud]
- Bugfix to address a numpy DeprecationWarning. [Martin Raspaud]

  Numpy won't take non-integer indices soon, so make index an int.
- Merge branch 'release-1.1.3' [Martin Raspaud]
- Merge branch 'licence-lgpl' into pre-master. [Martin Raspaud]
- Switch to lgplv3, and bump up version number. [Martin Raspaud]
- Swith badge to main repository. [Martin Raspaud]
- Merge branch 'hotfix-v1.1.2' into pre-master. [Martin Raspaud]
- Merge branch 'hotfix-v1.1.2' [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Merge branch 'mitkin-master' into hotfix-v1.1.2. [Martin Raspaud]
- Merge branch 'master' of https://github.com/mitkin/pyresample into
  mitkin-master. [Martin Raspaud]
- [test_plot] allow travis to test plot.py. [Mikhail Itkin]
- [pip+travis] use `requirements.txt` [Mikhail Itkin]

  Use `requirements.txt` instead of setuptools' `extras_require`
  for installing basemap.

  That is because PyPi basemap version won't find libgeos library
  so we resolve to use latest basemap from git. `Extras_require` don't
  allow providing custom links, only PyPi package names, so we have to
  specify links in requirements.txt. `dependency_links` argument to
  `setup` call is meant for cruicial dependencies, not custom ones, so we
  don't use them neither.

- [README] markdown + build status. [Mikhail Itkin]

   * Using markdown extension, added `README` symlink
   * Added travis build status badge

- remove pip `-e` switch. [Mikhail Itkin]
- Merge branch 'master' of github.com:mitkin/pyresample. [Mikhail Itkin]
- don't use setup.py for basemap installation. [Mikhail Itkin]

  Instead of putting basemap and matplotlib into `extras_require`
  install them directly
- don't use setup.py for basemap installation. [Mikhail Itkin]

  Instead of putting basemap and matplotlib into `extras_require`
  install them directly

- Using ubuntu GIS custom ppa. [Mikhail Itkin]

  Added custom ppa with more up-to-date libgeos dependencies
- Install extra requirements using pip functionality. [Mikhail Itkin]
- Added more meaningful "quicklooks" name. [Mikhail Itkin]

  Using quicklooks name as it's what matplotlib and basemap are needed for
- [setup] added plotting dependencies. [Mikhail Itkin]

  pyresample/plot requires two extra dependencies:
   * matplotlib
   * basemap

- [travis] added system dependencies. [Mikhail Itkin]

   * matplotlib requires libfreetype6-dev
   * basemap requires libgeos libgeos-c1 and libgeos-dev

- Merge branch 'release-v1.1.1' [Martin Raspaud]
- Merge branch 'release-v1.1.1' [Martin Raspaud]
- Restore API functionality by importing necessary modules in __init__
  [Martin Raspaud]
- Merge branch 'release-v1.1.1' into pre-master. [Martin Raspaud]

  Conflicts:
  	pyresample/geometry.py
  	pyresample/kd_tree.py
  	test/test_geometry.py

- Removing old test directory. [Martin Raspaud]
- Merge the hotfix and the unittest restructuring into the release
  branch. [Martin Raspaud]
- Merge branch 'release-v1.1.1' into hotfix-1.1.1. [Thomas Lavergne]

  Conflicts:
  	pyresample/geometry.py
  	test/test_geometry.py
  	test/test_grid.py

- Be specific about the valid range of longitudes. [Thomas Lavergne]
- Be more specific about the valid longitude range [-180:+180[. Add a
  test for utils.wrap_longitudes() [Thomas Lavergne]
- Add check on valid latitude in [-90:+90] (and associated test) [Thomas
  Lavergne]
- Automatic longitude wrapping (bugfix towards 1.1.1) [Thomas Lavergne]
- Merge branch 'release-v1.1.1' into pre-master. [Martin Raspaud]
- Add news about new release. [Martin Raspaud]
- remove some relative imports. [Martin Raspaud]
- Cleanup and bump up version number to v1.1.1. [Martin Raspaud]
- Add pykdtree to the list of requirements for travis. [Martin Raspaud]
- Add .travis.yml file for automatic testing. [Martin Raspaud]
- Correct handling of long type in kd_tree.py for Python 2. [Martin
  Valgur]
- Made testing of a Proj4 string independent of the order of elements
  inside the string since the order was different on Python 2 and 3.
  Replaced deprecated failIf with assertFalse. [Martin Valgur]
- Multiple small fixes to make the code work on both Python 2 and 3.
  shmem_as_ndarray() now uses numpy.frombuffer() to provide equivalent
  functionality. [Martin Valgur]
- Got rid of dependencies on the six package. [Martin Valgur]
- Applied python-modernize to pyresample. [Martin Valgur]
- Update README. [Martin Raspaud]
- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]
- A stray line of code is removed and I take back the recent enhancement
  concerning swath to swath mapping. [Adam Dybbroe]
- Removed debug printouts. [Adam Dybbroe]
- More active support of swath to swath reprojection. [Adam Dybbroe]
- Add a plot on multiprocessing performance increases. [Martin Raspaud]
- Added outer_boundary_corners property to the area def class. [Adam
  Dybbroe]
- corrected docs. [Esben S. Nielsen]
- modified uncert count to show above 0. Updated docs to relect uncert
  option. [Esben S. Nielsen]
- cleaned up code a bit in kd_tree.py. [Esben S. Nielsen]
- made API doc work with readthedocs and bumped version number. [Esben
  S. Nielsen]
- cleaned up code and tests. [Esben S. Nielsen]
- added masking of uncert counts. [Esben S. Nielsen]
- test passes again for uncertainty calculations. [Esben S. Nielsen]
- changed uncertainty API. First working uncertainty version. [Esben S.
  Nielsen]
- not quite there. [Esben S. Nielsen]
- basic uncertainty implemented. [Esben S. Nielsen]
- updated docs. [Esben S. Nielsen]
- Fixing bug, and adding unittest-main run. [Adam Dybbroe]
- Making get_xy_from_lonlat work on arrays of points as well as single
  points. [Adam Dybbroe]
- renamed functions in geometry.py and added proj_x_coords and
  proj_y_coords properties. [Esben S. Nielsen]
- corrected __eq__ in geometry. [Esben S. Nielsen]
- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Adam Dybbroe]
- now kd_tree resampling selects dtype. [Esben S. Nielsen]
- removed random print statement. [Esben S. Nielsen]
- made get_capabilites function. [Esben S. Nielsen]
- test passes again. [Esben S. Nielsen]
- removed caching from geometry. [Esben S. Nielsen]
- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]
- Optimize transform_lonlats with numexpr. [Martin Raspaud]
- Unittests should work for both py2.6 and 2.7. [Adam Dybbroe]
- updated docs. [Esben S. Nielsen]
- fixed unit tests. [Esben S. Nielsen]
- Using assertRaises in py2.6 and py2.7 compatible version. [Adam
  Dybbroe]
- bugfix to unittest suite. [Adam Dybbroe]
- Trying to make test-functions compatible with both python 2.6 and 2.7.
  [Adam Dybbroe]
- Fixing bug in get_xy_from_lonlat and adding unittests on this
  function. [Adam Dybbroe]
- Adding function get_xy_from_lonlat. [Adam Dybbroe]
- integrated pykdtree and handled latlong projection bug. [Esben S.
  Nielsen]
- updated unit tests according to deprecation warnings. [Esben S.
  Nielsen]
- Better parsing of a area definition (allow ':' in value fields) [Lars
  Orum Rasmussen]
- updated docs. [Esben S. Nielsen]
- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]
- doc version. [esn]
- improved Basemap integration with globe projections. Updated docs on
  epsilon. [esn]
- Accomodate for allclose behaviour change in numpy 1.6.2. [Martin
  Raspaud]

  From 1.6.2 numpy.allclose does not accept arrays that cannot be
  broadcasted to the same shape. Hence a ValueError catch to return False.

- updadet doc for plotting. [Esben S. Nielsen]
- updated plot test to use AGG. [Esben S. Nielsen]
- Now handles plotting in Plate Carre projection. Added utils.fwhm2sigma
  function. [Esben S. Nielsen]
- Merge branch 'master' of https://code.google.com/p/pyresample. [Esben
  S. Nielsen]
- added pypi info. [Esben S. Nielsen]
- built docs. [Esben S. Nielsen]
- corrected test_swath.py to account for implementation specific
  precision. [Esben S. Nielsen]
- more datatype specifications. [Esben S. Nielsen]
- removed warning check for python 2.5. [Esben S. Nielsen]
- corrected multi channnel bug. Added warnings for potential problematic
  neighbour query condition. [Esben S. Nielsen]
- Now str() generates a unique string for area and coordinate definition
  object. [Lars Orum Rasmussen]
- corrected manifest so doc images are included. [Esben S. Nielsen]
- Moved tests dir to test. Updated MANIFEST.in. [Esben S. Nielsen]
- Added MANIFEST.in. [Esben S. Nielsen]
- Applied setup.py patches. Made plotting more robust. [Esben S.
  Nielsen]
- applied patch for getting version number. [Esben S. Nielsen]
- Bugfixing quicklooks. [StorPipfugl]
- Updated docs. [StorPipfugl]
- Updated docs. [StorPipfugl]
- Updated docs. [StorPipfugl]
- Added Basemap integration. [StorPipfugl]
- Added Basemap integration. [StorPipfugl]
- Updated docs. [StorPipfugl]
- Rebuild docs. [StorPipfugl]
- Made setup.py more robust. [StorPipfugl]
- New doc version. [StorPipfugl]
- Updated tests. [StorPipfugl]
- Reduced size of linesample arrays. Restructures kd_tree query to
  remove redundant lon lat calculations. [StorPipfugl]
- Added geographic filtering. Swaths can now be concatenated and
  appended. User no langer have to ravel data before resampling.
  [StorPipfugl]
- Updated docs. [StorPipfugl]
- Updated install_requires. [StorPipfugl]
- version 0.7.3. [StorPipfugl]
- Bugfixes: Correct number of channels in empty result set. Resampling
  of masked data to 1d swath now works. [StorPipfugl]
- Added Martin's spherical geometry operations. Updated documentation.
  [StorPipfugl]
- Added equal and not equal operators for geometry defs. Restructured
  the geometry module to be pickable. Added correct handling of empty
  result data sets. [StorPipfugl]
- Incomplete - taskpyresample. [StorPipfugl]
- Set svn:mime-type. [StorPipfugl]
- Corrected doc errors. [StorPipfugl]
- Removed dist dir. [StorPipfugl]
- No commit message. [StorPipfugl]
- Updated documentation. New release. [StorPipfugl]
- Started updating docstrings. [StorPipfugl]
- Restructured API. [StorPipfugl]
- Now uses geometry types. Introduced API symmetry between swath->grid
  and grid->swath resampling. [StorPipfugl]
- Consolidated version tag. [StorPipfugl]
- Mime types set. [StorPipfugl]
- Mime types set. [StorPipfugl]
- Removed test. [StorPipfugl]
- Removed unneeded function. [StorPipfugl]
- Mime types set. [StorPipfugl]
- Mime types set. [StorPipfugl]
- No commit message. [StorPipfugl]
- Moved to Google Code under GPLv3 license. [StorPipfugl]
- moved to Google Code. [StorPipfugl]



